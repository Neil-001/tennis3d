"""Local app: compact export, installed weights, run lifecycle and HTTP safety."""

import http.client
import json
import subprocess
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
av = pytest.importorskip("av")

from tennis3d.app import capabilities, home, runs
from tennis3d.app.gallery import Gallery
from tennis3d.app.media import playback_copy
from tennis3d.app.runs import AlreadyRunning, RunError, RunStore
from tennis3d.app.server import App
from tennis3d.artifacts import file_sha256, json_sha256, read_json, write_json
from tennis3d.cli import build_parser, main
from tennis3d.config import load_config
from tennis3d.inference.settings import ReconstructionConfig
from tennis3d.lifting.checkpoint import load_lifter
from tennis3d.lifting.export import (
    export_lifter,
    manifest_path,
    qualify_export,
    verify_export,
)
from tennis3d.lifting.model import Lifter, ModelConfig
from tennis3d.perception.assets import MissingAssetError
from tennis3d.video.decode import decode

RESULT = Path(__file__).parent / "fixtures" / "trajectory.json"
HEADERS = {"X-Tennis3D": "1"}
BOUND_PATHS = ("result_path", "timeline_path")
# Stands in for the inference process: reports progress, then waits or finishes.
STUB = """
import shutil, sys, time
from pathlib import Path
from tennis3d.artifacts import read_json, write_json
directory = Path(sys.argv[1])
write_json(directory / "progress.json", {"stage": "detect", "progress": {"completed": 1, "total": 4, "unit": "frames"}}, overwrite=True)
while (directory.parent.parent / "hold").exists():
    time.sleep(0.02)
if (directory.parent.parent / "fail").exists():
    write_json(directory / "outcome.json", {"summary": {"status": "failed", "exit_code": 2, "frame_count": 0}, "produced_result": False}, overwrite=True)
    raise SystemExit(2)
(directory / "inference").mkdir(exist_ok=True)
shutil.copy(sys.argv[2], directory / "inference" / "result.json")
write_json(directory / "outcome.json", {"summary": {"status": "partial", "exit_code": 0}, "produced_result": True}, overwrite=True)
"""


def training_checkpoint(path):
    torch.manual_seed(3)
    config = ModelConfig(width=16, depth=2)
    model = Lifter(config)
    torch.save(
        {
            "schema_version": "1.0",
            "world_scale_m": 10.0,
            "step": 7,
            "run_id": "fixture",
            "config": {"model": config.model_dump(), "train_manifest": "/private/x"},
            "ema": {f"module.{k}": v for k, v in model.state_dict().items()}
            | {"n_averaged": torch.tensor(7)},
            "optimizer": {"state": {0: torch.ones(3)}},
            "rng": {"torch": torch.get_rng_state()},
        },
        path,
    )
    return path


def write_video(path, pts=(0, 40, 80, 130, 170, 200), codec="libx264"):
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=25)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.codec_context.time_base = Fraction(1, 1000)
        for index, value in enumerate(pts):
            image = np.full((48, 64, 3), index * 30, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts, frame.time_base = value, Fraction(1, 1000)
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return path


def test_compact_export_is_minimal_equal_and_integrity_checked(tmp_path):
    source = training_checkpoint(tmp_path / "best.pt")
    output = tmp_path / "export.pt"
    manifest = export_lifter(source, output)
    state = torch.load(output, weights_only=True)
    assert set(state) == {"schema_version", "world_scale_m", "config", "ema"}
    assert set(state["config"]) == {"model"} and "n_averaged" not in state["ema"]
    assert "private" not in output.read_bytes().decode("latin1")
    assert manifest["source_checkpoint_sha256"] == file_sha256(source)
    assert manifest["sha256"] == file_sha256(output) and manifest["source_step"] == 7
    assert qualify_export(source, output)["prediction_max_abs_difference"] == 0.0
    assert load_lifter(output)[1].get("step") is None
    with pytest.raises(OSError):
        export_lifter(source, output)
    with pytest.raises(ValueError, match="differ"):
        export_lifter(source, source, overwrite=True)
    assert load_lifter(source)[1]["step"] == 7

    tampered = dict(state, ema={k: v + 1 for k, v in state["ema"].items()})
    torch.save(tampered, output)
    with pytest.raises(ValueError, match="SHA-256"):
        verify_export(output)
    # A re-signed manifest still cannot pass off different weights as the source.
    forged = manifest | {
        "sha256": file_sha256(output),
        "bytes": output.stat().st_size,
    }
    with pytest.raises(ValueError, match="weights do not match"):
        verify_export(output, forged)
    write_json(
        manifest_path(output),
        forged | {"tensor_sha256": "0" * 64},
        overwrite=True,
    )
    with pytest.raises(ValueError):
        qualify_export(source, output)


@pytest.fixture
def weights(tmp_path, monkeypatch):
    """Packaged manifests that pin three small local files."""
    store = tmp_path / "downloads"
    store.mkdir()
    files = {name: store / name for name in ("tracknet.pth", "court.pth", "lifter.pt")}
    for name, path in files.items():
        path.write_bytes(name.encode() * 10)
    assets = {
        "version": 1,
        "assets": {
            key: {
                "url": "https://invalid.example/weights",
                "path": f"checkpoints/perception/{name}",
                "sha256": file_sha256(files[name]),
                "bytes": files[name].stat().st_size,
                "source": "fixture",
                "revision": "0",
                "license": "fixture",
                "purpose": "fixture",
            }
            for key, name in (("ball", "tracknet.pth"), ("court", "court.pth"))
        },
    }
    lifter = {
        "file": "lifter.pt",
        "sha256": file_sha256(files["lifter.pt"]),
        "bytes": files["lifter.pt"].stat().st_size,
        "url": "https://invalid.example/lifter.pt",
        "model_label": "Fixture lifter",
    }
    packaged = home.resource
    documents = {"assets.yaml": json.dumps(assets), "lifter.json": json.dumps(lifter)}
    monkeypatch.setattr(
        home,
        "resource",
        lambda name: documents[name].encode() if name in documents else packaged(name),
    )
    return store


def test_setup_imports_pinned_weights_and_config_ignores_working_directory(
    tmp_path, weights, monkeypatch
):
    root = tmp_path / "home"
    with pytest.raises(MissingAssetError):
        home.setup(root, offline=True)
    assert any("tennis3d setup" in problem for problem in home.problems(root))
    (weights / "court.pth").write_bytes(b"damaged")
    with pytest.raises(MissingAssetError):
        home.setup(root, weights=[weights], offline=True)
    (weights / "court.pth").write_bytes(b"court.pth" * 10)
    report = home.setup(root, weights=[weights], offline=True)
    assert set(report["weights"].values()) == {"imported", "present"}
    assert set(home.setup(root, offline=True)["weights"].values()) == {"present"}
    assert home.problems(root) == []

    monkeypatch.chdir(tmp_path)
    config = load_config(
        ReconstructionConfig, overrides=home.installed_config(root, "cpu")
    )
    assert config.lifting.checkpoint == root / "weights" / "lifter.pt"
    assert config.detection.assets_manifest.is_absolute()
    assert config.lifting.model_label == "Fixture lifter"
    assert config.detection.device == config.lifting.device == "cpu"
    (root / "weights" / "lifter.pt").write_bytes(b"changed")
    assert home.problems(root)


def test_playback_copy_keeps_source_times_and_reuses_identity(tmp_path):
    source = write_video(tmp_path / "source.mp4")
    receipt = playback_copy(source, tmp_path / "playback.mp4")
    original = list(decode(source))
    copied = list(decode(tmp_path / "playback.mp4"))
    assert [f.t_s for f in copied] == [f.t_s for f in original]
    assert [row["source_pts"] for row in receipt["frames"]] == [f.pts for f in original]
    assert [row["elapsed_s"] for row in receipt["frames"]] == [f.t_s for f in original]
    assert receipt["source_sha256"] == file_sha256(source)
    modified = (tmp_path / "playback.mp4").stat().st_mtime_ns
    assert playback_copy(source, tmp_path / "playback.mp4") == receipt
    assert (tmp_path / "playback.mp4").stat().st_mtime_ns == modified
    (tmp_path / "broken.mp4").write_bytes(b"not a video")
    with pytest.raises((ValueError, OSError)):
        playback_copy(tmp_path / "broken.mp4", tmp_path / "none.mp4")
    assert not (tmp_path / "none.mp4").exists()


class Client:
    def __init__(self, server):
        self.port = server.server_port

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def json(self, method, path, body=None, headers=HEADERS):
        status, _, payload = self.request(method, path, body, headers)
        return status, json.loads(payload)

    def wait(self, run_id, *statuses):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            run = self.json("GET", f"/api/runs/{run_id}")[1]
            if run["status"] in statuses:
                return run
            time.sleep(0.02)
        raise AssertionError(f"run stayed {run['status']}")


@pytest.fixture
def app(tmp_path, weights, monkeypatch):
    monkeypatch.setattr(
        capabilities, "bounded_probe", lambda: capabilities.unavailable("no-gpu")
    )
    root = tmp_path / "home"
    home.setup(root, weights=[weights], offline=True)
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    monkeypatch.setattr(
        RunStore,
        "worker_command",
        lambda self, path: [sys.executable, str(stub), str(path), str(RESULT)],
    )
    gallery = tmp_path / "gallery.json"
    write_json(
        gallery,
        {
            "schema_version": 1,
            "clips": [
                {
                    "id": "clip-a",
                    "title": "Example A",
                    "note": "partial: 9 of 12 source frames have exported XYZ",
                    "source_path": str(write_video(tmp_path / "clip-a.mp4")),
                    "source_sha256": file_sha256(tmp_path / "clip-a.mp4"),
                    "result_path": str(RESULT),
                    "result_sha256": file_sha256(RESULT),
                    "rights": {"redistribution": "unknown"},
                },
                {
                    "id": "clip-b",
                    "title": "Changed result",
                    "note": "ok: 12 of 12 source frames have exported XYZ",
                    "source_path": str(tmp_path / "absent.mp4"),
                    "source_sha256": "0" * 64,
                    "result_path": str(RESULT),
                    "result_sha256": "1" * 64,
                },
            ],
        },
    )
    server = App(root, 0, gallery)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, Client(server), tmp_path
    server.shutdown()
    server.server_close()


def test_device_validation_applies_to_upload_start_and_resume(app):
    server, client, tmp_path = app
    video = (tmp_path / "clip-a.mp4").read_bytes()
    for device in ("unknown", "", "intel-gpu", "npu"):
        assert (
            client.json("POST", f"/api/runs?name=a.mp4&device={device}", video)[0]
            == 400
        )
    assert server.runs.list() == []
    hold = server.root / "hold"
    hold.touch()
    run = client.json("POST", "/api/runs?name=a.mp4&device=cpu", video)[1]
    stored = server.runs.path(run["id"])
    original_request = (stored / "request.json").read_bytes()
    # Invalid overrides also fail for active runs without altering their config.
    for device in ("npu", "", "intel-gpu"):
        assert (
            client.json("POST", f"/api/runs/{run['id']}/start?device={device}")[0]
            == 400
        )
        assert (stored / "request.json").read_bytes() == original_request
    client.json("POST", f"/api/runs/{run['id']}/cancel")
    assert (
        client.json("POST", f"/api/runs/{run['id']}/start?device=intel-gpu")[0] == 400
    )
    assert server.runs.get(run["id"])["status"] == "cancelled"
    assert client.json("POST", f"/api/runs/{run['id']}/start?device=auto")[0] == 200
    config = read_json(stored / "request.json")["config"]
    assert config["detection"]["device"] == config["lifting"]["device"] == "auto"


def test_intel_capability_gates_create_and_resume_and_maps_cpu_lifting(app):
    server, client, tmp_path = app
    server.runs.intel.thread.join(timeout=5)
    server.runs.intel.report = {"status": "available", "device": "GPU"}
    assert client.json("GET", "/api/state")[1]["intel_gpu"]["status"] == "available"
    (server.root / "hold").touch()
    video = (tmp_path / "clip-a.mp4").read_bytes()
    status, run = client.json("POST", "/api/runs?name=a.mp4&device=intel-gpu", video)
    assert status == 201 and run["device"] == "intel-gpu"
    config = read_json(server.runs.path(run["id"]) / "request.json")["config"]
    assert config["detection"]["device"] == "intel-gpu"
    assert config["lifting"]["device"] == "cpu"
    client.json("POST", f"/api/runs/{run['id']}/cancel")
    server.runs.intel.report = capabilities.unavailable("runtime-check-failed")
    assert client.json("POST", f"/api/runs/{run['id']}/start")[0] == 400
    assert server.runs.get(run["id"])["device"] == "intel-gpu"
    with pytest.raises(RunError, match="unavailable"):
        server.runs.start(run["id"])
    assert client.json("POST", f"/api/runs/{run['id']}/start?device=cpu")[0] == 200
    assert server.runs.get(run["id"])["device"] == "cpu"


def test_intel_watchdog_stops_compile_stall_and_queue_continues(app, monkeypatch):
    server, client, tmp_path = app
    monkeypatch.setattr(runs, "INTEL_IDLE_TIMEOUT_S", 0.15)
    server.runs.intel.thread.join(timeout=5)
    server.runs.intel.report = {"status": "available", "device": "GPU"}
    (server.root / "hold").touch()
    video = (tmp_path / "clip-a.mp4").read_bytes()
    run = client.json("POST", "/api/runs?name=stall.mp4&device=intel-gpu", video)[1]
    failed = client.wait(run["id"], "failed")
    assert failed["exit_code"] == 3 and failed["device"] == "intel-gpu"
    assert "without progress" in failed["error"] and not failed["has_result"]
    assert (
        read_json(server.runs.path(run["id"]) / "outcome.json")["error_type"]
        == "TimeoutError"
    )
    (server.root / "hold").unlink()
    cpu = client.json("POST", "/api/runs?name=next.mp4&device=cpu", video)[1]
    assert client.wait(cpu["id"], "complete")["has_result"]


def test_stalled_intel_worker_exits_after_outcome_without_native_teardown(tmp_path):
    script = """
import atexit, sys, time
from pathlib import Path
from types import SimpleNamespace
from tennis3d.app import worker
from tennis3d.artifacts import write_json
directory = Path(sys.argv[1])
def fail(directory):
    write_json(directory / 'outcome.json', {'error': 'mock native timeout'})
    return 3
worker.main = fail
sys.modules['tennis3d.perception.accelerate'] = SimpleNamespace(STALLED=[object()])
# Represents native-request destruction that would otherwise wait forever.
atexit.register(lambda: time.sleep(60))
raise SystemExit(worker.process_main(directory))
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert process.returncode == 3
    assert read_json(tmp_path / "outcome.json")["error"] == "mock native timeout"


@pytest.mark.parametrize("mode", ["normal", "invalid-after-detection"])
def test_app_worker_detector_stall_preserves_failure_without_native_destruction(
    tmp_path, weights, mode
):
    """Real app main/pipeline/detector with a mock device and no lifting."""
    root = tmp_path / "stall-home"
    home.setup(root, weights=[weights], offline=True)
    directory = root / "runs" / ("a" * 16)
    directory.mkdir(parents=True)
    source = write_video(directory / "source.mp4")
    config = home.installed_config(root, "intel-gpu")
    config["detection"]["threads"] = config["lifting"]["threads"] = 2
    write_json(
        directory / "request.json",
        {
            "source_name": source.name,
            "config": config,
        },
    )
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "fixtures/mock_intel_stall_worker.py"),
            str(directory),
            mode,
        ],
        capture_output=True,
        timeout=12,
        check=False,
    )
    assert process.returncode == 3, process.stderr.decode()
    assert not (directory / "native-destruction-entered").exists()
    failed = read_json(directory / "inference/detect.json")
    assert failed["status"] == "failed" and failed["exit_code"] == 3
    assert failed["frame_count"] == 4 and failed["failure_kind"] == "runtime"
    assert "stalled" in failed["failure"]
    assert failed["source_sha256"] == file_sha256(source)
    with np.load(directory / "inference/frames.npz") as frames:
        original = list(decode(source))[:4]
        assert frames["pts"].tolist() == [frame.pts for frame in original]
        assert frames["time_base"].tolist() == [
            list(frame.time_base.as_integer_ratio()) for frame in original
        ]
    assert not (directory / "inference/result.json").exists()
    assert (
        "published detector failure" in read_json(directory / "outcome.json")["error"]
    )


@pytest.mark.parametrize("prefix_published", [False, True])
def test_app_parent_records_forced_detector_exit_and_keeps_previous_result(
    app, monkeypatch, prefix_published
):
    server, client, tmp_path = app
    monkeypatch.setattr(home.os, "cpu_count", lambda: 4)
    video = (tmp_path / "clip-a.mp4").read_bytes()
    run = client.json("POST", "/api/runs?name=existing.mp4&device=cpu", video)[1]
    assert client.wait(run["id"], "complete")["has_result"]
    path = server.runs.path(run["id"])
    previous = (path / "inference/result.json").read_bytes()
    write_json(path / "inference/export.json", {"status": "partial"})
    # A committed earlier detection with different settings causes a new
    # attempt, while its committed export remains available to the parent.
    write_json(path / "inference/detect.config.json", {})
    write_json(path / "inference/detect.json", {"config_sha256": json_sha256({})})
    server.runs.intel.thread.join(timeout=5)
    server.runs.intel.report = {"status": "available", "device": "GPU"}
    stub_command = server.runs.worker_command
    monkeypatch.setattr(
        server.runs,
        "worker_command",
        lambda path: [
            sys.executable,
            str(Path(__file__).parent / "fixtures/mock_intel_stall_worker.py"),
            str(path),
            "force-deadline-after-prefix" if prefix_published else "force-deadline",
            str(RESULT),
        ],
    )
    client.json("POST", f"/api/runs/{run['id']}/start?device=intel-gpu")
    failed = client.wait(run["id"], "failed")
    assert failed["exit_code"] == 1 and failed["has_result"]
    assert failed["summary"]["frame_count"] == 4
    assert failed["summary"]["failure_kind"] == "runtime"
    assert "Intel GPU detection failed" in failed["error"]
    assert (
        "this attempt's decoded prefix"
        if prefix_published
        else "earlier result is still shown"
    ) in failed["error"]
    assert (path / "outcome.json").exists() == prefix_published
    assert (path / "inference/frames.npz").is_file()
    assert (path / "inference/result.json").read_bytes() == previous
    monkeypatch.setattr(server.runs, "worker_command", stub_command)
    next_run = client.json("POST", "/api/runs?name=next.mp4&device=cpu", video)[1]
    assert client.wait(next_run["id"], "complete")["has_result"]


@pytest.mark.parametrize("returned_code", [0, 3])
def test_app_stalled_exported_prefix_remains_failed_and_viewable(
    app, monkeypatch, returned_code
):
    server, client, tmp_path = app
    server.runs.intel.thread.join(timeout=5)
    server.runs.intel.report = {"status": "available", "device": "GPU"}
    original_command = server.runs.worker_command
    script = """
import shutil, sys
from pathlib import Path
from types import SimpleNamespace
from tennis3d.app import worker
from tennis3d.artifacts import write_json
directory = Path(sys.argv[1])
def exported_prefix(directory):
    (directory / 'inference').mkdir(exist_ok=True)
    shutil.copy(sys.argv[2], directory / 'inference/result.json')
    write_json(directory / 'outcome.json', {'summary': {'status': 'partial', 'exit_code': int(sys.argv[3])}, 'produced_result': True})
    return int(sys.argv[3])
worker.main = exported_prefix
sys.modules['tennis3d.perception.accelerate'] = SimpleNamespace(STALLED=[object()])
raise SystemExit(worker.process_main(directory))
"""
    monkeypatch.setattr(
        server.runs,
        "worker_command",
        lambda path: [
            sys.executable,
            "-c",
            script,
            str(path),
            str(RESULT),
            str(returned_code),
        ],
    )
    video = (tmp_path / "clip-a.mp4").read_bytes()
    run = client.json("POST", "/api/runs?name=prefix.mp4&device=intel-gpu", video)[1]
    failed = client.wait(run["id"], "failed")
    assert failed["exit_code"] == 3 and failed["has_result"]
    assert failed["summary"]["status"] == "failed"
    assert failed["summary"]["exit_code"] == 3
    assert "this attempt's decoded prefix" in failed["error"]
    assert "earlier result" not in failed["error"]
    path = server.runs.path(run["id"])
    assert (path / "inference/result.json").read_bytes() == RESULT.read_bytes()
    assert read_json(path / "outcome.json")["detector_stalled"]
    monkeypatch.setattr(server.runs, "worker_command", original_command)
    next_run = client.json("POST", "/api/runs?name=next.mp4&device=cpu", video)[1]
    assert client.wait(next_run["id"], "complete")["has_result"]


@pytest.mark.parametrize("mismatch", ["stale", "source", "config", "device", "compile"])
def test_app_parent_does_not_attribute_unrelated_detection_failure(app, mismatch):
    server, client, tmp_path = app
    video = (tmp_path / "clip-a.mp4").read_bytes()
    run = client.json("POST", "/api/runs?name=previous.mp4&device=cpu", video)[1]
    client.wait(run["id"], "complete")
    path = server.runs.path(run["id"])
    record = server.runs._update(
        run["id"], status="running", device="intel-gpu", queued=time.time()
    )
    config = home.installed_config(server.root, "intel-gpu")
    write_json(path / "request.json", {"config": config}, overwrite=True)
    resolved = load_config(ReconstructionConfig, overrides=config)
    summary = {
        "status": "failed",
        "failure_kind": "runtime",
        "exit_code": 3,
        "device": "intel-gpu",
        "source_sha256": record["source_sha256"],
        "config_sha256": json_sha256(resolved.detection.model_dump(mode="json")),
    }
    if mismatch in ("source", "config"):
        summary[mismatch + "_sha256"] = "0" * 64
    if mismatch == "device":
        summary["device"] = "cpu"
    if mismatch != "compile":
        write_json(path / "inference/detect.json", summary)
    if mismatch == "stale":
        server.runs._update(run["id"], queued=time.time() + 1)
    (path / "outcome.json").unlink()
    server.runs._finish(run["id"], 1)
    failed = server.runs.get(run["id"])
    assert failed["status"] == "failed" and failed["summary"] is None
    assert "stopped unexpectedly" in failed["error"]


@pytest.mark.parametrize("code", [1, 3])
def test_app_fatal_worker_exit_does_not_complete_an_untagged_result(app, code):
    server, client, tmp_path = app
    video = (tmp_path / "clip-a.mp4").read_bytes()
    run = client.json("POST", "/api/runs?name=prefix.mp4&device=cpu", video)[1]
    client.wait(run["id"], "complete")
    # Represents a published result whose worker died before tagging the
    # outcome. No fresh detector receipt exists, so do not invent its cause.
    server.runs._update(
        run["id"], status="running", device="intel-gpu", queued=time.time()
    )
    server.runs._finish(run["id"], code)
    failed = server.runs.get(run["id"])
    assert failed["status"] == "failed" and failed["has_result"]
    assert failed["exit_code"] == code
    assert "result from this failed attempt" in failed["error"]
    assert "decoded prefix" not in failed["error"]


def test_intel_watchdog_escalates_when_termination_is_ignored(app, monkeypatch):
    server, _, _ = app
    monkeypatch.setattr(runs, "INTEL_IDLE_TIMEOUT_S", 0)
    path = server.runs.directory / ("f" * 16)
    path.mkdir()
    write_json(
        path / "run.json", {"id": path.name, "created": "fixture", "status": "running"}
    )

    class StalledProcess:
        terminated = False
        killed = False

        def wait(self, timeout=None):
            if self.killed:
                return -9
            raise subprocess.TimeoutExpired("mock native compile", timeout or 0)

        def terminate(self):
            self.terminated = True

        def kill(self):
            assert self.terminated
            self.killed = True

    process = StalledProcess()
    assert server.runs._wait(process, path, "intel-gpu") == 3
    assert process.terminated and process.killed
    assert "without progress" in read_json(path / "outcome.json")["error"]


def test_requests_are_confined_to_the_loopback_page(app):
    server, client, tmp_path = app
    status, headers, body = client.request("GET", "/")
    assert status == 200 and b"Tennis3D" in body
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert client.request("GET", "/api/state", headers={"Host": "evil.test"})[0] == 403
    video = (tmp_path / "clip-a.mp4").read_bytes()
    # A cross-site page cannot add the header, and a foreign origin is refused.
    assert client.request("POST", "/api/runs?name=a.mp4", video)[0] == 403
    local = HEADERS | {"Origin": f"http://localhost:{server.server_port}"}
    assert client.json("POST", "/api/runs?name=notes.txt", b"t", local)[0] == 400
    foreign = HEADERS | {"Origin": "http://evil.test"}
    assert client.request("POST", "/api/runs?name=a.mp4", video, foreign)[0] == 403
    assert client.json("POST", "/api/runs?name=notes.txt", b"text")[0] == 400
    assert client.json("POST", "/api/runs?name=a.mp4", b"")[0] == 400
    for path in (
        "/api/runs/..%2F..%2Fweights",
        "/api/runs/0000000000000000",
        "/api/runs/%2e%2e/result",
        "/api/gallery/..%2Fhome/result",
        "/weights/lifter.pt",
    ):
        assert client.request("GET", path)[0] == 404
    assert client.json("GET", "/api/state")[1]["runs"] == []
    assert list((server.root / "runs").iterdir()) == []


def test_run_progress_cancel_resume_result_and_delete(app):
    server, client, tmp_path = app
    video = (tmp_path / "clip-a.mp4").read_bytes()
    hold = server.root / "hold"
    hold.touch()
    status, run = client.json(
        "POST", "/api/runs?name=..%2F..%2Fevil%20name.mp4&device=cpu", video
    )
    assert status == 201 and run["name"] == "evil name.mp4"
    stored = server.root / "runs" / run["id"]
    assert (stored / "source.mp4").read_bytes() == video
    assert run["source_sha256"] == file_sha256(tmp_path / "clip-a.mp4")
    second = client.json("POST", "/api/runs?name=b.mp4", video)[1]
    running = client.wait(run["id"], "running")
    assert client.json("GET", f"/api/runs/{second['id']}")[1]["status"] == "queued"
    deadline = time.monotonic() + 10
    while "progress" not in running and time.monotonic() < deadline:
        running = client.json("GET", f"/api/runs/{run['id']}")[1]
    assert running["progress"] == {"completed": 1, "total": 4, "unit": "frames"}
    assert client.request("GET", f"/api/runs/{run['id']}/result")[0] == 404

    process = server.runs.process
    assert client.json("POST", f"/api/runs/{run['id']}/cancel")[1]["status"] == (
        "cancelled"
    )
    process.wait(timeout=10)
    assert client.json("GET", f"/api/runs/{run['id']}")[1]["status"] == "cancelled"
    assert (stored / "source.mp4").is_file()
    # The queued run starts next; cancelling it before it finishes also holds.
    client.wait(second["id"], "running")
    assert client.json("DELETE", f"/api/runs/{second['id']}")[0] == 200
    assert not (server.root / "runs" / second["id"]).exists()

    hold.unlink()
    assert client.json("POST", f"/api/runs/{run['id']}/start")[1]["status"] == "queued"
    done = client.wait(run["id"], "complete")
    assert done["has_result"] and done["summary"]["status"] == "partial"
    status, headers, body = client.request(
        "GET", f"/api/runs/{run['id']}/result?download=1"
    )
    assert status == 200 and body == RESULT.read_bytes()
    assert "attachment" in headers["Content-Disposition"]

    # A failed new attempt is reported as failed; the earlier result stays viewable.
    (server.root / "fail").touch()
    client.json("POST", f"/api/runs/{run['id']}/start")
    failed = client.wait(run["id"], "failed")
    assert failed["has_result"] and failed["exit_code"] == 2
    assert "earlier result" in failed["error"]
    (server.root / "fail").unlink()
    client.json("POST", f"/api/runs/{run['id']}/start")
    client.wait(run["id"], "complete")
    # Only one server may own an app directory.
    with pytest.raises(AlreadyRunning):
        RunStore(server.root)

    # A new server keeps finished runs and marks unfinished ones as resumable.
    hold.touch()
    third = client.json("POST", "/api/runs?name=c.mp4", video)[1]
    client.wait(third["id"], "running")
    server.runs.close()
    reopened = RunStore(server.root)
    try:
        states = {item["id"]: item["status"] for item in reopened.list()}
        assert states == {run["id"]: "complete", third["id"]: "interrupted"}
    finally:
        reopened.close()


def test_gallery_serves_verified_results_and_seekable_video(app):
    server, client, tmp_path = app
    assert server.gallery.prepare() == {"clip-a": "ready", "clip-b": "unavailable"}
    clips = {c["id"]: c for c in client.json("GET", "/api/state")[1]["gallery"]}
    assert clips["clip-a"]["video"] == "ready"
    assert clips["clip-b"]["video"] == "unavailable"
    assert clips["clip-a"]["rights"] == {"redistribution": "unknown"}
    # Manifest coverage notes are not card text; a video problem still is.
    assert clips["clip-a"]["note"] is None and clips["clip-b"]["note"] is None
    server.gallery.errors["clip-b"] = "Video unavailable: unreadable"
    described = {c["id"]: c for c in server.gallery.describe()}
    assert described["clip-b"]["note"] == "Video unavailable: unreadable"
    del server.gallery.errors["clip-b"]
    status, _, body = client.request("GET", "/api/gallery/clip-a/result")
    assert status == 200 and body == RESULT.read_bytes()
    assert client.request("GET", "/api/gallery/clip-b/result")[0] == 500
    assert client.request("GET", "/api/gallery/clip-b/video")[0] == 404
    copy = (
        server.root / "gallery-playback" / f"{file_sha256(tmp_path / 'clip-a.mp4')}.mp4"
    )
    data = copy.read_bytes()
    status, headers, body = client.request(
        "GET", "/api/gallery/clip-a/video", headers={"Range": "bytes=10-19"}
    )
    assert status == 206 and body == data[10:20]
    assert headers["Content-Range"] == f"bytes 10-19/{len(data)}"
    status, _, body = client.request(
        "GET", "/api/gallery/clip-a/video", headers={"Range": "bytes=-5"}
    )
    assert status == 206 and body == data[-5:]
    status, headers, _ = client.request(
        "GET", "/api/gallery/clip-a/video", headers={"Range": f"bytes={len(data)}-"}
    )
    assert status == 416
    # The original gallery files are never rewritten.
    assert read_json(copy.with_suffix(".json"))["source_sha256"] == file_sha256(
        tmp_path / "clip-a.mp4"
    )
    with pytest.raises(ValueError):
        Gallery(RESULT, tmp_path / "cache")
    # Changed cache bytes are no longer served as this clip's video.
    copy.write_bytes(data + b"x")
    assert client.request("GET", "/api/gallery/clip-a/video")[0] == 404
    assert server.gallery.video_state("clip-a") != "ready"


def test_busy_default_port_falls_back_to_a_free_one(app, tmp_path, monkeypatch):
    server, _, _ = app
    other = tmp_path / "other-home"
    # A busy port is an ordinary bind error, with nothing left locked behind it.
    with pytest.raises(OSError):
        App(other, server.server_port)
    assert not (other / "runs").exists()
    # `tennis3d dashboard` with no --port then takes a free port.
    from tennis3d.app import server as server_module

    monkeypatch.setattr(server_module, "DEFAULT_PORT", server.server_port)
    started = []

    def stop_after_start(self):
        started.append(self.server_port)

    monkeypatch.setattr(App, "serve_forever", stop_after_start)
    assert server_module.serve(other, open_browser=False) == 0
    assert started and started[0] != server.server_port
    assert not (other / "server.json").exists()
    with pytest.raises(OSError):
        server_module.serve(tmp_path / "third", port=server.server_port)


def test_dashboard_command_keeps_the_experiment_monitor(tmp_path, capsys):
    parser = build_parser()
    monitor = parser.parse_args(["monitor", "--run", str(tmp_path)])
    compatible = parser.parse_args(["dashboard", "--run", str(tmp_path)])
    assert monitor.port == 8765 and compatible.run == monitor.run == tmp_path
    assert compatible.kind == monitor.kind == "accuracy"
    app = parser.parse_args(["dashboard", "--no-open"])
    assert app.run is None and app.port is None
    assert main(["setup", "--check", "--home", str(tmp_path / "empty")]) == 3
    assert json.loads(capsys.readouterr().out)["ready"] is False


def test_gallery_checks_each_clip_against_its_own_video_hash(tmp_path):
    video = write_video(tmp_path / "shared.mp4")
    clip = {"result_path": str(RESULT), "result_sha256": file_sha256(RESULT)}
    write_json(
        tmp_path / "gallery.json",
        {
            "schema_version": 1,
            "clips": [
                clip
                | {
                    "id": "right",
                    "playback_path": "shared.mp4",
                    "playback_sha256": file_sha256(video),
                },
                clip
                | {
                    "id": "wrong",
                    "playback_path": "shared.mp4",
                    "playback_sha256": "2" * 64,
                },
                clip | {"id": "result-only"},
            ],
        },
    )
    gallery = Gallery(tmp_path / "gallery.json", tmp_path / "cache")
    states = {item["id"]: item["video"] for item in gallery.describe()}
    assert states == {
        "right": "ready",
        "wrong": "unavailable",
        "result-only": "unavailable",
    }
    assert gallery.prepare() == states
    with pytest.raises(ValueError, match="Invalid"):
        write_json(
            tmp_path / "bad.json",
            {"schema_version": 1, "clips": [clip | {"id": ".."}]},
        )
        Gallery(tmp_path / "bad.json", tmp_path / "cache")


def test_shipped_examples_include_display_copies_at_exact_source_times(tmp_path):
    from scripts.gallery_display_copies import verify_clip

    manifest = Path("examples/gallery-v2/gallery.json")
    gallery = Gallery(manifest, tmp_path / "cache")
    assert len(gallery.clips) == 20
    assert {clip["video"] for clip in gallery.describe()} == {"ready"}
    frames = 0
    for clip in read_json(manifest)["clips"]:
        folder = manifest.parent / clip["id"]
        assert gallery.video(clip["id"]) == (folder / "playback.mp4").resolve()
        assert gallery.result(clip["id"]) and gallery.timeline(clip["id"])
        # The original lossless source is named by hash and link, never shipped.
        assert not (manifest.parent / clip["source_path"]).exists()
        assert clip["playback_sha256"] != clip["source_sha256"]
        assert clip["rights"]["source_url"].startswith("https://")
        frames += verify_clip(clip, folder)
    assert frames == 10749
    assert not list((tmp_path / "cache").glob("*"))


def test_git_checkout_keeps_example_bytes_and_startup_restores_converted_clones(
    tmp_path,
):
    import os
    import shutil

    shipped = Path("examples/gallery-v2")
    origin = tmp_path / "origin"
    for path in sorted(shipped.rglob("*.json")):
        (origin / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, origin / path)
    environment = os.environ | {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    }

    def git(*arguments, cwd=origin):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *arguments],
            cwd=cwd,
            env=environment,
            check=True,
            capture_output=True,
        )

    def checkout(name):
        # A real checkout with the setting Git for Windows installs by default.
        git(
            "-c", "core.autocrlf=true", "clone", "-q", str(origin), str(tmp_path / name)
        )
        manifest = tmp_path / name / shipped / "gallery.json"
        gallery = Gallery(manifest, tmp_path / "cache")
        assert len(gallery.clips) == 20
        return manifest.parent, gallery

    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "examples without attributes")
    converted, gallery = checkout("converted")
    files = [Path(clip[key]) for clip in gallery.clips.values() for key in BOUND_PATHS]
    for clip_id in gallery.clips:
        with pytest.raises(ValueError, match="result does not match"):
            gallery.result(clip_id)
        with pytest.raises(ValueError, match="timeline does not match"):
            gallery.timeline(clip_id)
    # Three files differ by more than line endings and must stay as they are.
    edited, forged, truncated = files[0], files[1], files[2]
    edited.write_bytes(edited.read_bytes().replace(b'"frames"', b'"Frames"', 1))
    forged.write_bytes(forged.read_bytes() + b"\r\n")
    truncated.write_bytes(truncated.read_bytes()[:-2])
    refused = {path: path.read_bytes() for path in (edited, forged, truncated)}
    assert sorted(gallery.restore_line_endings()) == sorted(files[3:])
    for path in files[3:]:
        assert path.read_bytes() == (shipped / path.relative_to(converted)).read_bytes()
    assert all(path.read_bytes() == data for path, data in refused.items())
    assert not [path for path in converted.rglob("*") if path.name.startswith(".")]
    for clip_id, clip in gallery.clips.items():
        for kind, read in (("result", gallery.result), ("timeline", gallery.timeline)):
            if Path(clip[f"{kind}_path"]) in refused:
                with pytest.raises(ValueError, match="SHA-256"):
                    read(clip_id)
            else:
                assert file_sha256(read(clip_id)) == clip[f"{kind}_sha256"]
    assert gallery.restore_line_endings() == []

    # With the repository's attributes, Git leaves every bound file alone.
    video = shipped / "gallery-v2-02-03/playback.mp4"
    shutil.copyfile(video, origin / video)
    shutil.copyfile(".gitattributes", origin / ".gitattributes")
    git("add", ".")
    git("commit", "-q", "-m", "attributes")
    kept, gallery = checkout("kept")
    assert (kept / "gallery.json").read_bytes().count(b"\r\n") > 0
    assert (kept / video.relative_to(shipped)).read_bytes() == video.read_bytes()
    for path in files:
        path = path.relative_to(converted)
        assert (kept / path).read_bytes() == (shipped / path).read_bytes()
    assert gallery.restore_line_endings() == []
    assert gallery.video_state("gallery-v2-02-03") == "ready"
    for clip_id in gallery.clips:
        assert gallery.result(clip_id) and gallery.timeline(clip_id)


def test_dropped_connections_are_not_reported_as_errors(app, capsys, monkeypatch):
    import socket
    import struct

    server, _, _ = app
    # A browser resets an idle video connection when it seeks elsewhere.
    dropped = socket.create_connection(("127.0.0.1", server.server_port))
    dropped.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    dropped.sendall(b"GET /api/state HTTP/1.1\r\nHost: 127.0.0.1\r\n")
    dropped.close()
    monkeypatch.setattr(server.RequestHandlerClass, "route", lambda *_: 1 / 0)
    with pytest.raises(http.client.RemoteDisconnected):
        Client(server).request("GET", "/api/state")
    deadline = time.monotonic() + 5
    report = ""
    while "ZeroDivisionError" not in report and time.monotonic() < deadline:
        time.sleep(0.05)
        report += capsys.readouterr().err
    assert "ZeroDivisionError" in report and "ConnectionResetError" not in report
