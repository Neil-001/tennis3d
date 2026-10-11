"""Optional Intel backend checks. Real OpenVINO runs use its CPU device."""

import json
import sys
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("av")
torch = pytest.importorskip("torch")

from tennis3d.artifacts import read_npz
from tennis3d.perception import accelerate
from tennis3d.perception import detect as stage
from tennis3d.perception.config import DetectConfig
from tennis3d.video.decode import DecodedFrame


def small_networks():
    torch.manual_seed(0)
    return {
        "court": torch.nn.Conv2d(3, 15, 3, padding=1).eval(),
        "ball": torch.nn.Conv2d(9, 256, 3, padding=1).eval(),
    }


def test_missing_runtime_or_device_is_an_explicit_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "openvino", None)
    with pytest.raises(RuntimeError, match="intel extra"):
        accelerate.load_inference(small_networks(), "intel-gpu")

    class Core:
        available_devices = ("CPU",)

    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=Core))
    with pytest.raises(RuntimeError, match="no GPU device"):
        accelerate.load_inference(small_networks(), "intel-gpu")
    assert DetectConfig().device == "auto"
    assert stage.device_name("auto") in ("cpu", "cuda")


def test_cli_keeps_lifting_off_the_intel_device(capsys):
    from tennis3d.cli import build_parser, main
    from tennis3d.inference.settings import LiftConfig, lifting_device

    assert [lifting_device(name) for name in ("auto", "cuda", "intel-gpu")] == [
        "auto",
        "cuda",
        "cpu",
    ]
    with pytest.raises(ValueError):
        LiftConfig.model_validate({"device": "intel-gpu"})
    parser = build_parser()
    parser.parse_args(
        ["reconstruct", "a.mp4", "--output", "o", "--device", "intel-gpu"]
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["lift", "--run", "o", "--device", "intel-gpu"])
    capsys.readouterr()
    assert main(["doctor", "--device", "intel-gpu"]) == 3
    assert "requires --perception" in json.loads(capsys.readouterr().out)["accelerator"]


def test_openvino_stage_matches_torch_and_records_backend(tmp_path, monkeypatch):
    openvino = pytest.importorskip("openvino")
    rng = np.random.default_rng(1)
    images = rng.integers(0, 256, (4, 72, 128, 3), dtype=np.uint8)

    def decode(path):
        for i, image in enumerate(images):
            yield DecodedFrame(i, i, Fraction(1, 30), Fraction(i, 30), i / 30, image)

    networks = small_networks()
    monkeypatch.setattr(stage, "decode", decode)
    monkeypatch.setattr(stage, "verify", lambda asset: None)
    monkeypatch.setattr(
        stage, "load_network", lambda path, name, device: networks[name]
    )
    # The same conversion and wrappers, on the device every OpenVINO install has.
    monkeypatch.setitem(accelerate.OPENVINO_DEVICES, "intel-gpu", "CPU")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    summaries = {
        device: stage.detect(
            video, tmp_path / device, DetectConfig(device=device, overlay_every=0)
        )
        for device in ("cpu", "intel-gpu")
    }
    assert summaries["cpu"]["accelerator"] is None
    assert "openvino" not in summaries["cpu"]["runtime"]
    accelerated = summaries["intel-gpu"]
    assert accelerated["device"] == "intel-gpu"
    assert accelerated["failure"] is None
    assert accelerated["accelerator"] == {
        "backend": "openvino",
        "openvino": openvino.get_version(),
        "device": "CPU",
        "device_name": openvino.Core().get_property("CPU", "FULL_DEVICE_NAME"),
        "inference_precision": "f32",
    }
    assert "openvino" in accelerated["runtime"]
    for index in range(len(images)):
        name = f"raw/{index:08d}.npz"
        reference = read_npz(tmp_path / "cpu" / name)[0]
        candidate = read_npz(tmp_path / "intel-gpu" / name)[0]
        court = np.abs(
            reference["court_sigmoid_uint8"].astype(int)
            - candidate["court_sigmoid_uint8"]
        )
        # Float32 rounding may move a sigmoid across one uint8 boundary.
        assert court.max() <= 1 and (court > 0).mean() < 1e-3
        assert candidate["ball_has_context"] == (index >= 2)
        labels = reference["ball_intensity_class"]
        assert labels.any() == (index >= 2)
        assert (labels != candidate["ball_intensity_class"]).mean() < 1e-3
    for name in ("frames.npz", "court.npz"):
        reference = read_npz(tmp_path / "cpu" / name)[0]
        candidate = read_npz(tmp_path / "intel-gpu" / name)[0]
        for key in reference:
            np.testing.assert_array_equal(reference[key], candidate[key])


def test_openvino_rejects_non_finite_output():
    pytest.importorskip("openvino")
    networks = small_networks()
    with torch.no_grad():
        networks["ball"].bias[3] = float("inf")
        networks["court"].bias[0] = float("nan")
    accelerate.OPENVINO_DEVICES["cpu-test"] = "CPU"
    try:
        infer, _ = accelerate.load_inference(networks, "cpu-test", size=(8, 8))
    finally:
        del accelerate.OPENVINO_DEVICES["cpu-test"]
    with pytest.raises(RuntimeError, match="Non-finite court"):
        infer["court"](np.zeros((1, 3, 8, 8), np.float32))
    with pytest.raises(RuntimeError, match="Non-finite ball"):
        infer["ball"](np.zeros((1, 9, 8, 8), np.float32))


def test_stalled_openvino_request_keeps_every_native_handle():
    pytest.importorskip("openvino")
    import os
    import subprocess

    # A real CPU request that outlasts a 1 ms limit. The process leaves the way
    # a caller must, so neither the test nor its teardown waits on the request.
    script = """
import json, os, sys
import numpy as np
import openvino
import torch
from tennis3d.perception import accelerate

slow = torch.nn.Sequential(
    *[torch.nn.Conv2d(64, 64, 3, padding=1) for _ in range(4)]
).eval()
session = [openvino.Core()]
request = accelerate._compile(openvino, session, slow, 64, (360, 640), "CPU")
accelerate.TIMEOUT_MS = 1
try:
    accelerate._infer(session, request, np.zeros((1, 64, 360, 640), np.float32))
    error = None
except RuntimeError as raised:
    error = str(raised)
kept = [[type(handle).__name__ for handle in s] for s in accelerate.STALLED]
del request, session
print(json.dumps({"error": error, "kept": kept}))
sys.stdout.flush()
os._exit(0)
"""
    process = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        env=os.environ | {"OMP_NUM_THREADS": "2"},
        check=False,
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert "stalled" in report["error"]
    assert report["kept"] == [["Core", "CompiledModel", "InferRequest"]]


STALL_DRIVER = """
import json, os, sys
from fractions import Fraction
from pathlib import Path
import numpy as np
import torch
from tennis3d.perception import accelerate
from tennis3d.perception import detect as stage
from tennis3d.perception.config import DetectConfig
from tennis3d.video.decode import DecodedFrame

caller, video, output = sys.argv[1:]
rng = np.random.default_rng(1)
image = rng.integers(0, 256, (72, 128, 3), dtype=np.uint8)
def decode(path):
    for i in range(6):
        yield DecodedFrame(i, i, Fraction(1, 30), Fraction(i, 30), i / 30, image.copy())
torch.manual_seed(0)
networks = {
    "court": torch.nn.Conv2d(3, 15, 3, padding=1).eval(),
    "ball": torch.nn.Conv2d(9, 256, 3, padding=1).eval(),
}
stage.decode = decode
stage.verify = lambda asset: None
stage.load_network = lambda path, name, device: networks[name]
accelerate.TIMEOUT_MS = 200
accelerate.EXIT_GRACE_S = 3
if caller == "command":
    from tennis3d.cli import main
    code = main(["detect", video, "--output", output, "--device", "intel-gpu"])
else:
    summary = stage.detect(Path(video), Path(output), DetectConfig(device="intel-gpu"))
    print(json.dumps(summary))
    code = summary["exit_code"]
    if caller == "library-restores-stderr":
        # The caller had pointed stderr at a file for the call. Its own stderr
        # is a full pipe that nothing reads.
        reader, writer = os.pipe()
        os.set_blocking(writer, False)
        try:
            while True:
                os.write(writer, b"x" * 4096)
        except BlockingIOError:
            os.set_blocking(writer, True)
        os.dup2(writer, 2)
    if caller == "library-hard-exit":
        sys.stdout.flush()
        os._exit(code)
raise SystemExit(code)
"""
# The stand-in device answers court, court, court, ball, court, ball, ...
FOURTH_FRAME_COURT, THIRD_FRAME_BALL = 4, 3
HANDLES = ["CompiledModel", "CompiledModel", "Core", "InferRequest", "InferRequest"]


def run_stand_in_device(tmp_path, caller="command", stderr=None, **settings):
    """Detect in a fresh process on tests/fixtures/stalling_openvino."""
    import os
    import subprocess
    from pathlib import Path

    pytest.importorskip("openvino")  # detect.json records the installed version
    if os.name != "posix":
        pytest.skip("the stand-in device blocks through the C library")
    root = Path(__file__).resolve().parents[1]
    tmp_path.mkdir(exist_ok=True)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    log, run = tmp_path / "handles.log", tmp_path / "run"
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join(
            [str(root / "tests/fixtures/stalling_openvino"), str(root)]
        ),
        "STANDIN_LOG": str(log),
        "OMP_NUM_THREADS": "2",
    }
    environment.update({f"STANDIN_{k.upper()}": str(v) for k, v in settings.items()})
    # A process that outlives the limit is killed and fails the test. Nothing
    # reads a piped stderr while the process runs.
    with (tmp_path / "stderr").open("w+") as errors:
        process = subprocess.run(
            [sys.executable, "-c", STALL_DRIVER, caller, str(video), str(run)],
            stdout=subprocess.PIPE,
            stderr=stderr or errors,
            text=True,
            timeout=120,
            cwd=root,
            env=environment,
            check=False,
        )
        if not stderr:
            errors.seek(0)
            process.stderr = errors.read()
    published = run / "detect.json"
    summary = json.loads(published.read_text()) if published.exists() else None
    return process, summary, log.read_text().split("\n")[:-1] if log.exists() else []


def assert_published_stall(summary, frames):
    assert summary["status"] == "failed"
    assert summary["failure_kind"] == "runtime" and summary["exit_code"] == 3
    assert "Intel GPU inference stalled" in summary["failure"]
    # The frame whose request stalled keeps its timestamp and has no detection.
    assert summary["frame_count"] == frames
    assert summary["accelerator"]["device"] == "GPU"


@pytest.mark.parametrize(
    ("stall_at", "frames"), [(FOURTH_FRAME_COURT, 4), (THIRD_FRAME_BALL, 3)]
)
@pytest.mark.parametrize("caller", ["command", "library-hard-exit"])
def test_stalled_device_publishes_the_failure_and_exits_3(
    tmp_path, caller, stall_at, frames
):
    process, summary, handles = run_stand_in_device(tmp_path, caller, stall_at=stall_at)
    assert process.returncode == 3, process.stderr
    assert_published_stall(summary, frames)
    printed = json.loads(process.stdout)
    assert printed == summary
    frames_file = read_npz(tmp_path / "run" / "frames.npz")[0]
    assert len(frames_file["t_s"]) == frames
    # No native handle was destroyed, so none could block, and no timer fired.
    assert handles == []
    assert "Timeout" not in process.stderr


def test_blocked_finalizer_outside_the_backend_cannot_outlive_the_grace(tmp_path):
    process, summary, handles = run_stand_in_device(
        tmp_path, stall_at=FOURTH_FRAME_COURT, unowned_handle=1
    )
    # The command blocked on a handle it had returned in the summary.
    assert handles == ["blocked _DeviceName"]
    assert process.returncode == 1
    assert "Timeout (0:00:03)!" in process.stderr and "__del__" in process.stderr
    assert_published_stall(summary, 4)


def test_forced_exit_does_not_wait_for_a_full_stderr_pipe(tmp_path):
    import os

    reader, writer = os.pipe()
    try:
        # Fill the pipe; nothing reads it, so any further write would block.
        os.set_blocking(writer, False)
        with pytest.raises(BlockingIOError):
            while True:
                os.write(writer, b"x" * 4096)
        os.set_blocking(writer, True)
        process, summary, handles = run_stand_in_device(
            tmp_path, stderr=writer, stall_at=FOURTH_FRAME_COURT, unowned_handle=1
        )
    finally:
        os.close(reader)
        os.close(writer)
    assert handles == ["blocked _DeviceName"]
    assert process.returncode == 1
    assert_published_stall(summary, 4)


@pytest.mark.parametrize("caller", ["library", "library-restores-stderr"])
def test_library_caller_that_exits_normally_is_ended_after_the_grace(tmp_path, caller):
    process, summary, handles = run_stand_in_device(
        tmp_path, caller, stall_at=THIRD_FRAME_BALL
    )
    # Interpreter shutdown destroyed a kept handle, which blocked.
    assert len(handles) == 1 and handles[0].startswith("blocked ")
    assert process.returncode == 1
    assert "Timeout (0:00:03)!" in process.stderr
    assert_published_stall(summary, 3)
    assert json.loads(process.stdout) == summary


def test_unstalled_device_runs_keep_ordinary_exit_and_teardown(tmp_path):
    process, summary, handles = run_stand_in_device(tmp_path / "ok")
    assert process.returncode == 0, process.stderr
    assert summary["failure"] is None and summary["frame_count"] == 6
    assert summary["status"] in ("ok", "partial")
    assert json.loads(process.stdout) == summary
    assert sorted(handles) == [f"destroyed {name}" for name in HANDLES]
    assert process.stderr == ""

    process, summary, handles = run_stand_in_device(tmp_path / "bad", non_finite_at=4)
    assert process.returncode == 3
    assert summary["failure"] == "Non-finite court network output"
    assert summary["failure_kind"] == "runtime" and summary["frame_count"] == 4
    assert json.loads(process.stdout) == summary
    assert sorted(handles) == [f"destroyed {name}" for name in HANDLES]
    assert "Timeout" not in process.stderr


def test_command_line_skips_teardown_after_a_stall(monkeypatch, capsys):
    import os

    from tennis3d.cli import main

    left = []
    monkeypatch.setattr(os, "_exit", left.append)
    assert main(["doctor", "--device", "cpu"]) == 0
    assert left == []
    monkeypatch.setattr(accelerate, "STALLED", [object()])
    assert main(["doctor", "--device", "intel-gpu"]) == 3
    assert left == [3]
    capsys.readouterr()


def test_lifter_probe_cannot_pass_an_unverified_intel_device(
    tmp_path, monkeypatch, capsys
):
    from tennis3d.cli import main
    from tennis3d.inference import doctor

    monkeypatch.setattr(
        doctor, "check_lifter", lambda path, device: {"runtime": {"device": device}}
    )
    command = ["doctor", "--checkpoint", str(tmp_path / "lifter.pt"), "--device"]
    assert main([*command, "cpu"]) == 0
    capsys.readouterr()
    assert main([*command, "intel-gpu"]) == 3
    report = json.loads(capsys.readouterr().out)
    assert report["lifting"]["runtime"]["device"] == "cpu"
    assert "requires --perception" in report["accelerator"]


def test_cached_intel_detection_requires_the_same_openvino(monkeypatch):
    import importlib.metadata

    from tennis3d.inference.inputs import check_runtime
    from tennis3d.perception.provenance import DETECTION_RUNTIME, detection_runtime

    assert detection_runtime("cpu") == detection_runtime("cuda") == DETECTION_RUNTIME
    names = detection_runtime("intel-gpu")
    assert names == (*DETECTION_RUNTIME, "openvino")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1")
    recorded = dict.fromkeys(names, "1")
    check_runtime({"runtime": recorded}, names, "Detection")
    with pytest.raises(ValueError, match="runtime changed: openvino"):
        check_runtime({"runtime": recorded | {"openvino": "2"}}, names, "Detection")
    del recorded["openvino"]
    with pytest.raises(ValueError, match="identity is missing"):
        check_runtime({"runtime": recorded}, names, "Detection")


def test_device_error_while_compiling_exits_3_with_ordinary_teardown(tmp_path):
    # The second recorded device fault: OpenVINO raised and nothing stalled.
    process, summary, handles = run_stand_in_device(tmp_path, compile_error=1)
    assert process.returncode == 3
    assert json.loads(process.stdout) == {
        "status": "failed",
        "error": "[GPU] clWaitForEvents, error code: -14",
    }
    assert summary is None
    assert handles == ["destroyed Core"]
    assert process.stderr == ""
