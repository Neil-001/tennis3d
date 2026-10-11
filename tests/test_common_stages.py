"""Shared stage preparation and freezing, on the controlled cached gallery."""

import fcntl
import importlib.util
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("scipy")

from test_cached_real import cached_gallery, checkpoint

from tennis3d.artifacts import (
    ArtifactProvenance,
    file_sha256,
    json_sha256,
    read_json,
    read_npz,
    write_json,
    write_npz,
)
from tennis3d.lifting.model import ModelConfig

SCRIPT = Path(__file__).parents[1] / "scripts/common_stages.py"
SPEC = importlib.util.spec_from_file_location("common_stages", SCRIPT)
assert SPEC and SPEC.loader
stages = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stages)
adapter = stages.cached_real
SPEC = importlib.util.spec_from_file_location(
    "product_manifest", SCRIPT.with_name("product_manifest.py")
)
assert SPEC and SPEC.loader
product = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(product)


def stage_tree(tmp_path, **options):
    """The fixture's clip as separate stage commands leave it: one directory per clip."""
    fixture = cached_gallery(tmp_path, **options)
    fixture.stages = tmp_path / "stages"
    shutil.copytree(fixture.run, fixture.stages / fixture.clip.clip_id)
    return fixture


def frozen(fixture, name="common", **bindings):
    root = fixture.root / name
    manifest = fixture.root / f"{name}-manifest.json"
    stages.freeze(
        fixture.inventory,
        file_sha256(fixture.inventory),
        fixture.stages,
        root,
        manifest,
        bindings,
    )
    return root, manifest


def test_frozen_root_feeds_the_adapter_and_keeps_the_recorded_producer(
    tmp_path, monkeypatch
):
    fixture = stage_tree(tmp_path)
    before = {
        path.name: path.read_bytes()
        for path in (fixture.stages / "controlled").iterdir()
    }
    root, manifest = frozen(fixture, evidence=fixture.inventory)
    document = read_json(manifest)
    run = root / "clips/controlled"
    # Only the nine stage files are copied; the earlier model's lift is left behind.
    assert sorted(path.name for path in run.iterdir()) == sorted(adapter.UPSTREAM_FILES)
    assert all(
        run.joinpath(name).read_bytes() == before[name]
        for name in before
        if name != "lift.json"
    )
    assert document["original_inventory_sha256"] == file_sha256(fixture.inventory)
    assert [row["clip_id"] for row in document["clips"]] == ["controlled"]
    shared = document["common_stages"]
    assert shared["producer_code_revisions"] == ["python-source-sha256:" + "0" * 64]
    assert shared["frame_count"] == 60
    assert shared["bindings"]["evidence"]["sha256"] == file_sha256(fixture.inventory)
    assert read_npz(run / "frames.npz")[1].code_revision.endswith("0" * 64)

    release = checkpoint(tmp_path / "model.pt", ModelConfig(width=16, depth=2), 1)
    provenance = {"release": {"checkpoint": {"sha256": release.checkpoint.sha256}}}
    monkeypatch.setattr(
        adapter, "checked_release", lambda _: (release, None, provenance)
    )
    result = adapter.cached_real(
        tmp_path / "unused-release.json",
        fixture.inventory,
        root,
        manifest,
        file_sha256(manifest),
        tmp_path / "model",
    )
    record = {row["clip_id"]: row for row in result["records"]}["controlled"]
    assert record["status"] == "partial"
    shared_inputs = record["pipeline"]["cached_inputs"]
    assert shared_inputs["artifact_manifest_sha256"] == file_sha256(manifest)
    # The model read exactly the frozen stage bytes.
    assert shared_inputs["upstream_sha256"] == {
        name: entry["sha256"]
        for name, entry in document["clips"][0]["artifacts"].items()
    }

    # The app manifest lists every inventory clip and the unchanged result.
    release.run_id, release.step = "fixture-run", 1
    monkeypatch.setattr(
        product, "checked_release", lambda _: (release, None, provenance)
    )
    write_json(
        tmp_path / "clips.json",
        {
            "schema_version": 1,
            "clips": [
                {
                    "id": name,
                    "title": name.title(),
                    "rights": {"redistribution": "unknown"},
                    "source_sha256": fixture.clip.source_sha256,
                    "timeline_sha256": fixture.clip.timeline_sha256,
                }
                for name in ("controlled", "barred")
            ],
        },
    )
    arguments = (
        tmp_path / "model",
        fixture.inventory,
        file_sha256(fixture.inventory),
        tmp_path / "clips.json",
        tmp_path / "unused-release.json",
    )
    fixture.root.joinpath("unused-release.json").write_text("{}")
    binding = product.product(*arguments, tmp_path / "product")
    shown = read_json(tmp_path / "product/product-manifest.json")
    assert shown["schema_version"] == 1
    listed = {clip["id"]: clip for clip in shown["clips"]}
    result = tmp_path / "model/clips/controlled/result.json"
    assert listed["controlled"]["result_path"] == str(result)
    assert listed["controlled"]["result_sha256"] == file_sha256(result)
    assert listed["controlled"]["title"] == "Controlled"
    assert Path(listed["controlled"]["source_path"]).is_absolute()
    assert Path(listed["controlled"]["timeline_path"]).is_absolute()
    assert "result_path" not in listed["barred"]
    assert listed["barred"]["note"] == "ineligible: no result (no access)"
    rows = binding["clips"][0]["rows"]
    frames = read_json(result)["frames"]
    covered = int(read_npz(result.with_name("predictions_raw.npz"))[0]["covered"].sum())
    exported = sum(frame["xyz_m"] is not None for frame in frames)
    assert rows["source"] == 60 and rows["raw_covered"] == covered
    assert rows["exported_xyz"] == exported and rows["exported_gap"] == 60 - exported
    assert 0 < exported < covered
    assert binding["status_counts"] == {"ineligible": 1, "partial": 1}
    assert binding["checkpoint"]["sha256"] == release.checkpoint.sha256
    assert binding["identities"]["artifact_manifest"]["sha256"] == file_sha256(manifest)
    # A result changed after the evaluation is refused, not listed.
    saved = result.read_bytes()
    result.write_bytes(saved + b" ")
    with pytest.raises(product.ReleaseError, match="differs from its run summary"):
        product.product(*arguments, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()
    result.write_bytes(saved)
    summary = result.with_name("detect.json")
    summary.chmod(0o600)
    saved = summary.read_bytes()
    summary.write_bytes(saved + b" ")
    with pytest.raises(product.ReleaseError, match="differs from its run summary"):
        product.product(*arguments, tmp_path / "refused")
    summary.write_bytes(saved)

    # A stage byte changed after the freeze stops the next model without metrics.
    with open(run / "cameras.npz", "ab") as stream:
        stream.write(b"\0")
    with pytest.raises(adapter.CachedInputError, match="differs from manifest"):
        adapter.cached_real(
            tmp_path / "unused-release.json",
            fixture.inventory,
            root,
            manifest,
            file_sha256(manifest),
            tmp_path / "next",
        )
    assert not (tmp_path / "next").exists()


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("pin", "inventory differs"),
        ("missing", "controlled"),
        ("threshold", "reviewed defaults"),
        ("timeline", "frozen source timeline"),
        ("producer", "different producers"),
        ("runtime", "runtime changed"),
        ("source", "source video identity"),
    ],
)
def test_untrusted_stages_are_not_frozen(tmp_path, damage, message):
    options = {}
    if damage == "producer":
        options["camera_producer"] = "python-source-sha256:" + "1" * 64
    if damage == "runtime":
        options["numpy_version"] = "0.0.0"
    fixture = stage_tree(tmp_path, **options)
    run = fixture.stages / "controlled"
    pin = file_sha256(fixture.inventory)
    if damage == "pin":
        pin = "0" * 64
    if damage == "missing":
        (run / "court.npz").unlink()
    if damage == "threshold":
        # A self-consistent stage produced with a changed threshold.
        config = read_json(run / "calibrate.config.json") | {"min_landmarks": 7}
        write_json(run / "calibrate.config.json", config, overwrite=True)
        arrays, provenance = read_npz(run / "cameras.npz")
        write_npz(
            run / "cameras.npz",
            arrays,
            ArtifactProvenance.model_validate(
                provenance.model_dump() | {"config_sha256": json_sha256(config)}
            ),
            overwrite=True,
        )
        summary = read_json(run / "calibrate.json") | {
            "config_sha256": json_sha256(config),
            "cameras_sha256": file_sha256(run / "cameras.npz"),
        }
        write_json(run / "calibrate.json", summary, overwrite=True)
    if damage == "timeline":
        timeline = read_json(fixture.clip.timeline_path)
        timeline["frames"][5]["pts"] += 1
        write_json(fixture.clip.timeline_path, timeline, overwrite=True)
        inventory = read_json(fixture.inventory)
        inventory["clips"][0]["timeline_sha256"] = file_sha256(
            fixture.clip.timeline_path
        )
        write_json(fixture.inventory, inventory, overwrite=True)
        pin = file_sha256(fixture.inventory)
    if damage == "source":
        Path(fixture.clip.source_path).write_bytes(b"another file")
    manifest = fixture.root / "manifest.json"
    with pytest.raises(stages.StageError, match=message):
        stages.freeze(
            fixture.inventory,
            pin,
            fixture.stages,
            fixture.root / "common",
            manifest,
            {},
        )
    assert not manifest.exists()


STUB = """#!{python}
import fcntl, json, os, sys, time
from pathlib import Path
stage, mode = sys.argv[1], os.environ["STUB_MODE"]
run = Path(sys.argv[sys.argv.index("--output" if stage == "detect" else "--run") + 1])
run.mkdir(parents=True, exist_ok=True)
if stage == "detect":
    with open(os.environ["STUB_LOCK"], "a") as held:
        try:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            free = True
        except BlockingIOError:
            free = False
    (run / "lock_was_free").write_text(str(free))
    (run / "bounds_existed").write_text(str(Path(os.environ["STUB_BOUNDS"]).is_file()))
    (run / "pid").write_text(str(os.getpid()))
    if mode == "stubborn" and run.name == "second":
        import signal
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        (run / "pid").write_text(str(os.getpid()))
        time.sleep(600)
    if mode == "hang" and run.name == "second":
        time.sleep(600)
    if mode == "crash" and run.name == "second":
        os.abort()
    bad = mode in ("stage_failure", "runtime", "abort_after_summary") and run.name == "second"
    code = 3 if bad and mode == "runtime" else 2 if bad and mode == "stage_failure" else 0
    (run / "detect.json").write_text(json.dumps(
        {{"stage": "detect", "status": "failed" if code else "partial", "exit_code": code,
          "source_sha256": os.environ["STUB_SOURCE"]}}))
    if bad and mode == "abort_after_summary":
        os.abort()
    sys.exit(code)
(run / "calibrate.json").write_text(json.dumps({{"stage": "calibrate", "status": "ok"}}))
"""


def staged(tmp_path, monkeypatch, mode):
    """Two clips and a stub CLI; returns the prepare arguments."""
    fixture = cached_gallery(tmp_path)
    inventory = read_json(fixture.inventory)
    inventory["clips"][1] = inventory["clips"][0] | {"clip_id": "second"}
    inventory["clips"][0]["clip_id"] = "first"
    write_json(fixture.inventory, inventory, overwrite=True)
    stub = tmp_path / "tennis3d-stub"
    stub.write_text(STUB.format(python=sys.executable))
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    config = tmp_path / "detect.json"
    write_json(config, {"save_heatmaps": False, "overlay_every": 0, "device": "cpu"})
    output = tmp_path / "prepared"
    monkeypatch.setenv("STUB_MODE", mode)
    monkeypatch.setenv("STUB_LOCK", str(tmp_path / "device.lock"))
    monkeypatch.setenv("STUB_BOUNDS", str(output / "bounds.json"))
    monkeypatch.setenv("STUB_SOURCE", fixture.clip.source_sha256)
    return [
        fixture.inventory,
        file_sha256(fixture.inventory),
        output,
        stub,
        config,
        tmp_path / "device.lock",
    ]


def prepared(tmp_path, monkeypatch, mode):
    arguments = staged(tmp_path, monkeypatch, mode)
    monkeypatch.setattr(
        stages, "kernel_events", lambda since: {"readable": True, "lines": []}
    )
    summary = stages.prepare(*arguments, base_s=1.0, per_frame_s=0.05, calibrate_s=30.0)
    return arguments[2], summary


def test_prepare_holds_the_lock_and_freezes_bounds_first(tmp_path, monkeypatch):
    output, summary = prepared(tmp_path, monkeypatch, "ok")
    assert summary["status"] == "complete" and summary["clips_attempted"] == 2
    bounds = read_json(output / "bounds.json")
    assert bounds["clips"]["first"] == {"frame_count": 60, "detect_limit_s": 4.0}
    assert summary["bounds_sha256"] == file_sha256(output / "bounds.json")
    for clip in ("first", "second"):
        run = output / "stages" / clip
        assert (run / "lock_was_free").read_text() == "False"
        assert (run / "bounds_existed").read_text() == "True"
        assert read_json(run / "calibrate.json")["status"] == "ok"
    assert [row["detect"]["timed_out"] for row in summary["records"]] == [False, False]
    assert all(row["detect"]["lock_wait_s"] >= 0 for row in summary["records"])
    # An existing root is never reused or overwritten.
    with pytest.raises(ValueError, match="already exists"):
        stages.prepare(
            tmp_path / "inventory.json",
            file_sha256(tmp_path / "inventory.json"),
            output,
            tmp_path / "tennis3d-stub",
            tmp_path / "detect.json",
            tmp_path / "device.lock",
            base_s=1.0,
            per_frame_s=0.05,
            calibrate_s=30.0,
        )


@pytest.mark.parametrize(
    "events",
    [
        {"readable": True, "lines": ["kernel: Fence expiration time out i915"]},
        {"readable": False, "lines": []},
    ],
)
def test_kernel_graphics_event_stops_a_run_that_reported_success(
    tmp_path, monkeypatch, events
):
    arguments = staged(tmp_path, monkeypatch, "ok")
    monkeypatch.setattr(stages, "kernel_events", lambda since: events)
    limits = {"base_s": 1.0, "per_frame_s": 0.05, "calibrate_s": 30.0}
    # A CPU detection is not judged by another program's graphics fault.
    summary = stages.prepare(*arguments, **limits)
    assert summary["status"] == "complete" and summary["clips_attempted"] == 2
    assert summary["records"][0]["detect"]["kernel_events"] == events
    bounds = read_json(arguments[2] / "bounds.json")
    assert bounds["kernel_graphics_event_stops_run"] is False
    assert bounds["cpu_affinity"] == sorted(os.sched_getaffinity(0))
    write_json(
        arguments[4],
        {"save_heatmaps": False, "overlay_every": 0, "device": "intel-gpu"},
        overwrite=True,
    )
    arguments[2] = tmp_path / "prepared-graphics"
    monkeypatch.setenv("STUB_BOUNDS", str(arguments[2] / "bounds.json"))
    summary = stages.prepare(*arguments, **limits)
    # The first detection exited 0 with a summary; nothing may follow it.
    assert summary["status"] == "blocked" and "first" in summary["blocker"]
    assert summary["clips_attempted"] == 1
    first = summary["records"][0]
    assert first["detect"]["returncode"] == 0 and "calibrate" not in first
    assert (arguments[2] / "stages/first/detect.json").is_file()
    assert not (arguments[2] / "stages/second").exists()


def test_recorded_stage_failure_stays_and_the_run_continues(tmp_path, monkeypatch):
    _, summary = prepared(tmp_path, monkeypatch, "stage_failure")
    assert summary["status"] == "complete"
    second = summary["records"][1]
    assert second["detect"]["returncode"] == 2
    assert second["detect"]["status"] == "failed"
    assert second["calibrate"]["status"] == "ok"


@pytest.mark.parametrize("mode", ["hang", "crash", "abort_after_summary", "runtime"])
def test_native_failure_stops_without_retry(tmp_path, monkeypatch, mode):
    output, summary = prepared(tmp_path, monkeypatch, mode)
    assert summary["status"] == "blocked" and "second" in summary["blocker"]
    assert summary["clips_attempted"] == 2
    second = summary["records"][1]
    assert second["detect"]["timed_out"] is (mode == "hang")
    assert second["detect"]["returncode"] != 0
    assert "calibrate" not in second
    # A summary written before the failure does not make the stage finished.
    assert (output / "stages/second/detect.json").exists() is (
        mode in ("abort_after_summary", "runtime")
    )
    # The first clip's finished stages and every log remain for inspection.
    assert (output / "stages/first/calibrate.json").is_file()
    assert (output / "logs/second.detect.stderr").is_file()
    # The lock is free again once the process group is gone.
    with open(tmp_path / "device.lock", "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert os.path.getsize(output / "records.jsonl") > 0


@pytest.mark.parametrize(
    ("sign", "mode"),
    [(signal.SIGINT, "hang"), (signal.SIGTERM, "hang"), (signal.SIGINT, "stubborn")],
)
def test_interrupted_supervisor_stops_the_detector_before_freeing_the_lock(
    tmp_path, monkeypatch, sign, mode
):
    arguments = staged(tmp_path, monkeypatch, mode)
    names = ("inventory", "inventory-sha256", "output", "executable")
    command = [sys.executable, str(SCRIPT), "prepare"]
    for name, value in zip((*names, "detect-config", "lock"), arguments, strict=True):
        command += [f"--{name}", str(value)]
    command += ["--base-seconds", "600", "--seconds-per-frame", "1"]
    command += ["--calibrate-seconds", "30"]
    supervisor = subprocess.Popen(
        command, env=os.environ | {"PYTHONPATH": str(SCRIPT.parents[1])}
    )
    pid_file = arguments[2] / "stages/second/pid"
    deadline = time.monotonic() + 60
    while not pid_file.is_file() or not pid_file.read_text():
        assert supervisor.poll() is None and time.monotonic() < deadline
        time.sleep(0.05)
    detector = int(pid_file.read_text())
    supervisor.send_signal(sign)
    if mode == "stubborn":
        # A second signal during the ten-second grace must not cut cleanup short.
        time.sleep(0.75)
        supervisor.send_signal(sign)
    assert supervisor.wait(timeout=30) != 0
    # Once the supervisor is gone, so is its detector; then the lock is free.
    with pytest.raises(ProcessLookupError):
        os.kill(detector, 0)
    with open(tmp_path / "device.lock", "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (arguments[2] / "prepare.json").exists()


def test_stage_bytes_must_match_the_released_file_list(tmp_path):
    fixture = stage_tree(tmp_path)
    run = fixture.stages / "controlled"
    released = tmp_path / "released.json"
    write_json(
        released,
        [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in sorted(run.iterdir())
        ],
    )
    pin = file_sha256(fixture.inventory)
    arguments = (fixture.inventory, pin, fixture.stages)
    manifest = tmp_path / "a.json"
    document = stages.freeze(
        *arguments, tmp_path / "a", manifest, {}, (released, file_sha256(released))
    )
    shared = document["common_stages"]
    assert shared["stage_files_checked_against_released_list"] is True
    assert shared["bindings"]["released_files"]["sha256"] == file_sha256(released)
    # A self-consistent stage that is not the released one is refused.
    summary = read_json(run / "calibrate.json")
    write_json(run / "calibrate.json", summary | {"elapsed_s": 9.0}, overwrite=True)
    with pytest.raises(stages.StageError, match="released file list"):
        stages.freeze(
            *arguments,
            tmp_path / "b",
            tmp_path / "b.json",
            {},
            (released, file_sha256(released)),
        )
    with pytest.raises(stages.StageError, match="--released-sha256"):
        stages.freeze(
            *arguments, tmp_path / "c", tmp_path / "c.json", {}, (released, "0" * 64)
        )
    assert not (tmp_path / "b.json").exists() and not (tmp_path / "c.json").exists()


@pytest.mark.parametrize("damage", ["source", "timeline", "rows"])
def test_changed_source_stops_the_evaluation_instead_of_failing_the_model(
    tmp_path, monkeypatch, damage
):
    fixture = stage_tree(tmp_path)
    root, manifest = frozen(fixture)
    release = checkpoint(tmp_path / "model.pt", ModelConfig(width=16, depth=2), 1)
    provenance = {"release": {"checkpoint": {"sha256": release.checkpoint.sha256}}}
    monkeypatch.setattr(
        adapter, "checked_release", lambda _: (release, None, provenance)
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("no model may run on an untrusted source")

    monkeypatch.setattr(adapter, "lift_cached", forbidden)
    if damage == "rows":
        # A consistently repinned timeline that the stage rows do not match.
        timeline = read_json(fixture.clip.timeline_path)
        timeline["frames"][5]["pts"] += 1
        write_json(fixture.clip.timeline_path, timeline, overwrite=True)
        inventory = read_json(fixture.inventory)
        inventory["clips"][0]["timeline_sha256"] = file_sha256(
            fixture.clip.timeline_path
        )
        write_json(fixture.inventory, inventory, overwrite=True)
        document = read_json(manifest) | {
            "original_inventory_sha256": file_sha256(fixture.inventory)
        }
        write_json(manifest, document, overwrite=True)
    else:
        path = (
            fixture.clip.source_path
            if damage == "source"
            else fixture.clip.timeline_path
        )
        with open(path, "ab") as stream:
            stream.write(b" ")
    with pytest.raises(adapter.CachedInputError, match="controlled"):
        adapter.cached_real(
            tmp_path / "unused-release.json",
            fixture.inventory,
            root,
            manifest,
            file_sha256(manifest),
            tmp_path / "model",
        )
    assert not (tmp_path / "model/metrics.json").exists()
    assert read_json(tmp_path / "model/aborted.json")["status"] == "aborted"


@pytest.mark.parametrize("where", ["start", "start_and_cleanup"])
def test_interruption_while_the_detector_starts_still_stops_it(
    tmp_path, monkeypatch, where
):
    started = []

    class Interrupted(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            started.append(self.pid)
            # The signal lands after the child exists, before the caller has it.
            os.kill(os.getpid(), signal.SIGINT)

    stop_group = stages.stop_group

    def interrupted_cleanup(process):
        # A second signal as cleanup begins, and a third in the middle of it.
        os.kill(os.getpid(), signal.SIGINT)
        os.kill(process.pid, signal.SIGSTOP)
        os.kill(os.getpid(), signal.SIGTERM)
        stop_group(process)

    monkeypatch.setattr(stages.subprocess, "Popen", Interrupted)
    if where == "start_and_cleanup":
        monkeypatch.setattr(stages, "stop_group", interrupted_cleanup)
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            stages.supervised(
                [sys.executable, "-c", "import time; time.sleep(600)"],
                600,
                tmp_path / "log",
                tmp_path / "device.lock",
            )
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    finally:
        signal.signal(signal.SIGINT, previous)
    assert started and not stages.group_running(started[0])
    with open(tmp_path / "device.lock", "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_zombies_do_not_keep_a_group_running():
    child = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    deadline = time.monotonic() + 30
    # Unreaped, the exited child is a zombie that still answers killpg.
    while stages.group_running(child.pid):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    os.killpg(child.pid, 0)
    child.wait()
