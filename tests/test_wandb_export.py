"""The monitor runs without importing the training runtime or touching its files."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "wandb_export", Path(__file__).parents[1] / "scripts/wandb_export.py"
)
assert SPEC and SPEC.loader
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


def write(path, value):
    path.write_bytes(export.canonical(value) + b"\n")


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "training"
    root.mkdir()
    config = {
        "batch_size": 48,
        "max_steps": 100,
        "max_train_seconds": 1000,
        "max_job_seconds": 2000,
    }
    write(root / "config.resolved.json", config)
    write(
        root / "run.json", {"run_id": "abc123", "config_sha256": export.digest(config)}
    )
    write(
        root / "budget.json",
        {"run_id": "abc123", "step": 10, "train_seconds": 5, "job_seconds": 9},
    )
    return root


def loss(step=10):
    return {
        "step": step,
        "train_seconds": 5,
        "job_seconds": 9,
        "windows": step * 48,
        "loss": 0.3,
        "loss_units": "scaled radial objective",
        "step_seconds": 0.5,
        "gradient_norm": 1.2,
        "learning_rates": [0.0016],
        "optimizer_stability": [
            {"algorithm": "muon", "weight_rms": 0.2, "first_moment_rms": 0.01}
        ],
    }


def validation(invocation="first", step=10):
    return {
        "run_id": "abc123",
        "invocation_id": invocation,
        "resumed_from_step": 5,
        "step": step,
        "train_seconds": 5,
        "job_seconds": 9,
        "windows": step * 48,
        "modes": {
            "clean": {"val_xyz_m": 0.2, "views": [{"private": "data"}]},
            "corrupted": {"val_xyz_m": 0.3},
        },
    }


def reader(source):
    return export.Reader(source, source.parent / "monitoring", "tennis3d", "fixture")


def test_atomic_replacement_torn_line_and_read_only(source):
    path = source / "loss.jsonl"
    write(path, loss())
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    monitor = reader(source)
    _, _, events, pending = monitor.snapshot()
    first = events[0]["export/event_id"]
    assert not pending
    replacement = source / ".replacement"
    replacement.write_bytes(path.read_bytes() + export.canonical(loss(20))[:25])
    replacement.replace(path)
    _, _, events, pending = monitor.snapshot()
    assert events[0]["export/event_id"] == first
    assert pending == ["loss.jsonl: incomplete trailing line"]
    replacement.write_bytes(
        export.canonical(loss()) + b"\n" + export.canonical(loss(20)) + b"\n"
    )
    replacement.replace(path)
    _, _, events, pending = reader(source).snapshot()
    assert len(events) == 3 and not pending
    assert all(
        p.read_bytes() == before[p.name]
        for p in source.iterdir()
        if p.name != "loss.jsonl"
    )


def test_divergence_blocks_persistently_and_shrink_waits(source):
    path = source / "validation-history.jsonl"
    original = export.canonical(validation()) + b"\n"
    path.write_bytes(original)
    monitor = reader(source)
    monitor.snapshot()
    path.write_bytes(b"")
    with pytest.raises(export.Waiting, match="shorter"):
        reader(source).snapshot()
    write(path, validation("different"))
    for _ in range(2):
        with pytest.raises(export.Blocked, match="divergent"):
            reader(source).snapshot()
    assert json.loads((monitor.state_dir / "state.json").read_text())["journals"][
        path.name
    ] == [export.digest(validation())]
    path.write_bytes(original)
    reader(source).snapshot()


def test_replay_attempts_keep_custom_axes_and_distinct_event_ids(source):
    (source / "loss.jsonl").write_bytes(
        b"".join(export.canonical(loss(step)) + b"\n" for step in [10, 20, 10])
    )
    (source / "validation-history.jsonl").write_bytes(
        b"".join(
            export.canonical(validation(inv)) + b"\n" for inv in ["first", "resume"]
        )
    )
    _, _, events, _ = reader(source).snapshot()
    assert len({row["export/event_id"] for row in events}) == len(events)
    assert events[2]["export/replayed_step"] == 1
    assert events[2]["axes/training_step"] == 10
    assert events[2]["axes/accepted_windows"] == 480
    assert events[2]["by_time/throughput/windows_per_second"] == 96
    assert events[3]["export/validation_attempt"] == 1
    assert events[4]["export/validation_attempt"] == 2
    assert events[4]["export/invocation_id"] == "resume"
    assert not any("private" in key or "views" in key for key in events[3])


def test_identity_and_config_binding(source):
    monitor = reader(source)
    monitor.snapshot()
    write(source / "run.json", {"run_id": "different", "config_sha256": "bad"})
    with pytest.raises(export.Blocked, match="another trainer"):
        reader(source).snapshot()
    with pytest.raises(export.Blocked, match="outside"):
        export.Reader(source, source / "monitor", "tennis3d", "fixture")
    with pytest.raises(export.Blocked, match="project/entity"):
        export.Reader(source, monitor.state_dir, "other", "fixture")


def test_allocation_stop_is_not_training_completion(source):
    write(
        source / "summary.json",
        {
            "run_id": "abc123",
            "step": 10,
            "train_seconds": 5,
            "job_seconds": 9,
            "stop_reason": "requested",
        },
    )
    _, _, events, _ = reader(source).snapshot()
    assert events[-1]["allocation/stop_reason"] == "requested"
    assert not any("complete" in k for k in events[-1])


def test_invalid_complete_record_and_other_run_are_blocked(source):
    path = source / "validation-history.jsonl"
    path.write_bytes(b'{"torn":}\n')
    with pytest.raises(export.Blocked, match="invalid"):
        reader(source).snapshot()
    row = validation()
    row["run_id"] = "other"
    write(path, row)
    with pytest.raises(export.Blocked, match="invalid"):
        reader(source).snapshot()


class FakeSDK:
    __version__ = export.SDK_VERSION

    def __init__(self):
        self.remote_events = []
        self.logged = []
        self.initializations = []
        self.definitions = []

    def Settings(self, **kwargs):
        return kwargs

    def Api(self, **kwargs):
        return self

    def flush(self):
        pass

    def run(self, path):
        self.config = {"provenance": {"run_id": "abc123"}}
        return self

    def scan_history(self, **kwargs):
        assert kwargs["use_cache"] is False
        return [
            {"export/event_id": row["export/event_id"]} for row in self.remote_events
        ]

    def init(self, **kwargs):
        self.initializations.append(kwargs)
        return self

    def define_metric(self, *args, **kwargs):
        self.definitions.append((args, kwargs))

    def log(self, row):
        self.logged.append(row)

    def finish(self):
        self.remote_events.extend(self.logged)
        self.logged = []

    default_entity = "personal"


def test_server_ack_and_crash_recovery_do_not_use_training_step(source, monkeypatch):
    write(source / "loss.jsonl", loss())
    sdk = FakeSDK()
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    monitor = reader(source)
    run, config, events, _ = monitor.snapshot()
    backend = export.Backend(monitor, 1)
    acknowledged = backend.connect(run, config)
    backend.publish(events, acknowledged)
    backend.publish(events, acknowledged)
    assert len(sdk.logged) == len(events)
    assert backend.acknowledge() == set()  # SDK accepted, server has not acknowledged.
    assert not monitor.state["acknowledged"]
    backend.finish()
    # Simulate a crash after upload but before saving acknowledgement.
    restarted = export.Backend(reader(source), 1)
    acknowledged = restarted.connect(run, config)
    restarted.publish(events, acknowledged)
    assert not sdk.logged
    assert acknowledged == {row["export/event_id"] for row in events}
    assert sdk.initializations[-1]["id"] == "abc123"
    assert sdk.initializations[-1]["resume"] == "allow"
    settings = sdk.initializations[-1]["settings"]
    assert settings["x_disable_stats"] and settings["x_disable_meta"]
    assert all("step" not in row for row in events)
    assert (monitor.state_dir / "status.json").exists() is False


def test_credentials_private_and_never_recorded(source, monkeypatch):
    directory = source.parent / "private"
    directory.mkdir(mode=0o700)
    path = directory / "key"
    path.write_text("fixture-secret")
    path.chmod(0o644)
    with pytest.raises(export.Blocked, match="private"):
        export.authenticate(path)
    path.chmod(0o600)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    export.authenticate(path)
    assert export.os.environ["WANDB_API_KEY"] == "fixture-secret"
    monkeypatch.delenv("WANDB_API_KEY")


@pytest.fixture
def cli_cleanup():
    import signal

    old = {key: signal.getsignal(key) for key in (signal.SIGTERM, signal.SIGINT)}
    yield
    for key, handler in old.items():
        signal.signal(key, handler)


def cli_args(source):
    return [
        "--source",
        str(source),
        "--state-dir",
        str(source.parent / "monitoring"),
        "--entity",
        "personal",
        "--drain-timeout",
        "1",
    ]


def test_first_live_verifier_works_alongside_writer_and_preserves_state(
    source, monkeypatch, cli_cleanup
):
    import fcntl

    write(source / "loss.jsonl", loss())
    sdk = FakeSDK()
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    monitor = reader(source)
    _, _, events, _ = monitor.snapshot()
    sdk.remote_events = events
    state_before = monitor.path.read_bytes()
    with (monitor.state_dir / "exporter.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert (
            export.main(cli_args(source) + ["--verify-only", "--minimum-step", "1"])
            == 0
        )
    assert monitor.path.read_bytes() == state_before
    assert not sdk.initializations
    status = json.loads((monitor.state_dir / "verification.json").read_bytes())
    assert status["status"] == "verified"
    assert status["run_id"] == "abc123" and status["acknowledged_latest_step"] == 10
    assert status["verified_metrics"][0]["by_step/train/loss"] == 0.3
    assert status["run_url"].endswith("personal/tennis3d/abc123")


def test_network_error_reports_retry_without_secret_exception_text(
    source, monkeypatch, cli_cleanup
):
    sdk = FakeSDK()

    def fail(**kwargs):
        raise OSError("secret-must-not-be-output")

    monkeypatch.setattr(sdk, "Api", fail)
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    assert export.main(cli_args(source)) == 2
    status = (source.parent / "monitoring/status.json").read_text()
    assert "disconnected" in status and "OSError" in status
    assert "secret" not in status


def test_sigterm_finishes_and_server_verifies_final_batch(
    source, monkeypatch, cli_cleanup
):
    import os
    import signal

    write(source / "loss.jsonl", loss())
    sdk = FakeSDK()
    original = sdk.log

    def stop(row):
        original(row)
        if len(sdk.logged) == 1:
            with (source / "loss.jsonl").open("ab") as stream:
                stream.write(export.canonical(loss(20)) + b"\n")
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(sdk, "log", stop)
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    assert export.main(cli_args(source) + ["--watch"]) == 0
    assert len(sdk.remote_events) == 3
    assert [
        row["axes/training_step"]
        for row in sdk.remote_events
        if row["export/source"] == "loss.jsonl"
    ] == [10, 20]
    assert (
        json.loads((source.parent / "monitoring/status.json").read_bytes())["status"]
        == "drained"
    )


def test_absent_source_does_not_initialize_sdk(tmp_path, monkeypatch, cli_cleanup):
    source = tmp_path / "not-yet-created"
    sdk = FakeSDK()
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    assert export.main(cli_args(source)) == 2
    assert not sdk.initializations
    assert (
        json.loads((source.parent / "monitoring/status.json").read_bytes())["status"]
        == "waiting_for_source"
    )


def test_requested_entity_binds_preexisting_unconnected_state(
    source, monkeypatch, cli_cleanup
):
    sdk = FakeSDK()
    sdk.default_entity = "wrongteam"
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    reader(source).snapshot()  # Tracking state exists with entity null before auth.
    assert export.main(cli_args(source)) == 0
    assert sdk.initializations[0]["entity"] == "personal"
