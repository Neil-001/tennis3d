import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tennis3d.artifacts import read_json
from tennis3d.autoresearch.dashboard import make_server
from tennis3d.autoresearch.log import append, lock, read_events, reconcile, snapshot
from tennis3d.autoresearch.protocol import Protocol


def start(run: Path, trial="trial-1", segment="smoke"):
    if not (run / "experiments.jsonl").exists():
        append(run, "segment", segment=segment, profile="smoke")
    append(run, "started", segment=segment, trial=trial, hypothesis="test")


def test_torn_tail_recovery_preserves_bytes_and_prior_results(tmp_path):
    start(tmp_path)
    append(
        tmp_path,
        "finished",
        segment="smoke",
        trial="trial-1",
        status="kept",
        score=2.5,
        checks_passed=True,
    )
    path = tmp_path / "experiments.jsonl"
    with path.open("ab") as stream:
        stream.write(b'{"event":"progr')
    prior = path.read_bytes()
    events, warnings = read_events(path)
    assert len(events) == 3 and warnings
    append(tmp_path, "control", action="stop")
    assert path.read_bytes().startswith(prior)
    state = snapshot(tmp_path)
    assert state["best"]["smoke"]["score"] == 2.5
    assert state["stopped"] and state["warnings"]
    with path.open("ab") as stream:
        stream.write(b"corrupt interior\n")
    with pytest.raises(ValueError, match="corrupt experiment log"):
        append(tmp_path, "control", action="resume")


def test_stop_survives_reconciliation_and_incomplete_trial_never_scores(tmp_path):
    start(tmp_path)
    append(tmp_path, "control", action="stop")
    with lock(tmp_path / ".trial.lock", blocking=False):
        with (
            pytest.raises(ValueError, match="still active"),
            lock(tmp_path / ".trial.lock", blocking=False),
        ):
            pass
        assert reconcile(tmp_path) == ["trial-1"]
    state = snapshot(tmp_path)
    assert state["stopped"]
    assert state["trials"][0]["status"] == "interrupted"
    assert not state["best"]
    append(tmp_path, "control", action="resume")
    assert not snapshot(tmp_path)["stopped"]


def test_scores_require_passed_checks_and_improvement(tmp_path):
    start(tmp_path)
    for score, status, checks in [
        (0, "crashed", True),
        (float("nan"), "kept", True),
        (1, "kept", False),
    ]:
        with pytest.raises(ValueError):
            append(
                tmp_path,
                "finished",
                segment="smoke",
                trial="trial-1",
                score=score,
                status=status,
                checks_passed=checks,
            )
    append(
        tmp_path,
        "finished",
        segment="smoke",
        trial="trial-1",
        status="kept",
        score=1,
        checks_passed=True,
    )
    start(tmp_path, "trial-2")
    append(
        tmp_path,
        "finished",
        segment="smoke",
        trial="trial-2",
        status="discarded",
        score=2,
        checks_passed=True,
    )
    assert snapshot(tmp_path)["best"]["smoke"]["trial"] == "trial-1"


def test_protocol_budgets_cannot_be_shortened_in_standard():
    assert Protocol().train_seconds == 300
    with pytest.raises(ValueError, match="300s"):
        Protocol(train_seconds=1)
    with pytest.raises(ValueError, match="smoke protocols"):
        Protocol(profile="smoke", train_seconds=300)
    assert (
        Protocol(profile="smoke", train_seconds=0.1, total_seconds=10).profile
        == "smoke"
    )


def test_dashboard_is_read_only_and_reports_corruption(tmp_path):
    start(tmp_path)
    trial = tmp_path / "trial-1"
    trial.mkdir()
    (trial / "curve.jsonl").write_text('{"step":1,"loss":0.5,"train_seconds":1}\n')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with make_server(tmp_path, 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(base + "/api/snapshot") as response:
            assert json.load(response)["trials"][0]["status"] == "running"
        with urllib.request.urlopen(base + "/api/trial?id=trial-1") as response:
            assert json.load(response)["curve"][0]["loss"] == 0.5
        for path in ("/api/stop", "/experiments.jsonl", "/../pyproject.toml"):
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(base + path)
            assert error.value.code == 404
        request = urllib.request.Request(
            base + "/api/snapshot", method="POST", data=b"{}"
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 501
        assert before == {
            p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()
        }
        with (tmp_path / "experiments.jsonl").open("ab") as stream:
            stream.write(b"bad\n")
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base + "/api/snapshot")
        assert error.value.code == 422
        server.shutdown()
        thread.join()


def test_supervisor_deadline_stop_and_orphan_lock(tmp_path):
    import os
    import subprocess
    import sys
    import time

    from tennis3d.autoresearch.execution import TrialFailure, run_process

    with lock(tmp_path / ".trial.lock", blocking=False) as fd:
        for stopped, status in [
            (lambda: False, "timed-out"),
            (lambda: True, "interrupted"),
        ]:
            with pytest.raises(TrialFailure) as error:
                run_process(
                    [sys.executable, "-c", "import time; time.sleep(20)"],
                    cwd=tmp_path,
                    output=tmp_path / "process.log",
                    deadline=time.perf_counter() + 0.3,
                    lock_fd=fd,
                    stopped=stopped,
                    update=lambda: None,
                )
            assert error.value.status == status
        # Child retains the flock after the supervisor descriptor is closed.
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(20)"],
            pass_fds=(fd,),
            start_new_session=True,
        )
    try:
        with (
            pytest.raises(ValueError, match="still active"),
            lock(tmp_path / ".trial.lock", blocking=False),
        ):
            pass
    finally:
        os.kill(child.pid, 9)
        child.wait()
    with lock(tmp_path / ".trial.lock", blocking=False):
        pass


def test_frozen_segment_and_candidate_archive(tmp_path):
    import subprocess

    from tennis3d.autoresearch.protocol import identities, verify_sources
    from tennis3d.autoresearch.segments import create_segment, read_segment

    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("/runs/\n")
    (root / "fixed.py").write_text("evaluator = 1\n")
    (root / "configs").mkdir()
    (root / "configs/lifter.yaml").write_text("width: 64\n")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    run = root / "runs/search"
    protocol = Protocol(profile="smoke", train_seconds=1, total_seconds=5)
    segment = create_segment(
        root,
        run,
        "smoke-cpu",
        protocol,
        data={"manifest": "fixture"},
        hardware={"device": "cpu"},
    )
    assert read_segment(run) == segment
    (root / "configs/lifter.yaml").write_text("width: 128\n")
    assert (
        verify_sources(root, segment["frozen_files"])["candidate_sha256"]
        != segment["reference"]["candidate_sha256"]
    )
    assert (
        run / "segments/smoke-cpu/candidate/configs/lifter.yaml"
    ).read_text() == "width: 64\n"
    identity = identities(root)
    (root / "configs/lifter.yaml").write_text("width: 32\n")
    with pytest.raises(ValueError, match="during the trial"):
        verify_sources(root, segment["frozen_files"], identity["candidate_files"])
    (root / "new-evaluator.py").write_text("changed = True\n")
    with pytest.raises(ValueError, match="frozen files changed"):
        verify_sources(root, segment["frozen_files"])
    record = run / "segments/smoke-cpu/segment.json"
    value = json.loads(record.read_text())
    value["protocol"]["train_seconds"] = 2
    record.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="durable log identity"):
        read_segment(run)


def test_cli_explicit_stop_resume(tmp_path, capsys):
    from tennis3d.cli import main

    start(tmp_path)
    assert main(["experiment-state", "stop", "--run", str(tmp_path)]) == 0
    with lock(tmp_path / ".trial.lock", blocking=False):
        assert main(["experiment-state", "resume", "--run", str(tmp_path)]) == 2
    assert snapshot(tmp_path)["stopped"]
    assert main(["experiment-state", "resume", "--run", str(tmp_path)]) == 0
    state = snapshot(tmp_path)
    assert not state["stopped"]
    assert state["trials"][0]["status"] == "interrupted"
    capsys.readouterr()


def test_supervisor_setup_deadline_is_independent(tmp_path, monkeypatch):
    import argparse
    import time

    from tennis3d.artifacts import json_sha256, write_json
    from tennis3d.lifting import experiment

    run = tmp_path / "run"
    segment = {
        "segment": "tiny",
        "protocol": Protocol(
            profile="smoke", train_seconds=0.001, total_seconds=0.03
        ).model_dump(mode="json"),
        "frozen_files": {},
    }
    segment["sha256"] = json_sha256(segment)
    write_json(run / "segments/tiny/segment.json", segment)
    append(run, "segment", **segment)
    monkeypatch.setattr(experiment, "checkout", lambda: tmp_path / "outside")
    monkeypatch.setattr(experiment, "verify_sources", lambda *a: time.sleep(2))
    args = argparse.Namespace(
        output=run / "timeout",
        segment="tiny",
        config=None,
        hypothesis="deadline regression",
        baseline=None,
        loss_label=None,
    )
    before = time.perf_counter()
    assert experiment.trial(args) == 2
    assert time.perf_counter() - before < 1
    assert snapshot(run)["trials"][0]["status"] == "timed-out"
    assert snapshot(run)["best"] == {}


def test_observer_rejects_uncounted_optimizer_and_short_budget():
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    from tennis3d.autoresearch.observer import Observer
    from tennis3d.lifting.model import ModelConfig
    from tennis3d.lifting.settings import TrainConfig

    observer = Observer(
        TrainConfig(model=ModelConfig(width=16, depth=2), device="cpu"),
        SimpleNamespace(stop_requested=False),
    )
    observer.install()
    try:
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([parameter])
        parameter.grad = torch.ones(1)
        with pytest.raises(ValueError, match="uncounted optimizer"):
            optimizer.step()
        with pytest.raises(ValueError, match="incomplete or uncounted"):
            observer.finish({"step": 0})
    finally:
        observer.close()


@pytest.mark.parametrize("cut", [0, 1, 10, 50, 10000])
def test_recovery_survives_a_second_torn_append(tmp_path, monkeypatch, cut):
    from tennis3d.autoresearch import log

    start(tmp_path)
    append(
        tmp_path,
        "finished",
        segment="smoke",
        trial="trial-1",
        status="kept",
        score=1.0,
        checks_passed=True,
    )
    path = tmp_path / "experiments.jsonl"
    with path.open("ab") as stream:
        stream.write(b'{"event":"progr')
    original = path.read_bytes()
    real_append = log._append_bytes

    def torn_append(target, payload):
        real_append(target, payload[:cut])
        raise OSError("simulated interrupted recovery write")

    with monkeypatch.context() as patch:
        patch.setattr(log, "_append_bytes", torn_append)
        with pytest.raises(OSError, match="interrupted recovery"):
            append(tmp_path, "control", action="stop")
    damaged = path.read_bytes()
    journal = tmp_path / "experiments.jsonl.recovery.json"
    assert journal.exists()
    state = snapshot(tmp_path)
    assert state["best"]["smoke"]["score"] == 1.0
    assert state["warnings"]
    assert path.read_bytes() == damaged  # The dashboard reader cannot repair.
    append(tmp_path, "control", action="resume")
    state = snapshot(tmp_path)
    assert not state["stopped"] and state["best"]["smoke"]["score"] == 1.0
    assert state["warnings"] and not journal.exists()
    assert path.read_bytes().startswith(original)
    with path.open("ab") as stream:
        stream.write(b"unrelated interior corruption\n")
    with pytest.raises(ValueError, match="corrupt experiment log"):
        append(tmp_path, "control", action="resume")


def test_checkout_requires_editable_git_repository(tmp_path, monkeypatch):
    from tennis3d.autoresearch import protocol

    package = tmp_path / "installed/tennis3d/autoresearch"
    package.mkdir(parents=True)
    monkeypatch.setattr(protocol, "__file__", str(package / "protocol.py"))
    with pytest.raises(ValueError, match="editable tennis3d Git checkout"):
        protocol.checkout()


@pytest.mark.parametrize(
    ("summary", "stop_before_result"),
    [({}, False), (None, False), ({}, True), ({}, "signal")],
)
def test_result_failure_or_stop_finishes_unscored(
    tmp_path, monkeypatch, summary, stop_before_result
):
    from types import SimpleNamespace

    from tennis3d.artifacts import write_json
    from tennis3d.lifting import experiment

    run = tmp_path / "run"
    segment = {
        "segment": "smoke",
        "protocol": Protocol(
            profile="smoke",
            train_seconds=0.1,
            total_seconds=10,
            loss_interval_seconds=100,
        ).model_dump(mode="json"),
        "frozen_files": {},
        "training": {"batch_size": 1},
    }
    append(run, "segment", **segment)
    monkeypatch.setattr(experiment, "checkout", lambda: tmp_path / "outside")
    monkeypatch.setattr(experiment, "read_segment", lambda *a: segment)
    monkeypatch.setattr(experiment, "loss_label", lambda *a: "fixture objective")
    monkeypatch.setattr(experiment, "archive_candidate", lambda *a: None)
    monkeypatch.setattr(
        experiment,
        "verify_sources",
        lambda *a: {
            "revision": "fixture",
            "candidate_sha256": "fixture",
            "candidate_files": {},
        },
    )
    reads = []

    def counted_snapshot(path):
        reads.append(path)
        return snapshot(path)

    monkeypatch.setattr(experiment, "snapshot", counted_snapshot)

    def fake_child(stage, request, root, fd, deadline, stopped, update):
        before = len(reads)
        for _ in range(20):
            assert not stopped()
        assert len(reads) - before <= 1
        output = request.parent
        if stage == "resolve":
            write_json(output / "config.resolved.json", {"batch_size": 1})
        elif stage == "checks":
            write_json(output / "checks.json", {"passed": True, "parameter_count": 1})
            write_json(
                output / "metrics.json",
                {
                    "modes": {"corrupted": {"val_xyz_m": 1}},
                    "checkpoint_sha256": "fixture",
                },
            )
            write_json(output / "observed.json", {"batches": 1})
            write_json(output / "training/summary.json", summary)
            if stop_before_result == "signal":
                import signal

                signal.raise_signal(signal.SIGTERM)
                assert stopped()  # The cached poll interval cannot delay signals.
            elif stop_before_result:
                append(run, "control", action="stop")

    monkeypatch.setattr(experiment, "child", fake_child)
    args = SimpleNamespace(
        output=run / "broken",
        segment="smoke",
        config=None,
        hypothesis="malformed result",
        baseline=None,
        loss_label=None,
    )
    assert experiment.trial(args) == 2
    result = snapshot(run)["trials"][0]
    assert result["status"] == (
        "interrupted" if stop_before_result else "checks-failed"
    )
    assert result["score"] is None
    assert not snapshot(run)["best"]
    assert (args.output / "result.json").exists()


@pytest.mark.parametrize(
    ("total", "trainable", "baseline", "passes"),
    [
        (1_499_999, 1_499_999, None, False),
        (1_499_999, 1_499_999, "reference", False),
        (1_500_000, 1_500_000, None, True),
        (1_633_731, 1_633_731, "reference", True),
        (1_999_999, 1_999_999, None, True),
        (2_000_000, 2_000_000, None, False),
        (1_700_000, 1_400_000, None, False),  # frozen weights do not count
        (1_700_000, 1_500_000, None, True),
        (212_963, 212_963, "small", True),
        (2_000_000, 2_000_000, "small", False),
    ],
)
def test_parameter_band_and_small_baseline_exemption(
    total, trainable, baseline, passes
):
    from tennis3d.autoresearch.worker import check_parameter_count

    if passes:
        check_parameter_count(total, trainable, baseline)
    else:
        with pytest.raises(ValueError, match="outside"):
            check_parameter_count(total, trainable, baseline)


def test_training_stage_rejects_underfloor_model_before_training():
    pytest.importorskip("torch")
    from tennis3d.autoresearch import worker
    from tennis3d.lifting.model import ModelConfig
    from tennis3d.lifting.settings import TrainConfig

    request = {
        "training": TrainConfig(
            model=ModelConfig(width=64, depth=4), device="cpu"
        ).model_dump(mode="json")
    }
    with pytest.raises(ValueError, match="outside"):
        worker.training(request, Path("unused"))


@pytest.mark.parametrize(
    ("baseline", "status", "retained"),
    [("small", "discarded", False), ("reference", "kept", True)],
)
def test_small_baseline_is_recorded_but_never_incumbent(
    tmp_path, monkeypatch, baseline, status, retained
):
    from types import SimpleNamespace

    from tennis3d.artifacts import write_json
    from tennis3d.lifting import experiment

    run = tmp_path / "run"
    model = {"width": 128, "depth": 8}
    segment = {
        "segment": "smoke",
        "protocol": Protocol(
            profile="smoke",
            train_seconds=0.1,
            total_seconds=10,
            loss_interval_seconds=100,
        ).model_dump(mode="json"),
        "frozen_files": {},
        "training": {"batch_size": 1, "model": model},
        "reference": {"candidate_sha256": "fixture", "candidate_files": {}},
    }
    append(run, "segment", **segment)
    monkeypatch.setattr(experiment, "checkout", lambda: tmp_path / "outside")
    monkeypatch.setattr(experiment, "read_segment", lambda *a: segment)
    monkeypatch.setattr(experiment, "loss_label", lambda *a: "fixture objective")
    monkeypatch.setattr(experiment, "archive_candidate", lambda *a: None)
    monkeypatch.setattr(
        experiment,
        "verify_sources",
        lambda *a: {
            "revision": "fixture",
            "candidate_sha256": "fixture",
            "candidate_files": {},
        },
    )
    requests = []

    def fake_child(stage, request, root, fd, deadline, stopped, update):
        output = request.parent
        if stage == "resolve":
            requests.append(read_json(request))
            width, depth = (64, 4) if baseline == "small" else (128, 8)
            write_json(
                output / "config.resolved.json",
                {"batch_size": 1, "model": {"width": width, "depth": depth}},
            )
        elif stage == "checks":
            write_json(output / "checks.json", {"passed": True, "parameter_count": 1})
            write_json(
                output / "metrics.json",
                {
                    "modes": {"corrupted": {"val_xyz_m": 1}},
                    "checkpoint_sha256": "fixture",
                },
            )
            write_json(output / "observed.json", {"batches": 1})
            write_json(output / "training/summary.json", {"peak_allocated_bytes": 0})

    monkeypatch.setattr(experiment, "child", fake_child)
    args = SimpleNamespace(
        output=run / baseline,
        segment="smoke",
        config=None,
        hypothesis="baseline",
        baseline=baseline,
        loss_label=None,
    )
    assert experiment.trial(args) == 0
    assert requests[0]["baseline"] == baseline
    state = snapshot(run)
    trial = state["trials"][0]
    assert trial["status"] == status
    assert trial["score"] == 1
    assert trial["diagnostic"] is (baseline == "small")
    assert bool(state["best"]) is retained


def test_diagnostic_baseline_cannot_be_kept_in_log(tmp_path):
    start(tmp_path, "small")
    append(tmp_path, "progress", segment="smoke", trial="small", baseline="small")
    append(
        tmp_path,
        "finished",
        segment="smoke",
        trial="small",
        status="kept",
        score=1.0,
        checks_passed=True,
    )
    with pytest.raises(ValueError, match="diagnostic"):
        snapshot(tmp_path)


def test_incremental_stop_and_writer_reads_do_not_scale_with_prefix(
    tmp_path, monkeypatch
):
    from tennis3d.artifacts import canonical_json
    from tennis3d.autoresearch.log import StopReader

    start(tmp_path)
    path = tmp_path / "experiments.jsonl"
    progress = (
        canonical_json(
            {
                "event": "progress",
                "version": 1,
                "time": 1,
                "segment": "smoke",
                "trial": "trial-1",
                "phase": "train",
            }
        )
        + b"\n"
    )
    with path.open("ab") as stream:
        stream.write(progress * 20000)
    append(tmp_path, "control", action="resume")  # Prime the writer's prefix.
    reader = StopReader(tmp_path)
    real_open = Path.open
    reads = []

    class CountReads:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, *args):
            value = self.stream.read(*args)
            reads.append(len(value))
            return value

    def counted_open(target, mode="r", *args, **kwargs):
        stream = real_open(target, mode, *args, **kwargs)
        return CountReads(stream) if target == path and mode == "rb" else stream

    monkeypatch.setattr(Path, "open", counted_open)
    for _ in range(100):
        assert not reader.poll()
    assert sum(reads) == 0
    before = path.stat().st_size
    for action in ("stop", "resume", "stop"):
        append(tmp_path, "control", action=action)
        for _ in range(20):
            assert reader.poll() == (action == "stop")
    # Each new record is read by the stop reader and at most the next writer.
    assert sum(reads) < 2 * (path.stat().st_size - before) + 200
    assert reader.tail.offset == path.stat().st_size
    with make_server(tmp_path, 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}/api/snapshot"
        with urllib.request.urlopen(url) as response:
            assert json.load(response)["stopped"]
        reads.clear()
        for _ in range(10):
            with urllib.request.urlopen(url) as response:
                assert json.load(response)["stopped"]
        assert sum(reads) == 0
        before = path.stat().st_size
        append(tmp_path, "control", action="resume")
        with urllib.request.urlopen(url) as response:
            assert not json.load(response)["stopped"]
        assert sum(reads) < 2 * (path.stat().st_size - before) + 200
        server.shutdown()
        thread.join()


@pytest.mark.parametrize("cut", [0, 1, 10, 50, 10000])
def test_incremental_stop_preserves_interrupted_recovery(tmp_path, monkeypatch, cut):
    from tennis3d.autoresearch import log
    from tennis3d.autoresearch.log import StopReader

    append(tmp_path, "control", action="stop")
    reader = StopReader(tmp_path)
    assert reader.poll()
    path = tmp_path / "experiments.jsonl"
    with path.open("ab") as stream:
        stream.write(b'{"event":"cont')
    assert reader.poll() and reader.tail.warnings
    real_append = log._append_bytes

    def torn_append(target, payload):
        real_append(target, payload[:cut])
        raise OSError("interrupted repair")

    with monkeypatch.context() as patch:
        patch.setattr(log, "_append_bytes", torn_append)
        with pytest.raises(OSError):
            append(tmp_path, "control", action="resume")
    assert reader.poll() and reader.tail.warnings
    append(tmp_path, "control", action="resume")
    assert not reader.poll()
    assert reader.tail.warnings
    with path.open("ab") as stream:
        stream.write(b"unrelated corruption\n")
    with pytest.raises(ValueError, match="corrupt experiment log"):
        reader.poll()


def test_stop_reader_accepts_copied_history_without_lock(tmp_path):
    from tennis3d.autoresearch.log import StopReader

    append(tmp_path / "source", "control", action="stop")
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / "experiments.jsonl").write_bytes(
        (tmp_path / "source/experiments.jsonl").read_bytes()
    )
    assert StopReader(copy).poll()
    assert [p.name for p in copy.iterdir()] == ["experiments.jsonl"]


@pytest.mark.parametrize("endpoint", ["/api/snapshot", "/api/trial?id=trial-1"])
def test_slow_dashboard_client_does_not_hold_history_lock(
    tmp_path, monkeypatch, endpoint
):
    from typing import Any, cast

    start(tmp_path)
    waiting, release = threading.Event(), threading.Event()
    first_result = []
    with make_server(tmp_path, 0) as server:
        original = cast(Any, server.RequestHandlerClass).respond

        def stalled_response(handler, *args, **kwargs):
            if not waiting.is_set():
                waiting.set()
                release.wait(5)
            return original(handler, *args, **kwargs)

        monkeypatch.setattr(server.RequestHandlerClass, "respond", stalled_response)
        serving = threading.Thread(target=server.serve_forever, daemon=True)
        serving.start()
        base = f"http://127.0.0.1:{server.server_port}"

        def first_request():
            with urllib.request.urlopen(base + endpoint, timeout=6) as response:
                first_result.append(json.load(response))

        first = threading.Thread(target=first_request)
        first.start()
        try:
            assert waiting.wait(2)
            # A stalled response may not block another snapshot's projection.
            with urllib.request.urlopen(base + "/api/snapshot", timeout=1) as response:
                assert json.load(response)["trials"][0]["trial"] == "trial-1"
        finally:
            release.set()
            first.join(6)
            server.shutdown()
            serving.join()
        assert first_result
