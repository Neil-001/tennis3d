"""An interrupted legacy producer retains predictions and reports across phases."""

import os
import shutil
import signal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import continue_synthetic_evaluation as continuation
from tennis3d.artifacts import file_sha256, read_json, write_json
from tennis3d.evaluation.delivery import _synthetic_released
from tennis3d.evaluation.prediction_journal import PredictionJournal
from tennis3d.lifting.data import ViewDataset
from tennis3d.lifting.model import Lifter, ModelConfig

ROOT = Path(__file__).resolve().parents[1]


def warning_fit(numeric, metadata):
    result = continuation.timed_fit(numeric, metadata)
    os.kill(os.getppid(), signal.SIGUSR1)
    return result


def failing_fit(numeric, metadata):
    return None, "controlled fit failure", 0.25, 0.2


def no_fit(numeric, metadata):
    raise AssertionError("finished physical fit must not restart")


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    import tennis3d.lifting.checkpoint as checkpoint_module

    torch.set_num_threads(2)
    torch.manual_seed(814)
    model = Lifter(ModelConfig(width=16, depth=2))
    monkeypatch.setattr(checkpoint_module, "load_lifter", lambda *args: (model, {}))
    source = ROOT / "tests/fixtures/lifting/val.manifest.json"
    release = SimpleNamespace(
        checkpoint=SimpleNamespace(path=source, sha256=file_sha256(source)), step=1
    )
    provenance = {"release": {"checkpoint": {"sha256": file_sha256(source)}}}
    with ViewDataset(source) as dataset:
        expected = {
            "rallies": len({r["rally_id"] for r in dataset.records}),
            "views": len(dataset),
            "frames": sum(r["frames"] for r in dataset.records),
        }
    identity = {"fixture": "same immutable producer"}
    baseline = tmp_path / "baseline"
    with PredictionJournal(baseline, identity) as journal:
        result = _synthetic_released(
            release, provenance, source, journal, "cpu", expected_counts=expected
        )
    return source, release, provenance, expected, identity, baseline, result


def timing():
    return {"worker_cpu_s": 0.0, "worker_wall_s": 0.0, "fits_without_timing": 0}


def test_partial_mode_inference_resumes_and_complete_views_never_predict(
    fixture, tmp_path, monkeypatch
):
    import tennis3d.inference.predict as predictor

    source, release, _, expected, identity, baseline, _ = fixture
    output = tmp_path / "continued"
    with PredictionJournal(output, identity):
        pass
    (output / "predictions").mkdir()
    shutil.copyfile(
        baseline / "predictions/00000-corrupted.npz",
        output / "predictions/00000-corrupted.npz",
    )
    original, calls = predictor.predict_view, []

    def predict(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(predictor, "predict_view", predict)
    with PredictionJournal(output, identity, resume=True) as journal:
        seal = continuation.inference(release, source, journal, "cpu", expected)
        assert journal.counts["predictions_created"] == 1
        assert journal.counts["predictions_reused"] == 1
    assert len(calls) == 1
    with (
        np.load(output / "predictions/00000-clean.npz") as resumed,
        np.load(baseline / "predictions/00000-clean.npz") as fresh,
    ):
        for key in ("xyz", "covered", "disagreement"):
            np.testing.assert_array_equal(resumed[key], fresh[key])
    monkeypatch.setattr(
        predictor,
        "predict_view",
        lambda *a, **k: pytest.fail("finished prediction restarted"),
    )
    with PredictionJournal(output, identity, resume=True) as journal:
        assert continuation.inference(release, source, journal, "cpu", expected) == seal
        assert journal.counts["predictions_created"] == 0


@pytest.mark.parametrize("interrupt", [False, True])
def test_parallel_cpu_reports_and_metrics_equal_original_and_resume_without_fit(
    fixture, tmp_path, interrupt
):
    source, release, provenance, expected, identity, baseline, reference = fixture
    output = tmp_path / "continued"
    with PredictionJournal(output, identity):
        pass
    shutil.copytree(baseline / "predictions", output / "predictions")
    with (
        PredictionJournal(output, identity, resume=True) as journal,
        ViewDataset(source) as dataset,
    ):
        seal = {
            "producer_identity": identity,
            "source": {
                "identity": dataset.identity,
                "manifest_sha256": dataset.manifest_sha256,
                "records_sha256": continuation.json_sha256(dataset.records),
                "counts": expected,
            },
            "files": continuation.inventory(journal, dataset),
        }
    if interrupt:
        with (
            continuation.ReplayJournal(output, identity, resume=True) as journal,
            pytest.raises(InterruptedError),
        ):
            continuation.analysis(
                release,
                provenance,
                source,
                journal,
                seal,
                2,
                expected,
                timing(),
                fit=warning_fit,
            )
        assert not (output / "metrics.json").exists()
        assert (output / "physical-journal/00000.json").exists()
    with continuation.ReplayJournal(output, identity, resume=True) as journal:
        actual = continuation.analysis(
            release,
            provenance,
            source,
            journal,
            seal,
            2,
            expected,
            timing(),
            **({"fit": no_fit} if interrupt else {}),
        )
        assert journal.counts["predictions_created"] == 0
    for key in ("metrics", "exported_result_metrics", "physical"):
        assert actual[key] == reference[key]
    assert (output / "physical/00000.json").read_bytes() == (
        baseline / "physical/00000.json"
    ).read_bytes()


def test_failed_fit_has_cost_and_no_final_result(fixture, tmp_path):
    source, release, provenance, expected, identity, baseline, _ = fixture
    output = tmp_path / "continued"
    with PredictionJournal(output, identity):
        pass
    shutil.copytree(baseline / "predictions", output / "predictions")
    with (
        PredictionJournal(output, identity, resume=True) as journal,
        ViewDataset(source) as dataset,
    ):
        seal = {
            "producer_identity": identity,
            "source": {
                "identity": dataset.identity,
                "manifest_sha256": dataset.manifest_sha256,
                "records_sha256": continuation.json_sha256(dataset.records),
                "counts": expected,
            },
            "files": continuation.inventory(journal, dataset),
        }
    costs = timing()
    with (
        continuation.ReplayJournal(output, identity, resume=True) as journal,
        pytest.raises(ValueError, match="controlled fit failure"),
    ):
        continuation.analysis(
            release,
            provenance,
            source,
            journal,
            seal,
            2,
            expected,
            costs,
            fit=failing_fit,
        )
    assert costs["worker_cpu_s"] == 0.2
    assert costs["worker_wall_s"] == 0.25
    assert not (output / "metrics.json").exists()
    assert not (output / "physical-journal/00000.json").exists()


@pytest.mark.parametrize(
    "change", ["metadata", "producer", "extra", "source", "missing"]
)
def test_changed_or_missing_intermediates_fail_closed(fixture, tmp_path, change):
    source, _, _, _, identity, baseline, _ = fixture
    output = tmp_path / "continued"
    shutil.copytree(baseline, output)
    path = output / "predictions/00000-clean.npz"
    if change == "metadata":
        path.write_bytes(b"torn")
    elif change == "producer":
        identity = {"fixture": "different package"}
    elif change == "extra":
        (output / "predictions/extra").write_text("extra")
    elif change == "missing":
        path.unlink()
    with (
        pytest.raises((ValueError, OSError)),
        PredictionJournal(output, identity, resume=True) as journal,
        ViewDataset(source) as dataset,
    ):
        if change == "source":
            arrays = dataset[0]
            arrays["xyz_m"] = arrays["xyz_m"] + 0.01
            journal.prediction(
                0, dataset.records[0], arrays, "clean", continuation.forbidden
            )
        else:
            continuation.inventory(journal, dataset)


def test_replay_forbids_both_missing_prediction_and_report(tmp_path):
    with continuation.ReplayJournal(tmp_path / "run", {"fixture": True}) as journal:
        with pytest.raises(ValueError, match="cannot compute"):
            journal.prediction(
                0,
                {},
                {"x": np.arange(2)},
                "clean",
                lambda: pytest.fail("compute called"),
            )
        with pytest.raises(ValueError, match="cannot compute"):
            journal.physical_report(0, lambda: pytest.fail("fit called"))


def test_fit_metadata_matches_merged_extraction(fixture):
    from tennis3d.evaluation.delivery import physical_fit_inputs
    from tennis3d.evaluation.synthetic import input_mode

    source, release, _, _, _, baseline, _ = fixture
    with (
        ViewDataset(source) as dataset,
        np.load(baseline / "predictions/00000-corrupted.npz") as archive,
    ):
        arrays = dataset[0]
        args = (
            release,
            dataset,
            dataset.records[0],
            arrays,
            input_mode(arrays, "corrupted"),
            archive["xyz"],
            archive["covered"],
        )
        actual, metadata = continuation.fit_inputs(*args)
        expected, expected_metadata = physical_fit_inputs(*args)
        assert metadata == expected_metadata
        assert continuation.arrays_sha256(actual) == continuation.arrays_sha256(
            expected
        )


@pytest.fixture
def cli_fixture(fixture, tmp_path, monkeypatch):
    import tennis3d.lifting.data as data_module
    from tennis3d.evaluation import delivery
    from tennis3d.lifting.runtime import environment
    from tennis3d.provenance import code_identity

    tiny_source, release, provenance, expected, _, _, _ = fixture
    source = tmp_path / "test.source.json"
    write_json(
        source, read_json(ROOT / "configs/stage4-evaluation-main-v1/test.source.json")
    )
    provenance = {**provenance, "release_sha256": file_sha256(source)}
    identity = {
        "environment": environment("cpu"),
        "device": "cpu",
        "release_sha256": provenance["release_sha256"],
        "source_descriptor_sha256": file_sha256(source),
        "source_sha256": code_identity()["source_sha256"],
    }
    output, operations = tmp_path / "cli", tmp_path / "operations"
    with PredictionJournal(output, identity):
        pass
    original_dataset = data_module.ViewDataset
    monkeypatch.setattr(
        data_module,
        "ViewDataset",
        lambda path: original_dataset(tiny_source if Path(path) == source else path),
    )
    monkeypatch.setattr(
        delivery, "checked_release", lambda path: (release, {}, provenance)
    )
    monkeypatch.setattr(continuation, "COUNTS", expected)
    return source, output, operations


def test_run_accepts_actual_source_schema_and_warning_resume(cli_fixture, monkeypatch):
    import tennis3d.inference.predict as predictor

    source, output, operations = cli_fixture
    original, calls = predictor.predict_view, []

    def predict(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(1)
        if len(calls) == 1:
            os.kill(os.getpid(), signal.SIGUSR1)
        return result

    monkeypatch.setattr(predictor, "predict_view", predict)
    with pytest.raises(InterruptedError):
        continuation.run("inference", "fixture", source, output, operations, workers=2)
    assert read_json(operations / "0000.json")["status"] == "interrupted"
    assert len(list((output / "predictions").iterdir())) == 1
    continuation.run("inference", "fixture", source, output, operations, workers=2)
    assert len(calls) == 2
    assert (
        read_json(operations / "0001.json")["journal_counts"]["predictions_reused"] == 1
    )
    monkeypatch.setattr(
        predictor, "predict_view", lambda *a, **k: pytest.fail("CPU predicted")
    )
    continuation.run("analysis", "fixture", source, output, operations, workers=2)
    receipt = read_json(operations / "analysis-complete.json")
    assert receipt["status"] == "measured"
    assert receipt["invocations"]["0002.json"] == file_sha256(operations / "0002.json")
    for name, digest in receipt["files"].items():
        assert file_sha256(output / name) == digest
    with pytest.raises(ValueError, match="inference is complete"):
        continuation.run("inference", "fixture", source, output, operations, workers=2)
    assert not (operations / "0003.json").exists()
    continuation.run("analysis", "fixture", source, output, operations, workers=2)
    assert (
        read_json(operations / "0003.json")["journal_counts"]["physical_created"] == 0
    )


def test_restart_killed_after_invocation_publication_leaves_no_stale_receipt(
    cli_fixture, monkeypatch
):
    source, output, operations = cli_fixture
    continuation.run("inference", "fixture", source, output, operations, workers=2)
    continuation.run("analysis", "fixture", source, output, operations, workers=2)
    original = continuation.write_json

    def killed(path, value, **kwargs):
        original(path, value, **kwargs)
        if path == operations / "0002.json" and value["status"] == "running":
            assert not (operations / "analysis-complete.json").exists()
            raise KeyboardInterrupt("controlled hard interruption after publication")

    monkeypatch.setattr(continuation, "write_json", killed)
    with pytest.raises(KeyboardInterrupt):
        continuation.run("analysis", "fixture", source, output, operations, workers=2)
    assert read_json(operations / "0002.json")["status"] == "running"
    assert not (operations / "analysis-complete.json").exists()
    monkeypatch.setattr(continuation, "write_json", original)
    continuation.run("analysis", "fixture", source, output, operations, workers=2)
    assert (
        read_json(operations / "0003.json")["journal_counts"]["physical_created"] == 0
    )


@pytest.mark.parametrize("change", ["report", "cost", "extra", "missing"])
def test_completed_run_is_verified_before_withdrawing_result(cli_fixture, change):
    source, output, operations = cli_fixture
    continuation.run("inference", "fixture", source, output, operations, workers=2)
    continuation.run("analysis", "fixture", source, output, operations, workers=2)
    receipt_bytes = (operations / "analysis-complete.json").read_bytes()
    if change == "report":
        (output / "report.md").write_text("changed")
    elif change == "cost":
        record = read_json(operations / "0001.json")
        record["timing"]["worker_cpu_s"] = 9999
        write_json(operations / "0001.json", record, overwrite=True)
    elif change == "extra":
        (output / "physical/extra.json").write_text("extra")
    else:
        (output / "physical/00000.json").unlink()
    with pytest.raises((ValueError, OSError)):
        continuation.run("analysis", "fixture", source, output, operations, workers=2)
    assert (operations / "analysis-complete.json").read_bytes() == receipt_bytes
    assert not (operations / "0002.json").exists()
    assert (output / "metrics.json").exists()


def test_scoring_directory_sync_failure_cannot_publish_completion(
    cli_fixture, monkeypatch
):
    source, output, operations = cli_fixture
    continuation.run("inference", "fixture", source, output, operations, workers=2)
    original = continuation.sync_directory

    def sync(path):
        if path == output and (output / "metrics.json").exists():
            raise OSError("publication sync failed")
        original(path)

    monkeypatch.setattr(continuation, "sync_directory", sync)
    with pytest.raises(OSError, match="publication sync failed"):
        continuation.run("analysis", "fixture", source, output, operations, workers=2)
    assert not (operations / "analysis-complete.json").exists()
    assert read_json(operations / "0001.json")["status"] == "failed"
    monkeypatch.setattr(continuation, "sync_directory", original)
    continuation.run("analysis", "fixture", source, output, operations, workers=2)
    assert (
        read_json(operations / "0002.json")["journal_counts"]["physical_created"] == 0
    )


@pytest.mark.parametrize(
    "change",
    [
        "subset_sha256",
        "exclude_rally_ids",
        "revision",
        "repo_id",
        "manifest",
        "cache_bytes",
        "max_group_bytes",
    ],
)
def test_run_refuses_partial_wrong_or_unbounded_source(cli_fixture, change):
    source, output, operations = cli_fixture
    spec = read_json(source)
    spec[change] = {
        "subset_sha256": "a" * 64,
        "exclude_rally_ids": ["rally"],
        "revision": "wrong",
        "repo_id": "wrong",
        "manifest": {"sha256": "b" * 64},
        "cache_bytes": (1 << 30) + 1,
        "max_group_bytes": (1 << 28) + 1,
    }[change]
    write_json(source, spec, overwrite=True)
    with pytest.raises(ValueError, match="complete pinned test"):
        continuation.run("inference", "fixture", source, output, operations, workers=2)
    assert not operations.exists()
