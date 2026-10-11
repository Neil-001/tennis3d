"""Interrupted scoring reuses complete predictions while retaining all denominators."""

from pathlib import Path

import numpy as np
import pytest

from tennis3d.artifacts import read_json, write_json
from tennis3d.evaluation.prediction_journal import PredictionJournal
from tennis3d.evaluation.synthetic import evaluate_model

ROOT = Path(__file__).resolve().parents[1]


def test_interrupted_callback_replays_all_views_without_repeating_inference(
    tmp_path, monkeypatch
):
    import torch

    from tennis3d.lifting.model import Lifter, ModelConfig

    torch.set_num_threads(2)
    model = Lifter(ModelConfig(width=16, depth=2))
    source = ROOT / "tests/fixtures/lifting/val.manifest.json"
    from tennis3d.lifting.data import ViewDataset

    with ViewDataset(source) as dataset:
        expected_pairs = len(dataset) * 2
    output = tmp_path / "evaluation"
    identity = {"checkpoint": "fixture", "source": "fixture"}
    seen = []

    def interrupt(*args):
        seen.append(args[0]["view_id"])
        if len(seen) == 1:
            raise RuntimeError("controlled interruption during scoring")

    with PredictionJournal(output, identity) as journal:
        with pytest.raises(RuntimeError, match="controlled interruption"):
            evaluate_model(
                model, source, on_view=interrupt, prediction_provider=journal.prediction
            )
        assert journal.counts["predictions_created"] == 1
    with PredictionJournal(output, identity, resume=True) as journal:
        resumed = evaluate_model(model, source, prediction_provider=journal.prediction)
        assert journal.counts["predictions_reused"] == 1
        assert journal.counts["predictions_created"] == expected_pairs - 1
    fresh = evaluate_model(model, source)
    assert resumed == fresh
    with PredictionJournal(output, identity, resume=True) as journal:

        def forbidden(*args):
            raise AssertionError("restarted completed inference")

        import tennis3d.evaluation.synthetic as module

        with monkeypatch.context() as context:
            context.setattr(module, "predict_view", forbidden)
            assert (
                evaluate_model(model, source, prediction_provider=journal.prediction)
                == fresh
            )
        assert journal.counts["predictions_reused"] == expected_pairs


def test_changed_identity_input_damage_and_second_writer_are_rejected(tmp_path):
    root = tmp_path / "run"
    inputs = {"xyz": np.zeros((4, 3))}
    record = {"view_id": "fixture"}

    def compute():
        return np.ones((4, 3)), np.ones(4, dtype=bool), np.zeros(4), 0.3

    with PredictionJournal(root, {"source": "fixture"}) as first:
        first.prediction(0, record, inputs, "clean", compute)
        with (
            pytest.raises(BlockingIOError),
            PredictionJournal(root, first.identity, resume=True),
        ):
            pass
        with pytest.raises(ValueError, match="checksum or identity"):
            first.prediction(0, record, {"xyz": np.ones((4, 3))}, "clean", compute)
        path = root / "predictions/00000-clean.npz"
        with np.load(path, allow_pickle=False) as archive:
            saved = {k: archive[k] for k in archive.files}
        saved["xyz"][0, 0] += 1
        np.savez_compressed(path, **saved)
        with pytest.raises(ValueError, match="checksum or identity"):
            first.prediction(0, record, inputs, "clean", compute)
    with (
        pytest.raises(ValueError, match="resume identity"),
        PredictionJournal(root, {"source": "changed"}, resume=True),
    ):
        pass


def test_physical_report_atomic_reuse_and_checksum(tmp_path):
    root = tmp_path / "run"
    with PredictionJournal(root, {"source": "fixture"}) as journal:
        assert journal.physical_report(0, lambda: {"segments": [1]}) == {
            "segments": [1]
        }
    with PredictionJournal(root, {"source": "fixture"}, resume=True) as journal:

        def forbidden():
            raise AssertionError("repeated finished physical fitting")

        assert journal.physical_report(0, forbidden) == {"segments": [1]}
        path = root / "physical-journal/00000.json"
        saved = read_json(path)
        saved["report"]["segments"] = [2]
        write_json(path, saved, overwrite=True)
        with pytest.raises(ValueError, match="physical journal checksum"):
            journal.physical_report(0, forbidden)


def test_complete_delivery_fixture_has_full_reports_and_replays(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    import tennis3d.lifting.checkpoint as checkpoint_module
    from tennis3d.artifacts import file_sha256
    from tennis3d.evaluation.delivery import _synthetic_released
    from tennis3d.lifting.data import ViewDataset
    from tennis3d.lifting.model import Lifter, ModelConfig

    torch.set_num_threads(2)
    model = Lifter(ModelConfig(width=16, depth=2))
    monkeypatch.setattr(checkpoint_module, "load_lifter", lambda *args: (model, {}))
    source = ROOT / "tests/fixtures/lifting/val.manifest.json"
    with ViewDataset(source) as dataset:
        counts = {
            "rallies": len({r["rally_id"] for r in dataset.records}),
            "views": len(dataset),
            "frames": sum(r["frames"] for r in dataset.records),
        }
    release = SimpleNamespace(
        checkpoint=SimpleNamespace(path=source, sha256=file_sha256(source)), step=1
    )
    provenance = {"release": {"checkpoint": {"sha256": file_sha256(source)}}}
    output = tmp_path / "run"
    with PredictionJournal(output, {"fixture": "only"}) as journal:
        first = _synthetic_released(
            release, provenance, source, journal, "cpu", expected_counts=counts
        )
    assert len(list((output / "physical").glob("*.json"))) == counts["views"]
    assert len(list((output / "predictions").glob("*.npz"))) == counts["views"] * 2
    for mode in ("clean", "corrupted"):
        assert first["metrics"]["modes"][mode]["count"] == counts["frames"]
        assert (
            first["exported_result_metrics"]["modes"][mode]["source_frames"]
            == counts["frames"]
        )
    with PredictionJournal(output, {"fixture": "only"}, resume=True) as journal:
        second = _synthetic_released(
            release, provenance, source, journal, "cpu", expected_counts=counts
        )
        assert journal.counts["predictions_created"] == 0
        assert journal.counts["physical_created"] == 0
    assert first["metrics"] == second["metrics"]
    assert first["exported_result_metrics"] == second["exported_result_metrics"]
    assert first["physical"] == second["physical"]


def test_directory_links_are_synced_before_completed_artifacts(tmp_path, monkeypatch):
    import tennis3d.evaluation.prediction_journal as module

    calls = []
    original = module.sync_directory

    def sync(path):
        calls.append(path)
        original(path)

    monkeypatch.setattr(module, "sync_directory", sync)
    root = tmp_path / "nested" / "run"
    with PredictionJournal(root, {"fixture": "durability"}) as journal:
        assert tmp_path in calls and root.parent in calls
        journal.prediction(
            0,
            {},
            {"input": np.zeros(2)},
            "clean",
            lambda: (np.zeros((2, 3)), np.ones(2, dtype=bool), np.zeros(2), 0.1),
        )
        journal.physical_report(0, lambda: {"segments": []})
    # Top-level identity publication plus both new artifact-directory links are synced.
    assert calls.count(root) >= 3
    assert root / "predictions" in calls
    assert root / "physical-journal" in calls


def test_failed_atomic_publish_leaves_no_completed_prediction(tmp_path, monkeypatch):
    import tennis3d.evaluation.prediction_journal as module

    output = tmp_path / "run"
    with PredictionJournal(output, {"fixture": "atomic failure"}) as journal:

        def failed_write(stream, **arrays):
            stream.write(b"controlled incomplete bytes")
            raise OSError("controlled interrupted write")

        with monkeypatch.context() as context:
            context.setattr(module.np, "savez_compressed", failed_write)
            with pytest.raises(OSError, match="interrupted write"):
                journal.prediction(
                    0,
                    {},
                    {"input": np.zeros(2)},
                    "clean",
                    lambda: (
                        np.zeros((2, 3)),
                        np.ones(2, dtype=bool),
                        np.zeros(2),
                        0.1,
                    ),
                )
        assert list((output / "predictions").iterdir()) == []
        journal.prediction(
            0,
            {},
            {"input": np.zeros(2)},
            "clean",
            lambda: (np.zeros((2, 3)), np.ones(2, dtype=bool), np.zeros(2), 0.1),
        )
        assert journal.counts["predictions_created"] == 1


def test_warning_stops_at_next_prediction_boundary(tmp_path):
    with PredictionJournal(tmp_path / "run", {"fixture": "warning"}) as journal:
        journal.request_stop(None, None)
        with pytest.raises(InterruptedError, match="resume saved predictions"):
            journal.prediction(0, {}, {}, "clean", lambda: None)


@pytest.mark.parametrize("artifact", ["prediction", "physical"])
def test_resume_repairs_directory_link_after_interrupted_parent_sync(
    tmp_path, monkeypatch, artifact
):
    import tennis3d.evaluation.prediction_journal as module

    root = tmp_path / "run"
    dirname = "predictions" if artifact == "prediction" else "physical-journal"
    original = module.sync_directory
    failed = False

    def interrupt(path):
        nonlocal failed
        if not failed and path == root and (root / dirname).is_dir():
            failed = True
            raise OSError("controlled interrupted parent directory sync")
        original(path)

    def save(journal):
        if artifact == "prediction":
            journal.prediction(
                0,
                {},
                {"input": np.zeros(2)},
                "clean",
                lambda: (np.zeros((2, 3)), np.ones(2, dtype=bool), np.zeros(2), 0.1),
            )
        else:
            journal.physical_report(0, lambda: {"segments": []})

    with monkeypatch.context() as context:
        context.setattr(module, "sync_directory", interrupt)
        with (
            PredictionJournal(root, {"fixture": "interrupted mkdir"}) as journal,
            pytest.raises(OSError, match="interrupted parent"),
        ):
            save(journal)
    assert (root / dirname).is_dir()
    assert list((root / dirname).iterdir()) == []
    calls = []

    def sync(path):
        calls.append(path)
        original(path)

    monkeypatch.setattr(module, "sync_directory", sync)
    with PredictionJournal(
        root, {"fixture": "interrupted mkdir"}, resume=True
    ) as journal:
        save(journal)
    assert root.parent in calls
    assert root in calls
    assert root / dirname in calls


@pytest.mark.parametrize("artifact", ["prediction", "physical"])
def test_resume_syncs_file_link_after_interrupted_post_publication_sync(
    tmp_path, monkeypatch, artifact
):
    import tennis3d.evaluation.prediction_journal as module

    root = tmp_path / "run"
    dirname = "predictions" if artifact == "prediction" else "physical-journal"
    original = module.sync_directory
    failed = False

    def interrupt(path):
        nonlocal failed
        if not failed and path == root / dirname and any(path.iterdir()):
            failed = True
            raise OSError("controlled interrupted file link sync")
        original(path)

    def save(journal, compute=None):
        if artifact == "prediction":
            return journal.prediction(
                0,
                {},
                {"input": np.zeros(2)},
                "clean",
                compute
                or (
                    lambda: (np.zeros((2, 3)), np.ones(2, dtype=bool), np.zeros(2), 0.1)
                ),
            )
        return journal.physical_report(0, compute or (lambda: {"segments": []}))

    with monkeypatch.context() as context:
        context.setattr(module, "sync_directory", interrupt)
        with (
            PredictionJournal(root, {"fixture": "interrupted file link"}) as journal,
            pytest.raises(OSError, match="interrupted file link"),
        ):
            save(journal)
    assert len(list((root / dirname).iterdir())) == 1
    calls = []

    def sync(path):
        calls.append(path)
        original(path)

    def forbidden():
        raise AssertionError("recomputed completed artifact")

    monkeypatch.setattr(module, "sync_directory", sync)
    with PredictionJournal(
        root, {"fixture": "interrupted file link"}, resume=True
    ) as journal:
        save(journal, forbidden)
        field = "predictions_reused" if artifact == "prediction" else "physical_reused"
        assert journal.counts[field] == 1
    assert root.parent in calls and root in calls and root / dirname in calls
