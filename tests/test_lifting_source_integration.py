"""Source identity, reader lifetime, and unchanged evaluation exposure."""

from copy import deepcopy
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from tennis3d.evaluation import synthetic
from tennis3d.lifting.checkpoint import _same_dataset_identity
from tennis3d.lifting.data import ViewDataset
from tennis3d.lifting.settings import TrainConfig
from tennis3d.lifting.train import _validate_resume_data

FIXTURE = Path(__file__).parent / "fixtures/lifting"


def source(format_version=2):
    return SimpleNamespace(
        format_version=format_version,
        manifest_sha256="a" * 64,
        identity={
            "format_version": format_version,
            "manifest_sha256": "a" * 64,
            "revision": "b" * 40,
            "membership_sha256": "c" * 64,
            "split": "train",
            "order": "shard-group-rally-view-v1",
        },
    )


def checkpoint(config, datasets, *, legacy=False):
    state = {
        "config": config.model_dump(mode="json"),
        "manifest_sha256": {key: ds.manifest_sha256 for key, ds in datasets.items()},
    }
    if not legacy:
        state["dataset_identity"] = {
            key: deepcopy(ds.identity) for key, ds in datasets.items()
        }
    return state


def test_resume_accepts_relocated_sources():
    datasets = {"train": source(), "val": source()}
    original = TrainConfig(train_manifest="old/train.json", val_manifest="old/val.json")
    state = checkpoint(original, datasets)
    relocated = original.model_copy(
        update={"train_manifest": "new/train.json", "val_manifest": "new/val.json"}
    )
    _validate_resume_data(state, relocated, datasets)


@pytest.mark.parametrize(
    "key", ["revision", "manifest_sha256", "membership_sha256", "order", "split"]
)
def test_resume_rejects_changed_semantic_source(key):
    datasets = {"train": source(), "val": source()}
    config = TrainConfig()
    state = checkpoint(config, datasets)
    datasets["train"].identity[key] = "changed"
    with pytest.raises(ValueError, match="dataset identities differ"):
        _validate_resume_data(state, config, datasets)


def test_legacy_checkpoints_require_same_v1_manifest():
    config = TrainConfig()
    datasets = {"train": source(1), "val": source(1)}
    state = checkpoint(config, datasets, legacy=True)
    _validate_resume_data(state, config, datasets)
    datasets["train"].manifest_sha256 = "different"
    with pytest.raises(ValueError, match="legacy resume requires"):
        _validate_resume_data(state, config, datasets)
    datasets["train"] = source(2)
    with pytest.raises(ValueError, match="legacy resume requires"):
        _validate_resume_data(state, config, datasets)


def test_best_checkpoint_may_bridge_only_v1_identity_migration():
    old = {}
    assert _same_dataset_identity(
        old, {"dataset_identity": {"train": source(1).identity}}
    )
    assert not _same_dataset_identity(
        old, {"dataset_identity": {"train": source(2).identity}}
    )
    left = {"dataset_identity": {"train": source().identity}}
    right = deepcopy(left)
    right["dataset_identity"]["train"]["membership_sha256"] = "different"
    assert not _same_dataset_identity(left, right)


def test_trainer_closes_first_source_when_second_open_fails(tmp_path, monkeypatch):
    module = import_module("tennis3d.lifting.train")
    opened, closed = [], []

    class Dataset:
        def __init__(self, path):
            if opened:
                raise ValueError("second source failed")
            opened.append(path)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.extend(opened)

    monkeypatch.setattr(module, "ViewDataset", Dataset)
    with pytest.raises(ValueError, match="second source failed"):
        module.train(TrainConfig(device="cpu"), tmp_path / "run")
    assert closed == opened and len(closed) == 1


def test_retained_evaluator_preflight_once_preserves_all_supported_labels(monkeypatch):
    reads = []
    original = ViewDataset.__getitem__

    def read(dataset, index):
        reads.append(index)
        return original(dataset, index)

    def predict(model, arrays, **kwargs):
        return arrays["xyz_m"].copy(), arrays["camera_valid"].copy(), None

    monkeypatch.setattr(ViewDataset, "__getitem__", read)
    monkeypatch.setattr(synthetic, "predict_view", predict)
    with ViewDataset(FIXTURE / "val.manifest.json") as dataset:
        arrays = dataset[0]
        reads.clear()
        synthetic.preflight(dataset)
        result = synthetic.evaluate_model(None, dataset)
        again = synthetic.evaluate_model(None, dataset)
        assert len(reads) == 3 * len(dataset)  # One preflight plus two scoring passes.
        assert result == again
        assert result["source_identity"] == dataset.identity
        expected = arrays["camera_valid"] & arrays["label_valid"]
        for mode in ("clean", "corrupted"):
            metrics = result["modes"][mode]
            assert metrics["count"] == int(expected.sum())
            assert metrics["val_xyz_m"] == 0
        assert result["modes"]["corrupted"]["strata"]["occluded"]["count"] == int(
            (expected & ~arrays["ball_observed"]).sum()
        )


@pytest.mark.parametrize("fail", [False, True])
def test_evaluator_owns_and_closes_path_sources(monkeypatch, fail):
    closed = []
    close = ViewDataset.close

    def tracked_close(dataset):
        closed.append(dataset.path)
        close(dataset)

    def predict(model, arrays, **kwargs):
        if fail:
            raise RuntimeError("prediction failed")
        return arrays["xyz_m"].copy(), np.ones(len(arrays["t_s"]), dtype=bool), None

    monkeypatch.setattr(ViewDataset, "close", tracked_close)
    monkeypatch.setattr(synthetic, "predict_view", predict)
    path = FIXTURE / "val.manifest.json"
    if fail:
        with pytest.raises(RuntimeError, match="prediction failed"):
            synthetic.evaluate_model(None, path)
    else:
        synthetic.evaluate_model(None, path)
    assert closed == [path]
