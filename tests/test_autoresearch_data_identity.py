"""Frozen comparison identities cover membership/counts as well as local bytes."""

from copy import deepcopy
from pathlib import Path

import pytest

from tennis3d.artifacts import file_sha256, json_sha256
from tennis3d.autoresearch.inputs import data_identity, verify_data
from tennis3d.autoresearch.protocol import Protocol


def frozen_data(tmp_path):
    train, val = tmp_path / "train.json", tmp_path / "val.json"
    train.write_text("train source")
    val.write_text("val source")
    data = {
        "schema_version": 2,
        "files": {str(path): file_sha256(path) for path in (train, val)},
        "counts": {
            "train": {"physical_rallies": 2, "views": 4, "supported_frames": 10},
            "val": {"physical_rallies": 1, "views": 2, "supported_frames": 7},
        },
        "sources": {
            label: {
                "path": str(path),
                "identity": {
                    "format_version": 2,
                    "revision": "1" * 40,
                    "membership_sha256": "2" * 64,
                    "order": "shard-group-rally-view-v1",
                },
            }
            for label, path in (("train", train), ("val", val))
        },
    }
    return {**data, "sha256": json_sha256(data)}


@pytest.mark.parametrize("field", ["counts", "membership", "release", "files", "order"])
def test_tampered_exposure_or_source_invalidates_identity(tmp_path, field):
    data = frozen_data(tmp_path)
    verify_data(data)
    changed = deepcopy(data)
    if field == "counts":
        changed["counts"]["val"]["supported_frames"] -= 1
    elif field == "files":
        changed["files"].pop(next(iter(changed["files"])))
    else:
        key = {
            "membership": "membership_sha256",
            "release": "revision",
            "order": "order",
        }[field]
        changed["sources"]["train"]["identity"][key] = "different"
    with pytest.raises(ValueError, match="frozen data identity changed"):
        verify_data(changed)


def test_frozen_files_must_exist_and_match(tmp_path):
    data = frozen_data(tmp_path)
    path = Path(next(iter(data["files"])))
    path.write_text("changed bytes")
    with pytest.raises(ValueError, match="frozen data changed"):
        verify_data(data)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        verify_data(data)


def test_protocol_paths_must_be_frozen_sources(tmp_path):
    data = frozen_data(tmp_path)
    protocol = Protocol(
        train_manifest=tmp_path / "train.json", val_manifest=tmp_path / "val.json"
    )
    verify_data(data, protocol)
    with pytest.raises(ValueError, match="protocol source differs"):
        verify_data(
            data,
            protocol.model_copy(update={"train_manifest": tmp_path / "remote.json"}),
        )


def test_file_only_legacy_identity_requires_new_segment(tmp_path):
    data = frozen_data(tmp_path)
    legacy = {"files": data["files"], "counts": data["counts"]}
    legacy["sha256"] = json_sha256(legacy["files"])
    with pytest.raises(ValueError, match="initialize a new segment"):
        verify_data(legacy)


def test_v1_identity_counts_supported_arrays_and_closes_sources(monkeypatch):
    pytest.importorskip("torch")
    from tennis3d.lifting.data import ViewDataset

    fixture = Path(__file__).parent / "fixtures/lifting"
    protocol = Protocol(
        train_manifest=fixture / "train.manifest.json",
        val_manifest=fixture / "val.manifest.json",
    )
    closed = []
    close = ViewDataset.close

    def tracked_close(dataset):
        closed.append(dataset.path)
        close(dataset)

    monkeypatch.setattr(ViewDataset, "close", tracked_close)
    result = data_identity(protocol)
    assert set(closed) == {protocol.train_manifest, protocol.val_manifest}
    assert result["counts"]["val"] == {
        "physical_rallies": 1,
        "views": 1,
        "supported_frames": 32,
    }
    assert result["sources"]["train"]["identity"]["format_version"] == 1
    verify_data(result, protocol)
