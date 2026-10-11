"""Checksum audit rejects tampering beyond the model-input shard."""

import json
from pathlib import Path

import numpy as np
import pytest

from tennis3d.artifacts import (
    ArtifactProvenance,
    file_sha256,
    json_sha256,
    read_json,
    read_npz,
    write_json,
    write_npz,
)
from tennis3d.synthetic.data import audit_splits


@pytest.fixture
def dataset(tmp_path):
    arrays = {
        "t_s": np.array([0.0, 0.04]),
        "xyz_m": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
        "velocity_m_s": np.zeros((2, 3)),
        "spin_rad_s": np.zeros((2, 3)),
        "ball_uv": np.zeros((2, 2)),
        "ball_clean_uv": np.zeros((2, 2)),
        "court_uv": np.zeros((2, 14, 2)),
        "camera_K": np.tile(np.eye(3), (2, 1, 1)),
        "camera_R": np.tile(np.eye(3), (2, 1, 1)),
        "camera_t": np.zeros((2, 3)),
        "ball_observed": np.ones(2, dtype=bool),
        "label_valid": np.ones(2, dtype=bool),
        "camera_valid": np.ones(2, dtype=bool),
        "hit_before": np.empty((0, 9)),
        "hit_after": np.empty((0, 9)),
        "ball_candidate_uv": np.zeros((2, 2)),
        "ball_in_frame": np.ones(2, dtype=bool),
        "ball_confidence": np.ones(2),
        "court_clean_uv": np.zeros((2, 14, 2)),
        "estimated_camera_K": np.tile(np.eye(3), (2, 1, 1)),
        "image_size": np.array([1280, 720]),
        "segment_id": np.zeros(2, dtype=np.int64),
        "segment_kind": np.array([2]),
        "prior_id": np.array([0]),
        "event_t_s": np.array([0.02, 0.02]),
        "event_precision_s": np.array([0.001, 0.001]),
        "event_code": np.array([1, 4]),
        "event_xyz_m": np.array([[0.0, 1.0, 0.0335], [0.0, 1.0, 0.0335]]),
    }
    config_hash = json_sha256({})
    provenance = ArtifactProvenance(
        source_sha256="0" * 64,
        config_sha256=config_hash,
        code_revision="fixture",
        upstream_sha256={"shot_pool": "1" * 64},
    )
    write_npz(tmp_path / "rally.npz", {"xyz_m": arrays["xyz_m"]}, provenance)
    provenance.upstream_sha256["rally"] = file_sha256(tmp_path / "rally.npz")
    write_npz(tmp_path / "view.npz", arrays, provenance)
    truth = read_json("tests/fixtures/trajectory.json")
    truth["source"]["sha256"] = file_sha256(tmp_path / "view.npz")
    truth["provenance"]["config_sha256"] = config_hash
    write_json(tmp_path / "truth.json", truth)
    record = {
        "rally_id": "train-rally",
        "family_id": "train-family",
        "pool_sha256": "1" * 64,
        "view_id": 0,
        "split": "train",
        "path": "view.npz",
        "sha256": file_sha256(tmp_path / "view.npz"),
        "rally_path": "rally.npz",
        "rally_sha256": file_sha256(tmp_path / "rally.npz"),
        "truth_path": "truth.json",
        "truth_sha256": file_sha256(tmp_path / "truth.json"),
    }
    splits = {}
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}.manifest.json"
        write_json(
            path,
            {
                "schema_version": "1.0",
                "config_sha256": config_hash,
                "generator_sha256": "0" * 64,
                "split": split,
                "records": [record] if split == "train" else [],
            },
        )
        splits[split] = {"path": path.name, "sha256": file_sha256(path)}
    write_json(tmp_path / "config.resolved.json", {})
    write_json(
        tmp_path / "manifest.json",
        {
            "splits": splits,
            "config_sha256": config_hash,
            "generator_sha256": "0" * 64,
            "pools": [
                {
                    "family_id": "train-family",
                    "split": "train",
                    "sha256": "1" * 64,
                    "config": {"split": "train", "size": 24},
                }
            ],
        },
    )
    write_json(
        tmp_path / "manifest.sha256.json",
        {"manifest.json": file_sha256(tmp_path / "manifest.json")},
    )
    assert audit_splits(tmp_path)["train"]["frames"] == 2
    return tmp_path


@pytest.mark.parametrize(
    "name",
    ["manifest.json", "train.manifest.json", "view.npz", "rally.npz", "truth.json"],
)
def test_artifact_checksum_tampering(dataset, name):
    with (dataset / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        audit_splits(dataset)


def test_resolved_config_tampering(dataset):
    (dataset / "config.resolved.json").write_text(json.dumps({"changed": True}))
    with pytest.raises(ValueError, match="config checksum mismatch"):
        audit_splits(dataset)


@pytest.mark.parametrize(
    "change", ["orphan", "duplicate", "wrong_split", "wrong_pool", "small_pool"]
)
def test_pool_ownership_reconciled_with_views(dataset, change):
    root = read_json(dataset / "manifest.json")
    pool = root["pools"][0]
    if change == "orphan":
        root["pools"].append({**pool, "family_id": "unused", "sha256": "2" * 64})
    elif change == "duplicate":
        root["pools"].append(pool.copy())
    elif change == "wrong_split":
        pool["split"] = "test"
    elif change == "wrong_pool":
        pool["sha256"] = "3" * 64
    else:
        pool["config"]["size"] = 0
    write_json(dataset / "manifest.json", root, overwrite=True)
    refresh_manifest_hashes(dataset)
    with pytest.raises(ValueError, match="pool|ownership"):
        audit_splits(dataset)


def refresh_manifest_hashes(dataset):
    root = read_json(dataset / "manifest.json")
    for split, entry in root["splits"].items():
        entry["sha256"] = file_sha256(dataset / f"{split}.manifest.json")
    write_json(dataset / "manifest.json", root, overwrite=True)
    write_json(
        dataset / "manifest.sha256.json",
        {"manifest.json": file_sha256(dataset / "manifest.json")},
        overwrite=True,
    )


def test_consistent_manifest_relabel_conflicts_with_artifacts(dataset):
    changed = {"noise_px": 10}
    write_json(dataset / "config.resolved.json", changed, overwrite=True)
    for name in ("manifest", "train.manifest", "val.manifest", "test.manifest"):
        path = dataset / f"{name}.json"
        manifest = read_json(path)
        manifest["config_sha256"] = json_sha256(changed)
        write_json(path, manifest, overwrite=True)
    refresh_manifest_hashes(dataset)
    with pytest.raises(ValueError, match="provenance differs"):
        audit_splits(dataset)


@pytest.mark.parametrize(
    "artifact,identity",
    [
        (artifact, identity)
        for artifact in ("view", "rally", "truth")
        for identity in ("source", "config", "upstream")
        if (artifact, identity) != ("truth", "upstream")
    ],
)
def test_artifact_provenance_with_valid_checksums(dataset, artifact, identity):
    if artifact == "truth":
        path = dataset / "truth.json"
        payload = read_json(path)
        if identity == "source":
            payload["source"]["sha256"] = "9" * 64
        else:
            payload["provenance"]["config_sha256"] = "9" * 64
        write_json(path, payload, overwrite=True)
    else:
        path = dataset / f"{artifact}.npz"
        arrays, provenance = read_npz(path)
        if identity == "upstream":
            provenance.upstream_sha256["shot_pool"] = "9" * 64
        else:
            provenance = provenance.model_copy(update={f"{identity}_sha256": "9" * 64})
        write_npz(path, arrays, provenance, overwrite=True)
    manifest_path = dataset / "train.manifest.json"
    manifest = read_json(manifest_path)
    key = "sha256" if artifact == "view" else f"{artifact}_sha256"
    manifest["records"][0][key] = file_sha256(path)
    # Keep the truth-to-view link correct so the view's own provenance is tested.
    if artifact == "view":
        truth = read_json(dataset / "truth.json")
        truth["source"]["sha256"] = file_sha256(path)
        write_json(dataset / "truth.json", truth, overwrite=True)
        manifest["records"][0]["truth_sha256"] = file_sha256(dataset / "truth.json")
    write_json(manifest_path, manifest, overwrite=True)
    refresh_manifest_hashes(dataset)
    with pytest.raises(ValueError, match="provenance differs"):
        audit_splits(dataset)


def test_shared_code_changes_generator_identity(monkeypatch):
    pytest.importorskip("mujoco")
    pytest.importorskip("scipy")
    from tennis3d.synthetic.generate import generator_hash

    baseline = generator_hash()
    original = Path.read_bytes
    for suffix in (
        "court/model.py",
        "geometry/projection.py",
        "schemas.py",
        "artifacts.py",
    ):

        def changed(path, suffix=suffix):
            content = original(path)
            return (
                content + b"\n# changed dependency\n"
                if path.as_posix().endswith(suffix)
                else content
            )

        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_bytes", changed)
            assert generator_hash() != baseline


@pytest.mark.parametrize(
    "key,bad",
    [
        ("event_xyz_m", np.zeros((1, 3))),
        ("event_t_s", np.array([0.03, 0.02])),
        ("event_t_s", np.array([2, 1], dtype=np.uint64)),
        ("t_s", np.array([0, 2, 1], dtype=np.uint64)),
        ("event_code", np.array([1, 100])),
        ("segment_id", np.array([0, 1])),
        ("prior_id", np.array([-1])),
        ("hit_after", np.zeros((1, 9))),
        ("ball_clean_uv", np.array([[np.nan, np.nan], [0, 0]])),
        ("ball_clean_uv", np.array([[1280, 0], [0, 0]])),
        ("ball_candidate_uv", np.array([[np.nan, np.nan], [0, 0]])),
        ("ball_candidate_uv", np.array([[np.nan, 0], [0, 0]])),
        ("ball_candidate_uv", np.array([[np.inf, np.inf], [0, 0]])),
    ],
)
def test_malformed_event_and_segment_tables(dataset, key, bad):
    from tennis3d.synthetic.data import validate_view

    arrays, _ = read_npz(dataset / "view.npz")
    arrays[key] = bad
    with pytest.raises(ValueError):
        validate_view(arrays)


def test_masked_projection_and_rejected_candidate_remain_valid(dataset):
    from tennis3d.synthetic.data import validate_view

    arrays, _ = read_npz(dataset / "view.npz")
    arrays["ball_clean_uv"][0] = np.nan
    arrays["ball_in_frame"][0] = False
    arrays["ball_candidate_uv"][0] = np.nan
    arrays["ball_observed"][:] = False
    arrays["ball_uv"][:] = np.nan
    arrays["ball_confidence"][:] = 0
    # A finite rejected candidate at an in-frame point is also allowed.
    arrays["ball_candidate_uv"][1] = [-10, 800]
    validate_view(arrays)
