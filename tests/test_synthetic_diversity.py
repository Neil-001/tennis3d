"""Aggregate diversity diagnostics count physical rallies once and keep labels."""

import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np
import pytest

from tennis3d.artifacts import (
    ArtifactProvenance,
    file_sha256,
    json_sha256,
    read_json,
    write_json,
    write_npz,
)
from tennis3d.synthetic import audit
from tennis3d.synthetic.audit import (
    SplitAudit,
    _record_physical,
    _record_view,
    _split_result,
    audit_dataset,
)


def physical_arrays() -> dict[str, np.ndarray]:
    n = 300
    times = np.arange(n, dtype=np.float64) * 0.04
    segment_id = np.r_[np.zeros(150), np.ones(150)].astype(np.int64)
    xyz = np.column_stack(
        [np.zeros(n), np.linspace(-10, 10, n), 1 + np.sin(np.linspace(0, np.pi, n))]
    )
    velocity = np.tile([1.0, 20.0, 3.0], (n, 1))
    spin = np.tile([-100.0, 2.0, 3.0], (n, 1))
    before = np.array([[0.0, 0.0, 1.0, 1.0, 20.0, 3.0, -100.0, 2.0, 3.0]])
    after = before.copy()
    return {
        "t_s": times,
        "xyz_m": xyz,
        "velocity_m_s": velocity,
        "spin_rad_s": spin,
        "label_valid": np.ones(n, dtype=bool),
        "segment_id": segment_id,
        "segment_kind": np.array([0, 2], dtype=np.int64),
        "prior_id": np.array([0, 3], dtype=np.int64),
        "event_t_s": np.array([times[150]]),
        "event_precision_s": np.array([0.0]),
        "event_code": np.array([2], dtype=np.int64),
        "event_xyz_m": np.array([[0.0, 0.0, 1.0]]),
        "hit_before": before,
        "hit_after": after,
    }


def view_arrays() -> dict[str, np.ndarray]:
    arrays = physical_arrays()
    n = len(arrays["t_s"])
    observed = np.r_[np.zeros(250), np.ones(50)].astype(bool)
    ball_uv = np.full((n, 2), np.nan)
    ball_uv[observed] = [640.0, 360.0]
    rotation = np.tile(np.eye(3), (n, 1, 1))
    center = np.array([0.0, -20.0, 10.0])
    intrinsic = np.tile(
        np.array([[640.0, 0.0, 640.0], [0.0, 640.0, 360.0], [0.0, 0.0, 1.0]]),
        (n, 1, 1),
    )
    return {
        **arrays,
        "ball_uv": ball_uv,
        "ball_clean_uv": np.tile([640.0, 360.0], (n, 1)),
        "ball_candidate_uv": np.tile([640.0, 360.0], (n, 1)),
        "ball_in_frame": np.ones(n, dtype=bool),
        "ball_confidence": observed.astype(np.float64),
        "court_clean_uv": np.zeros((n, 14, 2)),
        "court_uv": np.zeros((n, 14, 2)),
        "camera_K": intrinsic,
        "camera_R": rotation,
        "camera_t": np.tile(-center, (n, 1)),
        "estimated_camera_K": intrinsic.copy(),
        "camera_valid": np.ones(n, dtype=bool),
        "image_size": np.array([1280, 720], dtype=np.int64),
        "ball_observed": observed,
    }


@pytest.fixture
def dataset(tmp_path):
    config_hash = json_sha256({})
    source_hash = "0" * 64
    pool_hash = "1" * 64
    records = []
    for rally_id, views, attempts in (("rally-a", 2, 1), ("rally-b", 1, 3)):
        rally_path = tmp_path / f"{rally_id}.npz"
        physical_provenance = ArtifactProvenance(
            source_sha256=source_hash,
            config_sha256=config_hash,
            code_revision="fixture",
            upstream_sha256={"shot_pool": pool_hash},
        )
        write_npz(rally_path, physical_arrays(), physical_provenance)
        rally_hash = file_sha256(rally_path)
        for view_id in range(views):
            view_path = tmp_path / f"{rally_id}-view-{view_id}.npz"
            view_provenance = ArtifactProvenance(
                source_sha256=source_hash,
                config_sha256=config_hash,
                code_revision="fixture",
                upstream_sha256={"rally": rally_hash, "shot_pool": pool_hash},
            )
            write_npz(view_path, view_arrays(), view_provenance)
            records.append(
                {
                    "rally_id": rally_id,
                    "family_id": "train-family",
                    "pool_sha256": pool_hash,
                    "view_id": view_id,
                    "split": "train",
                    "path": view_path.name,
                    "sha256": file_sha256(view_path),
                    "rally_path": rally_path.name,
                    "rally_sha256": rally_hash,
                    "truth_path": "unused.json",
                    "truth_sha256": "2" * 64,
                    "fps": 25,
                    "attempts": attempts,
                }
            )

    split_entries = {}
    for split in ("train", "val", "test"):
        split_path = tmp_path / f"{split}.manifest.json"
        write_json(
            split_path,
            {
                "schema_version": "1.0",
                "config_sha256": config_hash,
                "generator_sha256": source_hash,
                "split": split,
                "records": records if split == "train" else [],
            },
        )
        split_entries[split] = {
            "path": split_path.name,
            "sha256": file_sha256(split_path),
        }
    pool_attempts = {
        kind: 8 for kind in ("toss", "serve", "return", "volley", "lob", "slice", "out")
    }
    write_json(
        tmp_path / "manifest.json",
        {
            "generator_sha256": source_hash,
            "config_sha256": config_hash,
            "splits": split_entries,
            "pools": [
                {
                    "family_id": "train-family",
                    "split": "train",
                    "sha256": pool_hash,
                    "config": {"size": 4},
                    "attempts": pool_attempts,
                }
            ],
        },
    )
    write_json(
        tmp_path / "manifest.sha256.json",
        {"manifest.json": file_sha256(tmp_path / "manifest.json")},
    )
    return tmp_path


def test_audit_counts_physical_rallies_once_and_reports_exact_reuse(dataset):
    report = audit_dataset(dataset)
    train = report["splits"]["train"]
    assert train["physical_rallies"] == 2
    assert train["views"] == 3
    assert train["shot_counts"]["toss"] == 2
    assert train["shot_counts"]["return"] == 2
    assert train["adjacent_prior_pairs"] == {
        "total": 2,
        "distinct": 1,
        "reuses_after_first": 1,
        "tracking_limit": 100_000,
        "repeated_identity_count": 1,
        "detail_limit": 100,
        "details_truncated": False,
        "repeated": [
            {
                "family_id": "train-family",
                "from_kind": "toss",
                "from_prior_id": 0,
                "to_kind": "return",
                "to_prior_id": 3,
                "uses": 2,
            }
        ],
    }
    occupancy = next(row for row in train["prior_occupancy"] if row["kind"] == "return")
    assert occupancy == {
        "family_id": "train-family",
        "kind": "return",
        "pool_rows": 4,
        "used_rows": 1,
        "occupancy_fraction": 0.25,
        "segment_uses": 2,
        "prior_use_counts": {"3": 2},
    }
    assert train["reused_transitions"] == [
        {
            "family_id": "train-family",
            "kind": "return",
            "prior_id": 3,
            "uses": 2,
            "reuses_after_first": 1,
        },
        {
            "family_id": "train-family",
            "kind": "toss",
            "prior_id": 0,
            "uses": 2,
            "reuses_after_first": 1,
        },
    ]
    assert train["assembly_attempts"] == {
        "rally_counts": {"1": 1, "3": 1},
        "total_attempts": 4,
    }
    acceptance = next(
        row for row in report["pool_acceptance"] if row["kind"] == "return"
    )
    assert acceptance["accepted"] == 4
    assert acceptance["proposal_attempts"] == 8
    assert acceptance["acceptance_fraction"] == 0.5


def test_view_audit_keeps_labels_and_counts_all_missing_windows(dataset):
    train = audit_dataset(dataset)["splits"]["train"]
    assert train["physical_frames"] == 600
    assert train["valid_label_rows"] == 600
    assert train["view_frames"] == 900
    assert train["view_distributions"]["visibility_fraction_per_view"][
        "mean"
    ] == pytest.approx(1 / 6)
    assert train["view_distributions"]["in_frame_fraction_per_view"]["mean"] == 1
    assert train["view_distributions"]["focal_over_width_mean_per_view"]["mean"] == 0.5
    centers = train["view_distributions"]["camera_center_mean_m_per_view"]
    assert [centers[axis]["mean"] for axis in ("x", "y", "z")] == [0, -20, 10]
    assert train["all_missing_windows"]["250"] == {
        "total_windows": 153,
        "all_missing_windows": 3,
        "all_missing_fraction": pytest.approx(3 / 153),
    }
    runs = train["observation_runs"]
    assert runs["method"] == "tennis3d.evaluation.runs.run_report"
    assert runs["observation_counts"]["missing"] == 750
    assert runs["observation_counts"]["detected"] == 150
    assert runs["missing_duration_s"]["count"] == 3
    evaluation = train["evaluation_windows"]
    assert evaluation["modes"] == {
        "clean": {"total": 6, "all_missing": 0},
        "corrupted": {"total": 6, "all_missing": 3},
    }
    assert len(evaluation["failures"]) == 3
    assert all(
        row["start"] == 0 and row["stop"] == 250 for row in evaluation["failures"]
    )


@pytest.mark.parametrize(
    "length,evidence,expected",
    [
        (10, [], [(0, 10)]),
        (400, [240], [(250, 400)]),
        (500, [0, 99, 350, 499], []),
    ],
)
def test_exact_evaluation_windows_include_short_terminal_views(
    length, evidence, expected
):
    arrays = view_arrays()
    arrays = {
        key: np.resize(value, (length, *value.shape[1:]))
        if len(value) == 300
        else value
        for key, value in arrays.items()
    }
    arrays["t_s"] = np.arange(length) / 30
    arrays["ball_observed"] = np.zeros(length, dtype=bool)
    arrays["ball_observed"][evidence] = True
    # Exercise clean support independently, with no clean evidence in a short view.
    arrays["ball_in_frame"][:] = length != 10
    result = SplitAudit()
    _record_view(result, arrays, {"rally_id": "r", "view_id": 2})
    bad = [
        (row["start"], row["stop"])
        for row in result.evaluation_failures
        if row["mode"] == "corrupted"
    ]
    assert bad == expected
    assert result.evaluation_windows["clean"]["all_missing"] == (
        1 if length == 10 else 0
    )


def test_module_cli_writes_report(dataset, tmp_path):
    output = tmp_path / "audit.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tennis3d.synthetic.audit",
            str(dataset),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert read_json(output)["splits"]["train"]["physical_rallies"] == 2


def test_pair_tracking_limit_rejects_new_identity_without_silent_approximation(
    monkeypatch,
):
    monkeypatch.setattr(audit, "PAIR_TRACKING_LIMIT", 1)
    result = SplitAudit()
    record = {"family_id": "first", "fps": 25, "attempts": 1}
    uses = defaultdict(int)
    _record_physical(result, physical_arrays(), record, uses)
    _record_physical(result, physical_arrays(), record, uses)
    assert list(result.adjacent_prior_pairs.values()) == [2]
    with pytest.raises(ValueError, match="distinct adjacent pairs per split"):
        _record_physical(
            result, physical_arrays(), {**record, "family_id": "second"}, uses
        )
    assert list(result.adjacent_prior_pairs.values()) == [2]


def test_pair_details_are_bounded_but_aggregate_reuse_stays_exact(monkeypatch):
    monkeypatch.setattr(audit, "PAIR_DETAIL_LIMIT", 2)
    result = SplitAudit(
        adjacent_prior_pairs=Counter(
            {
                ("f", 0, index, 2, index): count
                for index, count in enumerate([2, 5, 5, 1])
            }
        )
    )
    pairs = _split_result(result, {}, {}, "train")["adjacent_prior_pairs"]
    assert (pairs["total"], pairs["distinct"], pairs["reuses_after_first"]) == (
        13,
        4,
        9,
    )
    assert pairs["repeated_identity_count"] == 3
    assert pairs["details_truncated"]
    assert len(pairs["repeated"]) == pairs["detail_limit"] == 2
    assert [row["from_prior_id"] for row in pairs["repeated"]] == [1, 2]


@pytest.mark.parametrize("valid_frames", [0, 1, 150])
def test_camera_statistics_ignore_invalid_rows_but_keep_support_counts(valid_frames):
    arrays = view_arrays()
    arrays["camera_valid"][valid_frames:] = False
    for name in ("camera_K", "camera_R", "camera_t"):
        arrays[name][valid_frames:] = np.nan
    result = SplitAudit()
    _record_view(result, arrays, {"rally_id": "r", "view_id": 0})
    assert result.view_frames == 300
    assert result.camera_valid_fraction_per_view.result()["mean"] == valid_frames / 300
    assert result.views_without_supported_labels == int(valid_frames == 0)
    assert result.camera_center_frames_m.axes[0].count == valid_frames
    assert result.focal_over_width_frames.count == valid_frames
    assert result.focal_over_width_mean_per_view.count == int(valid_frames > 0)
    assert result.camera_center_mean_per_view_m.axes[0].count == int(valid_frames > 0)
    if valid_frames:
        assert result.focal_over_width_frames.result()["mean"] == 0.5
        assert result.camera_center_frames_m.axes[1].result()["mean"] == -20


@pytest.mark.parametrize("aliased_path", [False, True])
def test_diversity_audit_rejects_cross_split_physical_reuse(dataset, aliased_path):
    train = read_json(dataset / "train.manifest.json")
    val = read_json(dataset / "val.manifest.json")
    val["records"] = [{**train["records"][0], "split": "val"}]
    if aliased_path:
        original = dataset / val["records"][0]["rally_path"]
        (dataset / "alias.npz").write_bytes(original.read_bytes())
        val["records"][0]["rally_path"] = "alias.npz"
    write_json(dataset / "val.manifest.json", val, overwrite=True)
    root = read_json(dataset / "manifest.json")
    root["splits"]["val"]["sha256"] = file_sha256(dataset / "val.manifest.json")
    write_json(dataset / "manifest.json", root, overwrite=True)
    write_json(
        dataset / "manifest.sha256.json",
        {"manifest.json": file_sha256(dataset / "manifest.json")},
        overwrite=True,
    )
    with pytest.raises(
        ValueError, match="physical rally artifact is reused across splits"
    ):
        audit_dataset(dataset)
