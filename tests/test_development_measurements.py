"""Irregular timestamp and grouping fixtures, without inferred real labels."""

import numpy as np
import pytest

from tennis3d.evaluation.development import aggregate_reports, distribution
from tennis3d.evaluation.runs import run_report


def clip(group, times, states):
    report = run_report(times, states)
    missing = sum(row["duration_s"] for row in report["runs"] if not row["visible"])
    report.update(
        identity={"clip_id": group, "recording_group": group},
        missing_time_s=missing,
        all_missing=all(state != "detected" for state in states),
        initial_missing=states[0] != "detected",
        initial_missing_duration_s=report["runs"][0]["duration_s"]
        if states[0] != "detected"
        else 0,
    )
    return report


def test_recording_balance_and_leave_one_out():
    a = clip(
        "a", [0, 1, 2, 3, 4], ["missing", "detected", "missing", "detected", "missing"]
    )
    b = clip("b", [0, 8, 10], ["missing", "detected", "detected"])
    report = aggregate_reports([a, b])
    durations = report["durations"]["missing_run_lengths_s"]
    assert durations["pooled"]["mean"] == pytest.approx(10 / 3)
    assert durations["recording_balanced"]["mean"] == 4.5
    assert report["pooled_missing_time_fraction"] == pytest.approx(10 / 14)
    assert report["recording_balanced_missing_time_fraction"] == 0.65
    assert report["leave_one_recording_out"]["a"]["pooled_missing_time_fraction"] == 0.8
    weights = report["empirical_corruption"]["missing_run_weights"]
    np.testing.assert_allclose(weights, [0.25, 0.25, 0.5])


def test_all_missing_group_stays_in_denominator():
    a = clip(
        "a",
        [0, 0.04, 0.11, 0.2, 0.24],
        ["detected", "duplicate", "ambiguous", "detected", "missing"],
    )
    b = clip("b", [0, 1], ["missing", "missing"])
    report = aggregate_reports([a, b])
    assert report["frame_count"] == 7
    assert report["all_missing_clips"] == ["b"]
    assert report["unfit_clip_ids"] == ["b"]
    assert report["durations"]["visible_run_lengths_s"]["groups_without_samples"] == [
        "b"
    ]
    assert a["runs"][-1]["duration_s"] == 0
    assert report["pooled_missing_time_fraction"] == pytest.approx(1.16 / 1.24)


def test_weighted_quantiles_and_empty_groups():
    assert distribution([1, 9], [9, 1])["quantiles"]["0.5"] == 1
    assert distribution([])["quantiles"] is None
    with pytest.raises(ValueError):
        distribution([1, 2], [1, 0])
    result = aggregate_reports([clip("only", [0, 1], ["missing", "missing"])])
    assert result["leave_one_recording_out"]["only"]["frame_count"] == 0
    assert not result["empirical_ready"]


def test_camera_motion_uses_adjacent_valid_pts_only():
    from tennis3d.evaluation.development import _camera_report

    times = np.array([0, 0.1, 0.5, 0.6])
    K = np.tile(np.diag([100.0, 100.0, 1.0]), (4, 1, 1))
    R = np.tile(np.eye(3), (4, 1, 1))
    t = np.array(
        [[0.0, 0.0, 20.0], [-1.0, 0.0, 20.0], [np.nan] * 3, [-10.0, 0.0, 20.0]]
    )
    raw = np.zeros((4, 14, 2))
    cameras = {
        "K": K,
        "R": R,
        "t": t,
        "camera_valid": np.array([True, True, False, True]),
        "inliers": np.ones((4, 14), bool),
    }
    court = {"raw_uv": raw, "observed": np.ones((4, 14), bool)}
    report = _camera_report(cameras, court, times, np.array([100, 100]), "")
    assert report["valid_fraction"] == 0.75
    assert report["center_speed_m_s"]["count"] == 1
    assert report["center_speed_m_s"]["mean"] == 10
    assert report["focal_over_width"]["mean"] == 1
    assert report["rotation_speed_deg_s"]["mean"] == 0


def test_source_hash_envelope_and_no_camera(tmp_path):
    from tennis3d.artifacts import (
        ArtifactProvenance,
        file_sha256,
        json_sha256,
        write_json,
        write_npz,
    )
    from tennis3d.evaluation.development import summarize_clip

    config = {"fixture": True}
    provenance = ArtifactProvenance(
        source_sha256="a" * 64,
        config_sha256=json_sha256(config),
        code_revision="controlled",
    )
    write_json(tmp_path / "detect.config.json", config)
    arrays = {
        "frames.npz": {
            "t_s": [0.0, 0.1, 0.2],
            "duplicate_of": [-1, -1, -1],
            "cut": [False] * 3,
        },
        "court.npz": {},
        "ball_candidates.npz": {},
        "ball_track.npz": {"observation": [0, 1, 0]},
    }
    for name, values in arrays.items():
        write_npz(
            tmp_path / name,
            {key: np.asarray(value) for key, value in values.items()},
            provenance,
        )
    summary = {"artifacts": {name: file_sha256(tmp_path / name) for name in arrays}}
    write_json(tmp_path / "detect.json", summary)
    identity = {
        "clip_sha256": "a" * 64,
        "clip_id": "controlled",
        "recording_group": "fixture",
        "frame_count": 3,
    }
    report = summarize_clip(tmp_path, identity)
    assert report["calibration"]["status"] == "absent"
    assert report["runs"][0]["left_censored"]
    assert report["runs"][-1]["right_censored"]
    assert report["runs"][-1]["duration_s"] == 0
    (tmp_path / "ball_track.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        summarize_clip(tmp_path, identity)
