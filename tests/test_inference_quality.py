import numpy as np
import pytest

from tennis3d.inference.export import build_result
from tennis3d.inference.quality import QualityConfig, quality_reasons
from tennis3d.schemas import Observation, Provenance, Source


def reasons(times, observations, camera=None, xyz=None, disagreement=None):
    n = len(times)
    camera = np.ones(n, dtype=bool) if camera is None else np.array(camera, dtype=bool)
    xyz = np.tile([0.0, 0.0, 1.0], (n, 1)) if xyz is None else xyz
    return quality_reasons(
        times,
        observations,
        camera,
        xyz,
        np.ones(n, dtype=bool),
        np.zeros(n) if disagreement is None else disagreement,
        QualityConfig(),
    )


def test_gap_rules_use_time_and_camera_intervals():
    # 0.5 s bracket passes; a slightly larger span rejects the whole gap.
    assert reasons([0, 0.2, 0.5], ["detected", "missing", "detected"]) == [[], [], []]
    assert reasons([0, 0.2, 0.501], ["detected", "missing", "detected"])[1] == [
        "long_gap"
    ]
    assert reasons([0, 0.15], ["missing", "detected"]) == [[], []]
    assert reasons([0, 0.151], ["detected", "missing"])[1] == ["long_gap"]
    result = reasons(
        [0, 0.04, 0.08, 0.12],
        ["detected", "missing", "ambiguous", "missing"],
        camera=[True, False, True, True],
    )
    assert result == [
        [],
        ["invalid_camera"],
        ["long_gap", "ambiguous_track"],
        ["long_gap"],
    ]


def test_export_keeps_source_times_duplicate_and_unclamped_out_ball():
    times = np.array([0, 0.04, 0.08, 0.12])
    xyz = np.array([[10, 18, 1], [10, 18, 1], [10, 18, -1], [10, 18, 1]], dtype=float)
    states: list[Observation] = ["detected", "duplicate", "detected", "detected"]
    failures = reasons(times, states, xyz=xyz)
    assert failures[2] == ["implausible_prediction"]
    result = build_result(
        source=Source(
            name="test.mp4",
            sha256="a" * 64,
            width=1920,
            height=1080,
            frame_count=4,
            time_origin_s=1.0,
            time_base=[1, 30000],
        ),
        provenance=Provenance(
            lifter_checkpoint_sha256="b" * 64,
            config_sha256="c" * 64,
            calibration_mode="estimated_per_frame",
            calibration_assumptions=["fixture"],
        ),
        t_s=times,
        observations=states,
        duplicate_of=np.array([-1, 0, -1, -1]),
        camera_valid=np.ones(4, dtype=bool),
        xyz_m=xyz,
        reasons=failures,
    )
    assert result.status == "partial"
    assert result.summary.valid_frame_fraction == 0.75
    assert (
        result.frames[1].duplicate_of == 0
        and result.frames[1].reconstruction == "inferred"
    )
    assert result.frames[0].xyz_m == [10, 18, 1]  # Out balls remain outside the court.
    assert result.frames[2].xyz_m is None and xyz[2, 2] == -1
    assert [frame.t_s for frame in result.frames] == times.tolist()


def test_disagreement_speed_and_nonfinite_predictions_rejected():
    xyz = np.array([[0, 0, 1], [10, 0, 1], [np.nan, 0, 1]])
    result = reasons([0, 0.01, 0.02], ["detected"] * 3, xyz=xyz)
    assert all("implausible_prediction" in row for row in result)
    assert reasons([0], ["detected"], disagreement=[3])[0] == ["window_disagreement"]
    with pytest.raises(ValueError, match="strictly increasing"):
        reasons([0, 0], ["detected"] * 2)
