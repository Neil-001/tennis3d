import numpy as np
import pytest

from tennis3d.evaluation.metrics import XYZMetrics, gap_durations, summarize


def test_physical_rally_pooling_and_frame_weighted_metrics_differ():
    metric = XYZMetrics()
    for rally, view, n, error in (
        ("a", "a0", 2, 1.0),
        ("a", "a1", 6, 3.0),
        ("b", "b0", 2, 10.0),
    ):
        arrays = {
            "t_s": np.arange(n) * 0.04,
            "label_valid": np.ones(n, dtype=bool),
            "camera_valid": np.ones(n, dtype=bool),
            "xyz_m": np.zeros((n, 3)),
        }
        prediction = np.zeros((n, 3))
        prediction[:, 0] = error
        observed = np.zeros(n, dtype=bool)
        observed[0] = True
        metric.add(
            {"rally_id": rally, "view_id": view},
            arrays,
            prediction,
            np.ones(n, dtype=bool),
            observed,
        )
    result = metric.finish()
    assert result["val_xyz_m"] == 6.25  # mean((2*1+6*3)/8, 10)
    assert result["mean_m"] == 4.0
    assert result["rallies"]["a"]["views"] == 2
    assert result["count"] == 10
    assert result["strata"]["occluded"]["count"] == 7
    assert result["strata"]["near_bounce"]["mean_m"] is None


def test_metrics_refuse_incomplete_outputs_and_define_units():
    delta = np.array([[0, 3, 4], [0, 0, -5]])
    assert summarize(delta)["rmse_m"] == 5.0
    assert summarize(delta)["vertical_mae_m"] == 4.5
    assert summarize(delta)["vertical_bias_m"] == -0.5
    metric = XYZMetrics()
    arrays = {
        "t_s": np.array([0.0, 0.1]),
        "label_valid": np.ones(2, dtype=bool),
        "camera_valid": np.ones(2, dtype=bool),
        "xyz_m": np.zeros((2, 3)),
    }
    with pytest.raises(ValueError, match="denominator"):
        metric.add(
            {"rally_id": "r", "view_id": "v"},
            arrays,
            np.zeros((2, 3)),
            [True, False],
            [True, False],
        )
    np.testing.assert_allclose(
        gap_durations([0.0, 0.1, 0.2, 0.3, 0.4], [False, True, False, False, True]),
        [0.1, 0, 0.3, 0.3, 0],
    )
