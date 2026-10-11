"""Configured geometry, empirical boundaries and time-correlated observation noise."""

import numpy as np
import pytest
from pydantic import ValidationError

from tennis3d.synthetic.cameras import observe
from tennis3d.synthetic.config import SimulationConfig
from tennis3d.synthetic.corrupt import corrupt_observations, visibility_mask
from tennis3d.synthetic.observation_config import CameraConfig, CorruptionConfig


@pytest.mark.parametrize(
    "payload",
    [
        {"components": [{"weight": 0.5}]},
        {"components": [{"z_m": [1.0, 0.0]}]},
        {"components": [{"focal_over_width": [0.0, 1.0]}]},
        {"components": [{"y_m": [0.0, 2.0]}]},
        {"components": [{"zoom_fraction": 1.0}]},
        {"focal_error_fraction": float("nan")},
    ],
)
def test_invalid_camera_prior(payload):
    with pytest.raises(ValidationError):
        CameraConfig.model_validate(payload)


def test_camera_geometry_motion_and_intrinsic_error():
    cfg = SimulationConfig.model_validate(
        {
            "camera_noise_px": 0.0,
            "camera": {
                "components": [
                    {
                        "x_m": [3.0, 3.0],
                        "y_m": [-35.0, -35.0],
                        "z_m": [18.0, 18.0],
                        "focal_over_width": [1.2, 1.2],
                        "x_motion_m": 0.0,
                        "z_motion_m": 0.0,
                        "zoom_fraction": 0.0,
                    }
                ],
                "focal_error_fraction": 0.03,
            },
        }
    )
    arrays = {"t_s": np.arange(30) / 30, "xyz_m": np.tile([0.0, 0.0, 2.0], (30, 1))}
    first = observe(arrays, cfg, 9)
    second = observe(arrays, cfg, 9)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    centers = -np.einsum("nji,nj->ni", first["camera_R"], first["camera_t"])
    np.testing.assert_allclose(centers, np.tile([3.0, -35.0, 18.0], (30, 1)))
    np.testing.assert_allclose(first["camera_K"][:, 0, 0], 1.2 * cfg.image_width)
    ratio = first["estimated_camera_K"][:, 0, 0] / first["camera_K"][:, 0, 0]
    assert np.all((ratio >= 0.97) & (ratio <= 1.03))
    assert not np.all(ratio == 1)


def test_initial_missing_and_all_missing_keep_labels():
    cfg = SimulationConfig.model_validate(
        {
            "visible_run_lengths_s": [0.2],
            "missing_run_lengths_s": [0.3],
            "corruption": {"initial_missing_probability": 1.0},
        }
    )
    times = np.array([2.0, 2.1, 2.29, 2.31, 2.49, 2.51])
    np.testing.assert_array_equal(
        visibility_mask(times, cfg, np.random.default_rng(1)),
        [False, False, False, True, True, False],
    )
    cfg = SimulationConfig.model_validate(
        {
            "visible_run_lengths_s": [0.2],
            "missing_run_lengths_s": [10.0],
            "corruption": {"initial_missing_probability": 1.0},
        }
    )
    clean = np.tile([640.0, 360.0], (6, 1))
    result = corrupt_observations(
        clean, times, np.array([1280, 720]), cfg, np.random.default_rng(1)
    )
    assert not result["ball_observed"].any()
    assert result["ball_uv"].shape == clean.shape
    assert np.isnan(result["ball_uv"]).all()
    np.testing.assert_array_equal(clean, np.tile([640.0, 360.0], (6, 1)))


def test_localization_correlation_uses_elapsed_seconds():
    cfg = SimulationConfig.model_validate(
        {
            "noise_px": 2.0,
            "missing_probability": 0.0,
            "corruption": {
                "occlusion_duration_s": [0.0, 0.0],
                "false_candidate_probability": 0.0,
                "motion_bias_fraction": [0.0, 0.0],
                "localization_correlation_s": 0.1,
            },
        }
    )
    n = 20000
    times = np.arange(n) * 0.01
    clean = np.tile([640.0, 360.0], (n, 1))
    result = corrupt_observations(
        clean, times, np.array([1280, 720]), cfg, np.random.default_rng(18)
    )
    residual = result["ball_uv"][:, 0] - clean[:, 0]
    assert abs(np.std(residual) - 2) < 0.15
    assert abs(np.corrcoef(residual[:-1], residual[1:])[0, 1] - np.exp(-0.1)) < 0.02
    assert result["ball_observed"].all()


@pytest.mark.parametrize(
    "payload",
    [
        {"initial_missing_probability": 1.1},
        {"confidence": [0.2, 1.1]},
        {"occlusion_duration_s": [-1.0, 1.0]},
        {"motion_bias_fraction": [0.5, 0.1]},
        {"false_candidate_noise_px": float("inf")},
    ],
)
def test_invalid_corruption_prior(payload):
    with pytest.raises(ValidationError):
        CorruptionConfig.model_validate(payload)
