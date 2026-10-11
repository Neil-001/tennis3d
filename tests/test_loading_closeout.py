"""Measured reader boundaries and parity against the frozen corruption oracle."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tennis3d.synthetic.config import SimulationConfig
from tennis3d.synthetic.corrupt import corrupt_observations, visibility_mask
from tennis3d.synthetic.observation_config import CorruptionConfig

ORACLE_PATH = Path(__file__).parent / "fixtures/stage3_corrupt_oracle.py"
spec = importlib.util.spec_from_file_location("stage3_corrupt_oracle", ORACLE_PATH)
assert spec is not None and spec.loader is not None
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def empirical(weights=True, initially_missing=0.0):
    return SimulationConfig(
        visible_run_lengths_s=[0.125, 0.25, 0.5],
        missing_run_lengths_s=[0.0625, 0.125, 0.25],
        visible_run_weights=[0.1, 0.2, 0.7] if weights else None,
        missing_run_weights=[0.6, 0.3, 0.1] if weights else None,
        corruption=CorruptionConfig(initial_missing_probability=initially_missing),
    )


@pytest.mark.parametrize("weights", [False, True])
@pytest.mark.parametrize("initially_missing", [0.0, 0.35, 1.0])
@pytest.mark.parametrize("order", ["increasing", "duplicates", "unsorted"])
def test_empirical_boundaries_long_views_and_rng(weights, initially_missing, order):
    # Exact binary boundaries, irregular cadence, and more transitions than a
    # retained short view. Direct visibility callers also permit repeated times.
    times = np.unique(np.r_[np.arange(0, 64, 0.0625), np.arange(0, 64, 0.13)])
    if order == "duplicates":
        times = np.repeat(times, 2)
    elif order == "unsorted":
        times[1:-1] = times[1:-1][::-1]
    cfg = empirical(weights, initially_missing)
    for seed in range(16):
        expected_rng, actual_rng = (
            np.random.default_rng(seed),
            np.random.default_rng(seed),
        )
        expected = oracle.visibility_mask(times, cfg, expected_rng)
        actual = visibility_mask(times, cfg, actual_rng)
        np.testing.assert_array_equal(actual, expected)
        assert actual_rng.bit_generator.state == expected_rng.bit_generator.state


@pytest.mark.parametrize("weights", [False, True])
def test_whole_view_corruption_and_resumed_rng(weights):
    cfg = empirical(weights, 0.35)
    times = np.cumsum(np.resize([0.02, 0.031, 0.05], 1000))
    clean = np.tile([640.0, 360.0], (len(times), 1))
    clean[::17] = np.nan
    clean[::19] = [-1, 721]
    original = clean.copy()
    for seed in range(16):
        expected_rng, actual_rng = (
            np.random.default_rng(seed),
            np.random.default_rng(seed),
        )
        for _ in range(2):
            args = clean, times, np.array([1280, 720]), cfg
            expected = oracle.corrupt_observations(*args, expected_rng)
            actual = corrupt_observations(*args, actual_rng)
            assert actual.keys() == expected.keys()
            for key in actual:
                assert actual[key].dtype == expected[key].dtype
                assert actual[key].tobytes() == expected[key].tobytes()
            assert actual_rng.bit_generator.state == expected_rng.bit_generator.state
            restored = np.random.default_rng()
            restored.bit_generator.state = actual_rng.bit_generator.state
            actual_rng = restored
    np.testing.assert_array_equal(clean, original)


@pytest.mark.parametrize(
    "bad_weights", [[0.2], [0.2, 0.2, 0.2], [-0.1, 0.4, 0.7], [float("nan"), 0.3, 0.7]]
)
def test_mutated_invalid_weights_keep_choice_validation_and_rng(bad_weights):
    cfg = empirical().model_copy(update={"visible_run_weights": bad_weights})
    expected_rng, actual_rng = np.random.default_rng(42), np.random.default_rng(42)
    for function, rng in (
        (oracle.visibility_mask, expected_rng),
        (visibility_mask, actual_rng),
    ):
        with pytest.raises(ValueError):
            function(np.arange(10) / 30, cfg, rng)
    assert actual_rng.bit_generator.state == expected_rng.bit_generator.state


def test_distribution_cache_sees_mutated_config_lists():
    cfg = empirical()
    for weights in ([0.1, 0.2, 0.7], [0.7, 0.2, 0.1], [0.1, 0.2, 0.7]):
        cfg.visible_run_weights[:] = weights
        expected_rng, actual_rng = np.random.default_rng(8), np.random.default_rng(8)
        np.testing.assert_array_equal(
            visibility_mask(np.arange(1000) / 30, cfg, actual_rng),
            oracle.visibility_mask(np.arange(1000) / 30, cfg, expected_rng),
        )
        assert actual_rng.bit_generator.state == expected_rng.bit_generator.state


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.longdouble])
def test_direct_visibility_preserves_time_dtype_boundary_rounding(dtype):
    times = np.asarray(np.arange(0, 8.1, 0.1), dtype=dtype)
    for weights in (None, [1.0]):
        cfg = SimulationConfig(
            visible_run_lengths_s=[0.7],
            missing_run_lengths_s=[0.3],
            visible_run_weights=weights,
            missing_run_weights=weights,
        )
        for seed in range(16):
            expected_rng, actual_rng = (
                np.random.default_rng(seed),
                np.random.default_rng(seed),
            )
            expected = oracle.visibility_mask(times, cfg, expected_rng)
            actual = visibility_mask(times, cfg, actual_rng)
            np.testing.assert_array_equal(actual, expected)
            assert actual_rng.bit_generator.state == expected_rng.bit_generator.state
