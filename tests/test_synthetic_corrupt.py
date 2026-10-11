import subprocess
import sys

import numpy as np

from tennis3d.synthetic.config import SimulationConfig
from tennis3d.synthetic.corrupt import corrupt_observations, visibility_mask


def test_empirical_run_lengths_use_seconds():
    cfg = SimulationConfig(visible_run_lengths_s=[0.2], missing_run_lengths_s=[0.1])
    times = np.array([0, 0.05, 0.19, 0.21, 0.29, 0.31, 0.49, 0.51])
    actual = visibility_mask(times, cfg, np.random.default_rng(0))
    np.testing.assert_array_equal(
        actual, [True, True, True, False, False, True, True, False]
    )


def test_training_redraw_and_rng_restore_preserve_clean_evidence():
    clean = np.tile([640.0, 360.0], (100, 1))
    times = np.arange(100) / 30
    rng = np.random.default_rng(30)
    state = rng.bit_generator.state
    args = (clean, times, np.array([1280, 720]), SimulationConfig())
    first = corrupt_observations(*args, rng)
    second = corrupt_observations(*args, rng)
    assert not np.array_equal(first["ball_observed"], second["ball_observed"])
    rng.bit_generator.state = state
    restored = corrupt_observations(*args, rng)
    np.testing.assert_array_equal(first["ball_uv"], restored["ball_uv"])
    np.testing.assert_array_equal(clean, np.tile([640.0, 360.0], (100, 1)))
    subprocess.run(
        [
            sys.executable,
            "-c",
            'import sys; from tennis3d.synthetic.corrupt import corrupt_observations; assert "mujoco" not in sys.modules',
        ],
        check=True,
    )
