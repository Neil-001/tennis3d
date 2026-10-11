"""NumPy-only ball observation draws; callers own the dedicated RNG and its state."""

import numpy as np

from tennis3d.synthetic.config import SimulationConfig


def visibility_mask(
    t_s: np.ndarray, cfg: SimulationConfig, rng: np.random.Generator
) -> np.ndarray:
    """Empirical visible/missing runs alternate in seconds with a configured start.

    Without empirical samples, use independent missing draws and one contiguous
    smoke occlusion. The measurements themselves are supplied later in config.
    """
    prior = cfg.corruption
    if cfg.visible_run_lengths_s is None:
        observed = rng.random(len(t_s)) >= cfg.missing_probability
        start = t_s[0] + rng.uniform(
            prior.occlusion_start_fraction[0], prior.occlusion_start_fraction[1]
        ) * (t_s[-1] - t_s[0])
        stop = start + rng.uniform(
            prior.occlusion_duration_s[0], prior.occlusion_duration_s[1]
        )
        observed[(t_s >= start) & (t_s < stop)] = False
        return observed
    assert cfg.missing_run_lengths_s is not None  # Paired by config validation.
    observed = np.zeros(len(t_s), dtype=bool)
    visible = (
        rng.random() >= prior.initial_missing_probability
        if prior.initial_missing_probability
        else True
    )
    start = float(t_s[0])
    # Advance by sampled physical durations. No frame-count assumptions.
    for _ in range(100000):
        if start > t_s[-1]:
            return observed
        samples = cfg.visible_run_lengths_s if visible else cfg.missing_run_lengths_s
        weights = cfg.visible_run_weights if visible else cfg.missing_run_weights
        stop = start + float(rng.choice(samples, p=weights))
        observed[(t_s >= start) & (t_s < stop)] = visible
        start, visible = stop, not visible
    raise ValueError(
        "empirical runs exceed 100000 transitions per view; use longer run samples"
    )


def corrupt_observations(
    clean_uv: np.ndarray,
    t_s: np.ndarray,
    image_size: np.ndarray,
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Draw accepted UV/masks/candidates from `(T,2)` clean pixels and `(T,)` times.

    `image_size` is width,height. Inputs are never modified. This function does
    not import MuJoCo, project cameras, alter XYZ, or own global random state.
    Training can redraw training evidence and retain fixed saved validation/test data.
    """
    t_s = np.asarray(t_s, dtype=np.float64)
    clean = np.asarray(clean_uv, dtype=np.float64)
    size = np.asarray(image_size)
    n = len(t_s)
    if n == 0 or clean.shape != (n, 2) or size.shape != (2,) or not np.all(size > 0):
        raise ValueError(
            "expected nonempty times, (T,2) clean UV and positive width,height"
        )
    if not np.isfinite(t_s).all() or not np.all(np.diff(t_s) > 0):
        raise ValueError("corruption times must be finite and strictly increasing")
    in_frame = np.asarray(
        np.isfinite(clean).all(axis=1)
        & (clean >= 0).all(axis=1)
        & (clean < size).all(axis=1),
        dtype=bool,
    )
    observed = in_frame & visibility_mask(t_s - t_s[0], cfg, rng)
    prior = cfg.corruption
    noise = rng.normal(0, cfg.noise_px, (n, 2))
    if prior.localization_correlation_s:
        rho = np.exp(-np.diff(t_s) / prior.localization_correlation_s)
        for i, correlation in enumerate(rho, 1):
            noise[i] = (
                correlation * noise[i - 1] + np.sqrt(1 - correlation**2) * noise[i]
            )
    noisy = clean + noise
    motion_uv = np.diff(clean, axis=0, prepend=clean[:1])
    # Missing clean projections do not poison the next positive-depth observation.
    motion_uv[~np.isfinite(motion_uv)] = 0
    noisy += np.clip(
        motion_uv, -prior.motion_clip_px, prior.motion_clip_px
    ) * rng.uniform(prior.motion_bias_fraction[0], prior.motion_bias_fraction[1])
    candidate = noisy.copy()
    false = rng.random(n) < prior.false_candidate_probability
    candidate[false] += rng.normal(0, prior.false_candidate_noise_px, (false.sum(), 2))
    observed[false] = False
    noisy[~observed] = np.nan
    return {
        "ball_uv": noisy,
        "ball_candidate_uv": candidate,
        "ball_observed": observed,
        "ball_in_frame": in_frame,
        "ball_confidence": np.where(
            observed, rng.uniform(prior.confidence[0], prior.confidence[1], n), 0
        ),
    }
