"""Observed/missing runs measured on the original presentation-time axis."""

import itertools
from collections import Counter
from typing import get_args

import numpy as np

from tennis3d.schemas import Observation

ALL_MISSING_WINDOW_FRAMES = (20, 50, 125, 250)


def all_missing_windows(observed, width: int) -> tuple[int, int]:
    """Count complete sliding frame windows and those with no observation."""
    total = max(len(observed) - width + 1, 0)
    if not total:
        return 0, 0
    seen = np.r_[0, np.cumsum(np.asarray(observed, dtype=bool))]
    return total, int(np.sum(seen[width:] == seen[:-width]))


def timeline(t_s) -> np.ndarray:
    times = np.asarray(t_s, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("timestamps must be a finite one-dimensional array")
    if len(times) and (times[0] < 0 or np.any(np.diff(times) <= 0)):
        raise ValueError("timestamps must be nonnegative and strictly increasing")
    return times


def mask_runs(mask):
    """Yield (start, exclusive stop, value) without removing any source rows."""
    mask = np.asarray(mask)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError("run mask must be one-dimensional boolean")
    edges = np.r_[0, np.flatnonzero(mask[1:] != mask[:-1]) + 1, len(mask)]
    for start, stop in itertools.pairwise(edges):
        if stop > start:
            yield int(start), int(stop), bool(mask[start])


def run_report(t_s, observations: list[Observation], *, end_time_s=None) -> dict:
    """Measure sample-and-hold runs [first sample, next run's first sample).

    The last run ends at explicit end_time_s or the last decoded timestamp.
    The latter gives no invented final-frame duration. Zero-duration endpoint
    runs remain in diagnostics but are excluded from empirical duration samples.
    Duplicate and ambiguous frames count as missing new accepted observations.
    """
    times = timeline(t_s)
    allowed = get_args(Observation)
    if len(observations) != len(times) or set(observations) - set(allowed):
        raise ValueError("one valid observation state is required per timestamp")
    end = float(times[-1]) if len(times) else 0.0
    if end_time_s is not None:
        if not np.isfinite(end_time_s) or end_time_s < end:
            raise ValueError("end_time_s must be finite and at or after the last frame")
        end = float(end_time_s)
    counts = Counter(observations)
    observed = np.array([state == "detected" for state in observations], dtype=bool)
    runs = []
    samples = {"visible_run_lengths_s": [], "missing_run_lengths_s": []}
    for start, stop, visible in mask_runs(observed):
        right = float(times[stop]) if stop < len(times) else end
        duration = right - float(times[start])
        runs.append(
            {
                "start_frame": start,
                "stop_frame": stop,
                "visible": visible,
                "start_s": float(times[start]),
                "end_s": right,
                "duration_s": duration,
            }
        )
        if duration > 0:
            key = "visible_run_lengths_s" if visible else "missing_run_lengths_s"
            samples[key].append(duration)
    return {
        "endpoint_convention": "sample_hold_to_next_run; final_end_explicit"
        if end_time_s is not None
        else "sample_hold_to_next_run; final_end_last_timestamp",
        "frame_count": len(times),
        "duration_s": end - float(times[0]) if len(times) else 0.0,
        "observation_counts": {key: counts[key] for key in sorted(allowed)},
        "runs": runs,
        "empirical_corruption": samples,
        "empirical_ready": all(samples.values()),
        "limitations": "Development observation continuity only; no localization or false-positive accuracy without labels. A short clip is not a representative tennis noise distribution.",
    }
