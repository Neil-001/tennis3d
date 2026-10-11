"""Production quality rules applied after raw window prediction, never to training scores."""

from typing import get_args

import numpy as np
from pydantic import Field

from tennis3d.evaluation.runs import mask_runs, timeline
from tennis3d.schemas import Observation, ReasonCode, StrictModel


class QualityConfig(StrictModel):
    bracketed_gap_s: float = Field(default=0.5, ge=0)
    boundary_gap_s: float = Field(default=0.15, ge=0)
    max_window_disagreement_m: float = Field(default=2.0, gt=0)
    min_height_m: float = -0.05
    max_height_m: float = Field(default=30.0, gt=0)
    max_speed_m_s: float = Field(default=100.0, gt=0)


def quality_reasons(
    t_s,
    observations: list[Observation],
    camera_valid,
    xyz_m,
    covered,
    disagreement_m,
    config: QualityConfig,
    *,
    positive_depth=None,
) -> list[list[ReasonCode]]:
    """Return reasons at every exact source index without clamping predictions.

    Bracketed duration includes both bracketing observations. Boundary duration
    spans the missing run to its nearest observation. No support crosses invalid
    cameras. Evidence-free intervals remain unavailable even if raw XYZ exists.
    Speed checks flag both endpoints of implausible adjacent supported steps;
    they never compare across an unavailable interval or smooth a hit/bounce.
    Thresholds are configurable engineering defaults, not validated guarantees.
    """
    times = timeline(t_s)
    n = len(times)
    cameras = np.asarray(camera_valid)
    covered = np.asarray(covered)
    xyz = np.asarray(xyz_m, dtype=np.float64)
    disagreement = np.asarray(disagreement_m, dtype=np.float64)
    if len(observations) != n or set(observations) - set(get_args(Observation)):
        raise ValueError("one valid observation state is required per source frame")
    if any(
        value.shape != (n,) or value.dtype != np.bool_ for value in (cameras, covered)
    ):
        raise ValueError(
            "camera_valid and covered must be source-aligned boolean masks"
        )
    if xyz.shape != (n, 3) or disagreement.shape != (n,):
        raise ValueError(
            "raw prediction and disagreement shapes must match source frames"
        )
    if config.min_height_m >= config.max_height_m:
        raise ValueError("minimum height must be below maximum height")
    depth = (
        np.ones(n, dtype=bool) if positive_depth is None else np.asarray(positive_depth)
    )
    if depth.shape != (n,) or depth.dtype != np.bool_:
        raise ValueError("positive_depth must be a source-aligned boolean mask")
    detected = np.array([state == "detected" for state in observations], dtype=bool)
    supported = detected & cameras
    for begin, end, valid_camera in mask_runs(cameras):
        if not valid_camera:
            continue
        for start, stop, observed in mask_runs(detected[begin:end]):
            if observed:
                continue
            start += begin
            stop += begin
            left = start - 1 if start > begin else None
            right = stop if stop < end else None
            if left is not None and right is not None:
                duration = times[right] - times[left]
                limit = config.bracketed_gap_s
            elif left is not None:
                duration = times[stop - 1] - times[left]
                limit = config.boundary_gap_s
            elif right is not None:
                duration = times[right] - times[start]
                limit = config.boundary_gap_s
            else:
                continue
            # Seconds are float64 PTS; tolerate arithmetic roundoff at a threshold.
            if duration <= limit + 1e-12:
                supported[start:stop] = True
    reasons: list[list[ReasonCode]] = [[] for _ in range(n)]
    for i in range(n):
        if not cameras[i]:
            reasons[i].append("invalid_camera")
        elif not supported[i] or not covered[i]:
            reasons[i].append("long_gap")
            if observations[i] == "ambiguous":
                reasons[i].append("ambiguous_track")
        if covered[i] and cameras[i]:
            if (
                not np.isfinite(disagreement[i])
                or disagreement[i] < 0
                or disagreement[i] > config.max_window_disagreement_m
            ):
                reasons[i].append("window_disagreement")
            if (
                not np.isfinite(xyz[i]).all()
                or not depth[i]
                or not config.min_height_m <= xyz[i, 2] <= config.max_height_m
            ):
                reasons[i].append("implausible_prediction")
    eligible = np.array([not row for row in reasons], dtype=bool)
    pairs = np.flatnonzero(eligible[1:] & eligible[:-1])
    for index in pairs:
        i = int(index)
        speed = np.linalg.norm(xyz[i + 1] - xyz[i]) / (times[i + 1] - times[i])
        if speed > config.max_speed_m_s:
            for j in (i, i + 1):
                if "implausible_prediction" not in reasons[j]:
                    reasons[j].append("implausible_prediction")
    return reasons
