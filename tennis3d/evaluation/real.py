"""Pixel metrics on declared annotation rows, and unlabelled execution reports."""

from collections import Counter

import numpy as np

from tennis3d.evaluation.annotations import AnnotationManifest
from tennis3d.evaluation.runs import mask_runs, run_report, timeline
from tennis3d.schemas import Result


def error_summary(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("errors must be finite one-dimensional values")
    return {
        "count": len(values),
        "mean": float(values.mean()) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "p95": float(np.percentile(values, 95)) if len(values) else None,
        "rmse": float(np.sqrt(np.mean(values**2))) if len(values) else None,
    }


def pixel_metrics(
    labels: AnnotationManifest,
    ball_uv,
    ball_observed,
    *,
    source_sha256: str,
    image_size: tuple[int, int],
    tolerances_px: list[float],
    t_s=None,
) -> dict:
    """One selected prediction per labelled frame, in original-image pixels.

    A localized match is TP. An incorrect location is both FP and FN. A selected
    prediction on visibility 0 is FP. Missing visible/inferred labels are FN.
    Class 3 is included but reported separately because its target is inferred.
    Unannotated frames never become negatives. Precision/recall with an empty
    denominator are null. Localization errors include *all* selected labelled
    pairs, not only tolerance matches. No dense XYZ accuracy is implied.
    """
    n = labels.source_frame_count
    uv = np.asarray(ball_uv, dtype=np.float64)
    observed = np.asarray(ball_observed)
    if source_sha256 != labels.source_sha256 or image_size != (
        labels.width,
        labels.height,
    ):
        raise ValueError("annotation source hash or original image dimensions differ")
    if uv.shape != (n, 2) or observed.shape != (n,) or observed.dtype != np.bool_:
        raise ValueError("prediction arrays must match the declared source frame count")
    if not np.isfinite(uv[observed]).all():
        raise ValueError("accepted predictions must be finite")
    if not tolerances_px or any(not np.isfinite(v) or v <= 0 for v in tolerances_px):
        raise ValueError("declare positive finite pixel tolerances")
    if len(set(tolerances_px)) != len(tolerances_px):
        raise ValueError("pixel tolerances must be unique")
    if t_s is not None:
        times = timeline(t_s)
        if len(times) != n or labels.time_basis != "source_pts":
            raise ValueError(
                "source timing checks require matching source_pts annotations"
            )
        if any(row.t_s != float(times[row.frame_index]) for row in labels.annotations):
            raise ValueError(
                "annotation timestamps differ from exact source timestamps"
            )
    rows = labels.annotations
    indices = np.array([row.frame_index for row in rows], dtype=np.int64)
    visibility = np.array([row.visibility for row in rows], dtype=np.int64)
    selected = observed[indices]
    target = visibility > 0
    errors = np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        if selected[i] and row.uv is not None:
            errors[i] = np.linalg.norm(uv[row.frame_index] - row.uv)

    def metrics(mask):
        paired = mask & target & selected
        scores = []
        for tolerance in tolerances_px:
            tp = int(np.sum(paired & (errors <= tolerance)))
            fp = int(np.sum(mask & selected)) - tp
            fn = int(np.sum(mask & target)) - tp
            false_frames = np.zeros(n, dtype=bool)
            false_frames[indices] = mask & selected & ~(target & (errors <= tolerance))
            false_runs = [
                {"start_frame": start, "stop_frame": stop}
                for start, stop, value in mask_runs(false_frames)
                if value
            ]
            scores.append(
                {
                    "tolerance_px": tolerance,
                    "true_positive": tp,
                    "false_positive": fp,
                    "false_negative": fn,
                    "false_track_frame_fraction": fp / int(mask.sum())
                    if mask.any()
                    else None,
                    "false_track_runs": false_runs,
                    "precision": tp / (tp + fp) if tp + fp else None,
                    "recall": tp / (tp + fn) if tp + fn else None,
                }
            )
        return {
            "labelled_frames": int(mask.sum()),
            "localization_error_px": error_summary(errors[paired]),
            "scores": scores,
        }

    return {
        "source_sha256": source_sha256,
        "labels_sha256": labels.labels_sha256,
        "clip_id": labels.clip_id,
        "time_basis": labels.time_basis,
        "pixel_convention": labels.pixel_convention,
        "matching": "single selected location; wrong location counts FP and FN",
        "all": metrics(np.ones(len(rows), dtype=bool)),
        "by_visibility": {str(v): metrics(visibility == v) for v in range(4)},
    }


def execution_report(
    result: Result,
    *,
    calibration_residual_px=(),
    runtime_s: dict[str, float] | None = None,
    end_time_s=None,
) -> dict:
    """Coverage/failure/runtime diagnostics that need no manual ball labels."""
    runtime_s = runtime_s or {}
    if any(not np.isfinite(v) or v < 0 for v in runtime_s.values()):
        raise ValueError("runtime seconds must be finite and nonnegative")
    residuals = np.asarray(calibration_residual_px, dtype=np.float64)
    if np.any(residuals < 0):
        raise ValueError("calibration residuals cannot be negative")
    return {
        "status": result.status,
        "source": result.source.model_dump(),
        "coverage": result.summary.model_dump(),
        "failure_reasons": dict(
            Counter(reason for frame in result.frames for reason in frame.reason_codes)
        ),
        "input_failure_reasons": result.failure_reasons,
        "calibration_residual_px": error_summary(residuals),
        "runtime_s": runtime_s,
        "observation_runs": run_report(
            [frame.t_s for frame in result.frames],
            [frame.observation for frame in result.frames],
            end_time_s=end_time_s,
        ),
        "accuracy_claim": "none; unlabelled execution and consistency diagnostics",
    }
