"""Metre-space XYZ metrics with explicit frame and physical-rally denominators."""

from collections import defaultdict

import numpy as np

DEFINITIONS = {
    "rmse_m": "sqrt(mean(sum((prediction_m - truth_m)^2, axis=XYZ)))",
    "mean_m": "frame-weighted mean Euclidean XYZ error",
    "axis_rmse_m": "sqrt(mean(axis_error^2)), ordered X/Y/Z",
    "vertical_mae_m": "mean absolute Z error",
    "val_xyz_m": "mean over physical rally_id of mean Euclidean error pooled over its declared view/frame records",
    "exposure": "merge overlapping windows within each view first; retain all manifest views/excerpts without physical-time deduplication",
    "supported": "dataset label_valid AND camera_valid, fixed before model execution; no export quality filters",
    "below_ground_fraction": "fraction of supported predictions with ball-centre Z < 0 metres",
    "event": "within 0.05 seconds of a saved event; overlapping event strata are allowed",
    "distance": "Euclidean distance from ground-truth ball centre to saved true camera centre",
    "gap": "missing-run elapsed span between bracketing observations, or clip endpoint at an unbracketed edge; observed rows have zero",
    "empty_strata": "count=0 and null metrics, never a zero-error score",
}


def summarize(delta: np.ndarray) -> dict:
    """Summarize signed XYZ errors, in metres."""
    delta = np.asarray(delta, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(delta).all():
        raise ValueError("metrics require finite errors at every supported frame")
    if len(delta) == 0:
        return {
            "count": 0,
            "rmse_m": None,
            "mean_m": None,
            "median_m": None,
            "p95_m": None,
            "axis_rmse_m": None,
            "vertical_mae_m": None,
            "vertical_bias_m": None,
        }
    error = np.linalg.norm(delta, axis=-1)
    return {
        "count": len(delta),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "mean_m": float(error.mean()),
        "median_m": float(np.median(error)),
        "p95_m": float(np.percentile(error, 95)),
        "axis_rmse_m": np.sqrt(np.mean(delta**2, axis=0)).tolist(),
        "vertical_mae_m": float(np.abs(delta[:, 2]).mean()),
        "vertical_bias_m": float(delta[:, 2].mean()),
    }


def gap_durations(t_s, observed) -> np.ndarray:
    times, observed = np.asarray(t_s), np.asarray(observed)
    durations = np.zeros(len(times))
    edges = np.diff(np.r_[False, ~observed, False].astype(int))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        left, right = max(0, start - 1), min(len(times) - 1, stop)
        durations[start:stop] = times[right] - times[left]
    return durations


def strata(arrays: dict, observed: np.ndarray) -> dict[str, np.ndarray]:
    xyz = arrays["xyz_m"]
    height = xyz[:, 2]
    gap = gap_durations(arrays["t_s"], observed)
    masks = {
        "observed": observed,
        "occluded": ~observed,
        "height_under_1m": height < 1,
        "height_1_to_3m": (height >= 1) & (height < 3),
        "height_3m_plus": height >= 3,
        "gap_up_to_0.15s": (~observed) & (gap <= 0.15),
        "gap_0.15_to_0.5s": (~observed) & (gap > 0.15) & (gap <= 0.5),
        "gap_over_0.5s": (~observed) & (gap > 0.5),
    }
    if "camera_R" in arrays and "camera_t" in arrays:
        center = -np.einsum("tji,tj->ti", arrays["camera_R"], arrays["camera_t"])
        distance = np.linalg.norm(xyz - center, axis=-1)
        masks.update(
            distance_under_20m=distance < 20,
            distance_20_to_40m=(distance >= 20) & (distance < 40),
            distance_40m_plus=distance >= 40,
        )
    for name, code in (("bounce", 1), ("hit", 2), ("net_crossing", 3), ("out", 4)):
        mask = np.zeros(len(xyz), dtype=bool)
        for event in arrays.get("event_t_s", np.empty(0))[
            arrays.get("event_code", np.empty(0)) == code
        ]:
            mask |= np.abs(arrays["t_s"] - event) <= 0.05
        masks[f"near_{name}"] = mask
    if "segment_kind" in arrays:
        kind = arrays["segment_kind"][arrays["segment_id"]]
        for i, name in enumerate(
            ("toss", "serve", "return", "volley", "lob", "slice", "out")
        ):
            masks[f"shot_{name}"] = kind == i
    return masks


class XYZMetrics:
    """Retain numeric errors for exact quantiles; never retain input shards."""

    def __init__(self):
        self.errors = []
        self.stratum_errors = defaultdict(list)
        self.rallies = defaultdict(lambda: {"error_sum": 0.0, "frames": 0, "views": 0})
        self.source_frames = self.supported_frames = self.below_ground = 0
        self.views = []

    def add(self, record: dict, arrays: dict, prediction, covered, observed):
        n = len(arrays["t_s"])
        expected = np.asarray(arrays["label_valid"]) & np.asarray(
            arrays["camera_valid"]
        )
        prediction = np.asarray(prediction)
        if prediction.shape != (n, 3) or np.asarray(covered).shape != (n,):
            raise ValueError("prediction coverage shape does not match source frames")
        if not expected.any():
            raise ValueError("declared evaluation view has no supported labels")
        if (
            not np.asarray(covered)[expected].all()
            or not np.isfinite(prediction[expected]).all()
        ):
            raise ValueError(
                "incomplete or nonfinite predictions cannot reduce the evaluation denominator"
            )
        delta = prediction[expected] - arrays["xyz_m"][expected]
        summary = summarize(delta)
        self.errors.append(delta)
        self.source_frames += n
        self.supported_frames += len(delta)
        self.below_ground += int((prediction[expected, 2] < 0).sum())
        rally = self.rallies[record["rally_id"]]
        rally["error_sum"] += float(np.linalg.norm(delta, axis=-1).sum())
        rally["frames"] += len(delta)
        rally["views"] += 1
        self.views.append(
            {
                "rally_id": record["rally_id"],
                "view_id": record["view_id"],
                "physical_time_origin_s": record.get("physical_time_origin_s", 0.0),
                "source_frames": n,
                **summary,
            }
        )
        for name, mask in strata(arrays, np.asarray(observed)).items():
            self.stratum_errors[name].append(delta[mask[expected]])

    def finish(self) -> dict:
        if not self.errors:
            raise ValueError("evaluation manifest contains no scored views")
        physical = {
            key: {
                "mean_m": value["error_sum"] / value["frames"],
                "frames": value["frames"],
                "views": value["views"],
            }
            for key, value in sorted(self.rallies.items())
        }
        return {
            **summarize(np.concatenate(self.errors)),
            "val_xyz_m": float(np.mean([r["mean_m"] for r in physical.values()])),
            "physical_rally_count": len(physical),
            "view_count": len(self.views),
            "source_frames": self.source_frames,
            "supported_frames": self.supported_frames,
            "valid_coverage": 1.0,
            "supported_source_fraction": self.supported_frames / self.source_frames,
            "below_ground_fraction": self.below_ground / self.supported_frames,
            "strata": {
                name: summarize(np.concatenate(values))
                for name, values in sorted(self.stratum_errors.items())
            },
            "rallies": physical,
            "views": self.views,
            "definitions": DEFINITIONS,
        }
