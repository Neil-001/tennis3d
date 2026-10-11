"""Shared pixel normalization and CPU float64 time quantization."""

from dataclasses import dataclass, fields

import numpy as np
import torch

WORLD_SCALE_M = 10.0
TIME_BIN_S = 0.002


@dataclass
class Features:
    """Batched tensors; masks are (B,T), UV ends in (2) or (14,2).

    t_s preserves raw float64 seconds. time_bins is relative to each window's
    first real timestamp. True padding excludes a row; other true masks include.
    """

    ball_uv: torch.Tensor
    court_uv: torch.Tensor
    ball_observed: torch.Tensor
    camera_valid: torch.Tensor
    padding: torch.Tensor
    t_s: torch.Tensor
    time_bins: torch.Tensor

    def to(self, device: str | torch.device) -> "Features":
        return Features(
            **{f.name: getattr(self, f.name).to(device) for f in fields(self)}
        )


def build_features(
    *,
    t_s,
    ball_uv,
    ball_observed,
    court_uv,
    camera_valid,
    image_size,
    pad_to: int | None = None,
) -> Features:
    """Build one (1,T,...) window from original pixels, without changing time.

    Missing coordinates are sanitized before learned projections. Invalid-camera
    rows can be represented here, but must be split out before model execution.
    """
    times = np.asarray(t_s, dtype=np.float64)
    n = len(times)
    if times.shape != (n,) or n == 0 or not np.isfinite(times).all():
        raise ValueError("timestamps must be a nonempty finite vector")
    if np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must increase strictly")
    observed = np.asarray(ball_observed)
    camera = np.asarray(camera_valid)
    for name, mask in (("ball_observed", observed), ("camera_valid", camera)):
        if mask.shape != (n,) or mask.dtype != np.bool_:
            raise ValueError(f"{name} must be a boolean (T,) mask")
    ball = np.asarray(ball_uv, dtype=np.float64)
    court = np.asarray(court_uv, dtype=np.float64)
    size = np.asarray(image_size, dtype=np.float64)
    if ball.shape != (n, 2) or court.shape != (n, 14, 2):
        raise ValueError("expected ball (T,2) and court (T,14,2)")
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("image_size must contain positive width and height")
    if not np.isfinite(ball[observed]).all() or not np.isfinite(court[camera]).all():
        raise ValueError("declared-valid coordinates must be finite")
    length = n if pad_to is None else pad_to
    if length < n:
        raise ValueError("pad_to cannot truncate real timestamps")
    arrays = {
        "ball_uv": np.zeros((length, 2), dtype=np.float32),
        "court_uv": np.zeros((length, 14, 2), dtype=np.float32),
        "ball_observed": np.zeros(length, dtype=bool),
        "camera_valid": np.zeros(length, dtype=bool),
        "padding": np.arange(length) >= n,
        "t_s": np.zeros(length, dtype=np.float64),
        "time_bins": np.zeros(length, dtype=np.int64),
    }
    arrays["ball_uv"][:n][observed] = ball[observed] / size
    arrays["court_uv"][:n][camera] = court[camera] / size
    arrays["ball_observed"][:n] = observed
    arrays["camera_valid"][:n] = camera
    arrays["t_s"][:n] = times
    bins = np.rint((times - times[0]) / TIME_BIN_S)
    if np.any(bins >= np.iinfo(np.int64).max):
        raise ValueError("timestamp span exceeds supported time bins")
    arrays["time_bins"][:n] = bins.astype(np.int64)
    if (
        not np.isfinite(arrays["ball_uv"]).all()
        or not np.isfinite(arrays["court_uv"]).all()
    ):
        raise ValueError("normalized coordinates exceed float32 range")
    return Features(
        **{name: torch.from_numpy(value[None]) for name, value in arrays.items()}
    )


def batch_features(windows: list[Features]) -> Features:
    """Concatenate equally padded windows along the batch axis."""
    if not windows:
        raise ValueError("cannot batch an empty window list")
    return Features(
        **{
            f.name: torch.cat([getattr(w, f.name) for w in windows])
            for f in fields(Features)
        }
    )
