"""Deterministic windows and source-index merging shared by evaluation/export."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Window:
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


def make_windows(
    camera_valid, *, max_frames: int = 250, stride: int = 125
) -> list[Window]:
    """Split at invalid cameras. Advance by stride until the interval is covered.

    The final window may be short. Its real rows use centre weights based on its
    real length; tensor padding never enters the merge. No timestamp rounding.
    """
    mask = np.asarray(camera_valid)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError("camera_valid must be a boolean vector")
    if not 1 <= stride <= max_frames <= 250:
        raise ValueError("require 1 <= stride <= max_frames <= 250")
    changes = np.diff(np.r_[False, mask, False].astype(int))
    windows = []
    for start, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        while True:
            end = min(start + max_frames, stop)
            windows.append(Window(int(start), int(end)))
            if end == stop:
                break
            start += stride
    return windows


def centre_weights(length: int) -> np.ndarray:
    if length < 1:
        raise ValueError("window length must be positive")
    i = np.arange(length)
    return np.minimum(i + 1, length - i).astype(np.float64)


class WindowMerger:
    """Online weighted mean plus RMS overlap disagreement in world metres.

    Uncovered rows return NaN and zero coverage. Consumers choose whether that
    is an export gap or an evaluation error; this helper never filters quality.
    """

    def __init__(self, frame_count: int):
        self.total = np.zeros((frame_count, 3), dtype=np.float64)
        self.square = np.zeros((frame_count, 3), dtype=np.float64)
        self.weight = np.zeros(frame_count, dtype=np.float64)
        self.count = np.zeros(frame_count, dtype=np.int64)

    def add(self, window: Window, xyz_m) -> None:
        xyz = np.asarray(xyz_m, dtype=np.float64)
        if not 0 <= window.start < window.stop <= len(self.weight):
            raise ValueError("window is outside source frames")
        if xyz.shape != (window.length, 3) or not np.isfinite(xyz).all():
            raise ValueError("each real window prediction must be finite (T,3)")
        index = slice(window.start, window.stop)
        weights = centre_weights(window.length)
        self.total[index] += xyz * weights[:, None]
        self.square[index] += xyz**2 * weights[:, None]
        self.weight[index] += weights
        self.count[index] += 1

    def finish(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        covered = self.weight > 0
        xyz = np.full_like(self.total, np.nan)
        disagreement = np.full(len(covered), np.nan)
        xyz[covered] = self.total[covered] / self.weight[covered, None]
        variance = self.square[covered] / self.weight[covered, None] - xyz[covered] ** 2
        disagreement[covered] = np.sqrt(np.maximum(variance, 0).sum(axis=-1))
        return xyz, covered, disagreement
