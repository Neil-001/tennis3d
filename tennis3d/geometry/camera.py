"""Validated world-to-camera pose and full pinhole intrinsics."""

from dataclasses import dataclass
from typing import Literal

import numpy as np


def finite_array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (self.width, self.height)):
            raise ValueError("image width and height must be positive integers")


@dataclass(frozen=True, eq=False)
class Camera:
    """X_camera = R @ X_world + t; metres, camera X right/Y down/Z forward.

    K (3, 3), R (3, 3), t (3,), image_size in pixels. Only undistorted
    pinhole images are supported here; distorted pixels must be undistorted first.
    """

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    image_size: ImageSize
    distortion_model: Literal["none"] = "none"

    def __post_init__(self):
        for name, shape in (("K", (3, 3)), ("R", (3, 3)), ("t", (3,))):
            object.__setattr__(
                self, name, finite_array(getattr(self, name), shape, name)
            )
        if not isinstance(self.image_size, ImageSize):
            raise TypeError("image_size must be ImageSize(width, height)")
        if self.distortion_model != "none":
            raise ValueError("only distortion_model='none' is supported")
        if (
            not np.allclose(self.K[2], [0, 0, 1], atol=1e-12, rtol=0)
            or self.K[1, 0] != 0
            or self.K[0, 0] <= 0
            or self.K[1, 1] <= 0
        ):
            raise ValueError(
                "K must be upper triangular with positive focal lengths and K[2]=[0,0,1]"
            )
        if not np.allclose(
            self.R.T @ self.R, np.eye(3), atol=1e-7, rtol=0
        ) or not np.isclose(np.linalg.det(self.R), 1, atol=1e-7, rtol=0):
            raise ValueError("R must be a proper orthonormal rotation")

    @property
    def center(self) -> np.ndarray:
        """World camera centre (3,) in metres."""
        return -self.R.T @ self.t

    def world_to_camera(self, xyz) -> np.ndarray:
        """Transform finite (..., 3) points to camera coordinates."""
        return points(xyz, 3) @ self.R.T + self.t

    def camera_to_world(self, xyz) -> np.ndarray:
        """Inverse pose for finite (..., 3) points."""
        return (points(xyz, 3) - self.t) @ self.R


def points(value, dimensions: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != dimensions or not np.isfinite(array).all():
        raise ValueError(f"points must be finite (..., {dimensions}) arrays")
    return array
