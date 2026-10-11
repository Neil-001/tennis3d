"""Pixel-coordinate transforms use u' = sx*(u-left), v' = sy*(v-top)."""

from dataclasses import dataclass, replace

import numpy as np

from tennis3d.geometry.camera import Camera, ImageSize, finite_array, points


@dataclass(frozen=True, eq=False)
class ImageTransform:
    """Axis-aligned crop/resize. A maps source pixels to target pixels."""

    source: ImageSize
    target: ImageSize
    A: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "A", finite_array(self.A, (3, 3), "A"))
        if (
            not np.array_equal(self.A[2], [0, 0, 1])
            or self.A[0, 1] != 0
            or self.A[1, 0] != 0
            or self.A[0, 0] <= 0
            or self.A[1, 1] <= 0
        ):
            raise ValueError("A must be a positive axis-aligned crop/resize transform")

    def apply(self, uv) -> np.ndarray:
        """Map finite (..., 2) pixels into the target image."""
        return points(uv, 2) @ self.A[:2, :2].T + self.A[:2, 2]

    def inverse(self) -> "ImageTransform":
        return ImageTransform(self.target, self.source, np.linalg.inv(self.A))

    def camera(self, camera: Camera) -> Camera:
        if camera.image_size != self.source:
            raise ValueError("camera image size does not match transform source")
        return replace(camera, K=self.A @ camera.K, image_size=self.target)


def crop_resize(
    source: ImageSize,
    target: ImageSize,
    *,
    left: float = 0,
    top: float = 0,
    width: float | None = None,
    height: float | None = None,
) -> ImageTransform:
    """Crop inside source, then resize. Defaults resize the full source image.

    No implicit half-pixel offset is applied. Adapters must use the coordinate
    convention of their resampler and encode any required offset explicitly.
    """
    width = source.width if width is None else width
    height = source.height if height is None else height
    if (
        not np.isfinite([left, top, width, height]).all()
        or left < 0
        or top < 0
        or width <= 0
        or height <= 0
        or left + width > source.width
        or top + height > source.height
    ):
        raise ValueError("crop must have positive size and lie inside source image")
    sx, sy = target.width / width, target.height / height
    return ImageTransform(
        source, target, np.array([[sx, 0, -sx * left], [0, sy, -sy * top], [0, 0, 1]])
    )


def normalize_pixels(uv, image_size: ImageSize) -> np.ndarray:
    """(..., 2) original pixels to width/height-normalized model coordinates."""
    return points(uv, 2) / [image_size.width, image_size.height]
