"""Projection and rays for undistorted pixels; no ground-plane lifting."""

import numpy as np

from tennis3d.geometry.camera import Camera, points


def project(camera: Camera, xyz) -> tuple[np.ndarray, np.ndarray]:
    """Return pixels (..., 2) and positive-depth mask (...,) for XYZ (..., 3).

    Invalid-depth pixels are NaN, not plausible image observations. Positive-depth
    points outside the image remain projected; this mask is not image visibility.
    """
    camera_xyz = camera.world_to_camera(xyz)
    valid = camera_xyz[..., 2] > 0
    homogeneous = camera_xyz @ camera.K.T
    uv = np.full(camera_xyz.shape[:-1] + (2,), np.nan)
    np.divide(
        homogeneous[..., :2], homogeneous[..., 2, None], out=uv, where=valid[..., None]
    )
    return uv, valid


def world_rays(camera: Camera, uv) -> tuple[np.ndarray, np.ndarray]:
    """Return centre (3,) and unit world directions (..., 3) for pixels (..., 2)."""
    uv = points(uv, 2)
    pixels = np.concatenate((uv, np.ones(uv.shape[:-1] + (1,))), axis=-1)
    directions = pixels @ np.linalg.inv(camera.K).T @ camera.R
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    return camera.center, directions
