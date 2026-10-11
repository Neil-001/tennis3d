"""Broadcast-like moving cameras and separate clean/corrupted numeric evidence."""

import numpy as np

from tennis3d.court.model import court_landmarks
from tennis3d.geometry.camera import Camera, ImageSize
from tennis3d.geometry.projection import project
from tennis3d.synthetic.config import SimulationConfig
from tennis3d.synthetic.corrupt import corrupt_observations
from tennis3d.synthetic.priors import choose


def look_at(
    center: np.ndarray, target: np.ndarray, focal: float, size: ImageSize
) -> Camera:
    forward = target - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.array([right, down, forward])
    intrinsic = np.array(
        [[focal, 0, size.width / 2], [0, focal, size.height / 2], [0, 0, 1]]
    )
    return Camera(intrinsic, rotation, -rotation @ center, size)


def observe(
    arrays: dict[str, np.ndarray], cfg: SimulationConfig, seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    times, xyz = arrays["t_s"], arrays["xyz_m"]
    n = len(times)
    size = ImageSize(cfg.image_width, cfg.image_height)
    components = cfg.camera.components
    # A single component takes no draw, preserving smoke camera samples.
    component = components[0] if len(components) == 1 else choose(rng, components)
    center = rng.uniform(*np.transpose([component.x_m, component.y_m, component.z_m]))
    focal = (
        rng.uniform(component.focal_over_width[0], component.focal_over_width[1])
        * size.width
    )
    phase = rng.uniform(0, 2 * np.pi)
    camera_bias = rng.normal(0, cfg.camera_noise_px, 2)
    focal_error = (
        rng.uniform(-cfg.camera.focal_error_fraction, cfg.camera.focal_error_fraction)
        if cfg.camera.focal_error_fraction
        else 0
    )
    landmarks = court_landmarks()
    cameras, estimated = [], []
    for time in times:
        motion = np.sin(component.angular_frequency_rad_s * time + phase)
        camera = look_at(
            center + [component.x_motion_m * motion, 0, component.z_motion_m * motion],
            np.array(
                [
                    component.target_x_motion_m * motion,
                    component.target_y_m,
                    component.target_z_m,
                ]
            ),
            focal * (1 + component.zoom_fraction * motion),
            size,
        )
        estimated_k = camera.K.copy()
        estimated_k[:2, :2] *= 1 + focal_error
        estimated_k[:2, 2] += camera_bias + rng.normal(
            0, cfg.camera_noise_px * cfg.camera.principal_point_jitter_fraction, 2
        )
        cameras.append(camera)
        estimated.append(Camera(estimated_k, camera.R, camera.t, size))
    clean = np.array([project(c, point)[0] for c, point in zip(cameras, xyz)])
    corruption = corrupt_observations(
        clean,
        times,
        np.array([size.width, size.height]),
        cfg,
        np.random.default_rng(np.random.SeedSequence([seed, 9])),
    )
    return {
        **arrays,
        "ball_clean_uv": clean,
        **corruption,
        "court_clean_uv": np.array([project(c, landmarks)[0] for c in cameras]),
        "court_uv": np.array([project(c, landmarks)[0] for c in estimated]),
        "camera_K": np.array([c.K for c in cameras]),
        "camera_R": np.array([c.R for c in cameras]),
        "camera_t": np.array([c.t for c in cameras]),
        "estimated_camera_K": np.array([c.K for c in estimated]),
        "camera_valid": np.ones(n, dtype=bool),
        "image_size": np.array([size.width, size.height], dtype=np.int64),
    }
