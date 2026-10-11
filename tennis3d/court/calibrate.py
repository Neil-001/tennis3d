"""TT3D's constrained focal/PnP method, independently implemented with SciPy/OpenCV."""

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from tennis3d.court.model import court_landmarks
from tennis3d.geometry.camera import Camera, ImageSize
from tennis3d.geometry.projection import project, world_rays
from tennis3d.perception.config import CalibrationConfig


@dataclass
class Estimate:
    camera: Camera | None = None
    valid: bool = False
    reason: str = "insufficient_landmarks"
    inliers: np.ndarray = field(default_factory=lambda: np.zeros(14, dtype=bool))
    residual_px: np.ndarray = field(default_factory=lambda: np.full(14, np.nan))
    homography_uv: np.ndarray = field(default_factory=lambda: np.full((14, 2), np.nan))
    alternatives: list = field(default_factory=list)


def intrinsic(focal, size):
    return np.array(
        [[focal, 0, size.width / 2], [0, focal, size.height / 2], [0, 0, 1.0]]
    )


def plausible(camera):
    xyz = court_landmarks()
    uv, depth = project(camera, xyz)
    center = camera.center
    # Canonical baseline broadcast view. These deliberately broad bounds exclude
    # mirrored/below-ground and overhead/non-baseline solutions.
    return bool(
        depth.all()
        and 1 < center[2] < 100
        and -250 < center[1] < -12
        and abs(center[0]) < 100
        and uv[0, 0] < uv[1, 0]
        and uv[0, 1] < uv[2, 1]
    )


def supported(mask, config):
    xyz = court_landmarks()[mask]
    return (
        len(xyz) >= config.min_landmarks
        and np.ptp(xyz[:, 0]) >= 8
        and np.ptp(xyz[:, 1]) >= 18
        and (xyz[:, 1] < 0).any()
        and (xyz[:, 1] > 0).any()
    )


def quality(camera, uv, mask, config):
    projected, depth = project(camera, court_landmarks())
    residual = np.linalg.norm(projected - uv, axis=1)
    diagonal = np.hypot(camera.image_size.width, camera.image_size.height)
    good = (
        supported(mask, config)
        and depth.all()
        and plausible(camera)
        and np.median(residual[mask]) <= config.median_diagonal * diagonal
        and np.quantile(residual[mask], 0.9) <= config.p90_diagonal * diagonal
    )
    return bool(good), residual


def estimate_camera(
    uv, observed, size: ImageSize, config: CalibrationConfig, prior=None
):
    result = Estimate()
    xyz = court_landmarks()
    mask = np.asarray(observed, dtype=bool) & np.isfinite(uv).all(axis=1)
    if not supported(mask, config):
        return result
    diagonal = np.hypot(size.width, size.height)
    h, inliers = cv2.findHomography(
        xyz[mask, :2], uv[mask], cv2.RANSAC, config.ransac_diagonal * diagonal
    )
    if h is None or inliers is None:
        result.reason = "homography_failed"
        return result
    indices = np.flatnonzero(mask)
    mask[indices] = inliers[:, 0].astype(bool)
    result.inliers = mask
    if not supported(mask, config):
        result.reason = "insufficient_spatial_support"
        return result
    result.homography_uv = cv2.perspectiveTransform(xyz[None, :, :2], h)[0]
    objects, pixels = xyz[mask], np.ascontiguousarray(uv[mask], dtype=float)
    low, high = size.width * config.focal_min_width, size.width * config.focal_max_width

    def pose(focal):
        k = intrinsic(float(focal), size)
        ok, rvec, tvec = cv2.solvePnP(
            objects, pixels, k, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            raise ValueError("PnP did not converge")
        return Camera(k, cv2.Rodrigues(rvec)[0], tvec.ravel(), size)

    def residual(log_f):
        camera = pose(np.exp(log_f[0]))
        rvec = Rotation.from_matrix(camera.R).as_rotvec()
        projected, _ = cv2.projectPoints(objects, rvec, camera.t, camera.K, np.zeros(5))
        return (projected[:, 0] - pixels).ravel()

    starts = [size.width * factor for factor in (0.7, 1.5, 3.0, 6.0)]
    if prior is not None:
        starts.insert(0, float(prior.K[0, 0]))
    solutions = []
    for focal in starts:
        try:
            fit = least_squares(
                residual,
                [np.log(np.clip(focal, low * 1.001, high * 0.999))],
                bounds=([np.log(low)], [np.log(high)]),
                loss="soft_l1",
                f_scale=2.0,
                max_nfev=60,
            )
            if fit.success:
                camera = pose(np.exp(fit.x[0]))
                error = float(np.sqrt(np.mean(residual(fit.x) ** 2)))
                if plausible(camera):
                    solutions.append((error, camera))
        except (ValueError, cv2.error, np.linalg.LinAlgError):
            continue
    if not solutions:
        result.reason = "no_plausible_pose"
        return result
    solutions.sort(key=lambda item: item[0])
    error, best = solutions[0]
    result.camera = best
    accepted, result.residual_px = quality(best, uv, mask, config)
    if not accepted:
        result.reason = "reprojection_error"
        return result
    focal = best.K[0, 0]
    if focal < low * 1.01 or focal > high * 0.99:
        result.reason = "focal_at_bound"
        return result
    # Probe the near-optimal planar family, not just optimizer restarts that
    # converge to the same minimum. A weakly determined focal must remain invalid.
    for alternate_f in [focal * 0.65, focal * 1.5]:
        if not low < alternate_f < high:
            continue
        try:
            alternate = pose(alternate_f)
            alternate_error = float(
                np.sqrt(np.mean(residual([np.log(alternate_f)]) ** 2))
            )
            if (
                not plausible(alternate)
                or alternate_error > error + config.ambiguity_slack_px
            ):
                continue
            fractions = np.array([[0.25, 0.25], [0.75, 0.75], [0.5, 0.5]])
            sample = fractions * [size.width, size.height]
            _, a = world_rays(best, sample)
            _, b = world_rays(alternate, sample)
            angle = float(
                np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1))).max()
            )
            height_change = abs(alternate.center[2] / best.center[2] - 1)
            result.alternatives.append(
                [alternate_f, *alternate.center, alternate_error, angle]
            )
            if (
                angle > config.ambiguity_ray_degrees
                or height_change > config.ambiguity_height_fraction
            ):
                result.reason = "ambiguous_focal_depth"
                return result
        except (ValueError, cv2.error):
            continue
    result.valid, result.reason = True, "accepted"
    return result
