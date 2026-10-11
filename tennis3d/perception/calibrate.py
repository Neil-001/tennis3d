"""Calibration artifact production from independent court observations."""

import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from tennis3d.artifacts import (
    ArtifactProvenance,
    file_sha256,
    read_json,
    read_npz,
    write_json,
    write_npz,
)
from tennis3d.config import record_config
from tennis3d.court.calibrate import estimate_camera, intrinsic, quality
from tennis3d.court.model import court_landmarks
from tennis3d.court.track import CameraFilter
from tennis3d.geometry.camera import Camera, ImageSize
from tennis3d.geometry.projection import project
from tennis3d.perception.provenance import (
    CALIBRATION_RUNTIME,
    code_identity,
    runtime_versions,
)

REASONS = [
    "accepted",
    "insufficient_landmarks",
    "homography_failed",
    "insufficient_spatial_support",
    "no_plausible_pose",
    "reprojection_error",
    "focal_at_bound",
    "ambiguous_focal_depth",
    "duplicate",
    "shot_cut",
    "unavailable",
    "measured",
    "predicted",
    "innovation_reset",
    "unsupported_gap",
    "filtered_reprojection_error",
    "static_reprojection_error",
]


def calibrate(run: Path, config, *, overwrite=False, progress=None):
    from tennis3d.inference.progress import report

    started = time.perf_counter()
    detection = read_json(run / "detect.json")
    for name in ("frames.npz", "court.npz"):
        if file_sha256(run / name) != detection["artifacts"][name]:
            raise ValueError(f"Detection artifact hash mismatch: {name}")
    frames, frame_provenance = read_npz(run / "frames.npz")
    court, court_provenance = read_npz(run / "court.npz")
    if frame_provenance != court_provenance:
        raise ValueError("Court and frame provenance disagree")
    times = frames["t_s"]
    n = len(times)
    if n == 0:
        raise ValueError("Cannot calibrate an empty decode")
    if times[0] != 0 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid camera timeline")
    if court["raw_uv"].shape != (n, 14, 2) or court["observed"].shape != (n, 14):
        raise ValueError("Court artifact shape mismatch")
    size = ImageSize(*map(int, frames["image_size"]))
    config_hash = record_config(
        run / "calibrate.config.json", config, overwrite=overwrite
    )
    arrays = {
        "t_s": times,
        "K": np.full((n, 3, 3), np.nan),
        "R": np.full((n, 3, 3), np.nan),
        "t": np.full((n, 3), np.nan),
        "camera_valid": np.zeros(n, bool),
        "raw_K": np.full((n, 3, 3), np.nan),
        "raw_R": np.full((n, 3, 3), np.nan),
        "raw_t": np.full((n, 3), np.nan),
        "raw_valid": np.zeros(n, bool),
        "raw_reason": np.zeros(n, np.int16),
        "filter_reason": np.zeros(n, np.int16),
        "inliers": np.zeros((n, 14), bool),
        "raw_residual_px": np.full((n, 14), np.nan),
        "filtered_residual_px": np.full((n, 14), np.nan),
        "homography_uv": np.full((n, 14, 2), np.nan),
        "court_uv": np.full((n, 14, 2), np.nan),
    }
    estimates, alternatives, offsets = [], [], [0]
    prior, cut_seen = None, False
    for index in range(n):
        report(progress, "calibrate", index, n * 2)
        estimate = estimate_camera(
            court["raw_uv"][index], court["observed"][index], size, config, prior
        )
        cut_seen = cut_seen or bool(frames["cut"][index])
        if frames["duplicate_of"][index] >= 0:
            estimate.valid, estimate.reason = False, "duplicate"
        if cut_seen:
            estimate.valid, estimate.reason = False, "shot_cut"
        estimates.append(estimate)
        arrays["raw_valid"][index] = estimate.valid
        arrays["raw_reason"][index] = REASONS.index(estimate.reason)
        arrays["inliers"][index] = estimate.inliers
        arrays["raw_residual_px"][index] = estimate.residual_px
        arrays["homography_uv"][index] = estimate.homography_uv
        alternatives.extend(estimate.alternatives)
        offsets.append(len(alternatives))
        if estimate.camera is not None:
            for name in ("K", "R", "t"):
                arrays["raw_" + name][index] = getattr(estimate.camera, name)
        if estimate.valid:
            prior = estimate.camera
    tracker = CameraFilter(config)
    static = None
    reliable = [e.camera for e in estimates if e.valid]
    if config.mode == "static" and reliable:
        center = np.median([c.center for c in reliable], axis=0)
        quaternions = Rotation.from_matrix(np.stack([c.R for c in reliable])).as_quat()
        # q and -q denote the same rotation. Align signs before taking a robust
        # componentwise median so one raw pose cannot rotate the entire shot.
        quaternions *= np.where(quaternions @ quaternions[0] < 0, -1, 1)[:, None]
        rotation = Rotation.from_quat(np.median(quaternions, axis=0)).as_matrix()
        focal = np.median([c.K[0, 0] for c in reliable])
        static = Camera(intrinsic(focal, size), rotation, -rotation @ center, size)
    for index, estimate in enumerate(estimates):
        report(progress, "calibrate", n + index, n * 2)
        camera, reason = tracker.update(
            float(times[index]),
            estimate,
            court["raw_uv"][index],
            size,
            cut=estimate.reason == "shot_cut",
        )
        if static is not None and camera is not None:
            camera = static
            if (
                estimate.valid
                and not quality(
                    camera, court["raw_uv"][index], estimate.inliers, config
                )[0]
            ):
                camera, reason = None, "static_reprojection_error"
        arrays["filter_reason"][index] = REASONS.index(reason)
        if camera is not None:
            arrays["camera_valid"][index] = True
            for name in ("K", "R", "t"):
                arrays[name][index] = getattr(camera, name)
            arrays["court_uv"][index] = project(camera, court_landmarks())[0]
            arrays["filtered_residual_px"][index] = np.linalg.norm(
                arrays["court_uv"][index] - court["raw_uv"][index], axis=1
            )
    report(progress, "calibrate", n * 2, n * 2)
    arrays["alternatives"] = np.asarray(alternatives).reshape(-1, 6)
    arrays["alternative_offsets"] = np.asarray(offsets, dtype=np.int64)
    provenance = ArtifactProvenance(
        source_sha256=frame_provenance.source_sha256,
        config_sha256=config_hash,
        upstream_sha256={
            name: file_sha256(run / name)
            for name in ("frames.npz", "court.npz", "detect.json")
        },
        code_revision=code_identity(),
    )
    write_npz(run / "cameras.npz", arrays, provenance, overwrite=overwrite)
    count = int(arrays["camera_valid"].sum())
    summary = {
        "stage": "calibrate",
        "runtime": runtime_versions(CALIBRATION_RUNTIME),
        "elapsed_s": time.perf_counter() - started,
        "exit_code": detection.get("exit_code", 0)
        or (2 if detection["status"] == "failed" else 4 if not count else 0),
        "status": "failed"
        if detection["status"] == "failed" or not count
        else "ok"
        if count == n
        else "partial",
        "frame_count": n,
        "valid_camera_fraction": count / n,
        "mode": "estimated_static"
        if config.mode == "static"
        else "estimated_per_frame",
        "assumptions": [
            "square pixels",
            "zero skew",
            "centered principal point",
            "zero distortion",
            "ideal court dimension-line intersections",
            "canonical baseline view",
        ],
        "matrix_direction": "world_to_camera",
        "units": "metres",
        "image_size": [size.width, size.height],
        "reason_codes": REASONS,
        "config_sha256": config_hash,
        "cameras_sha256": file_sha256(run / "cameras.npz"),
        "raw_reason_counts": {
            reason: int((arrays["raw_reason"] == i).sum())
            for i, reason in enumerate(REASONS)
            if (arrays["raw_reason"] == i).any()
        },
    }
    write_json(run / "calibrate.json", summary, overwrite=overwrite)
    return summary
