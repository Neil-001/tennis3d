"""Calibration-dependent bounce XY references at exact annotation timestamps."""

from typing import Literal

import numpy as np

from tennis3d.evaluation.annotations import AnnotationManifest
from tennis3d.evaluation.real import error_summary
from tennis3d.evaluation.runs import timeline
from tennis3d.geometry.camera import Camera
from tennis3d.geometry.projection import world_rays


def bounce_reference(
    camera: Camera,
    uv,
    *,
    convention: Literal["ball_center", "ground"],
    ball_radius_m: float = 0.0335,
) -> np.ndarray:
    """Intersect the annotated pixel ray with Z=radius or Z=0, never an airborne ray fit."""
    if convention not in ("ball_center", "ground"):
        raise ValueError("declare ball_center or ground annotation convention")
    if not np.isfinite(ball_radius_m) or ball_radius_m <= 0:
        raise ValueError("ball radius must be positive and finite")
    center, direction = world_rays(camera, uv)
    if direction.shape != (3,) or abs(direction[2]) < 1e-10:
        raise ValueError(
            "bounce ray is parallel to contact plane or not a single point"
        )
    z = ball_radius_m if convention == "ball_center" else 0.0
    distance = (z - center[2]) / direction[2]
    if distance <= 0:
        raise ValueError("contact plane is behind the annotated ray")
    return center + distance * direction


def landing_metrics(
    labels: AnnotationManifest,
    t_s,
    xyz_m,
    available,
    cameras: list[Camera | None],
    *,
    source_sha256: str,
    camera_source: str,
    convention: Literal["ball_center", "ground"],
    ball_radius_m: float = 0.0335,
) -> dict:
    """Score each bounce-labelled source frame, with no prediction-time search.

    Repeated bounce rows remain separate. This default is explicitly row-weighted,
    not an independent-event count. Invalid/unavailable rows remain denominators.
    Physical bounce evaluation refuses nominal timestamps or unknown source PTS.
    """
    times = timeline(t_s)
    xyz = np.asarray(xyz_m, dtype=np.float64)
    valid = np.asarray(available)
    n = labels.source_frame_count
    if labels.time_basis != "source_pts" or source_sha256 != labels.source_sha256:
        raise ValueError(
            "landing evaluation needs verified source identity and source PTS"
        )
    if (
        len(times) != n
        or xyz.shape != (n, 3)
        or valid.shape != (n,)
        or valid.dtype != np.bool_
        or len(cameras) != n
    ):
        raise ValueError("landing inputs must retain every source frame")
    if (
        convention not in ("ball_center", "ground")
        or not np.isfinite(ball_radius_m)
        or ball_radius_m <= 0
    ):
        raise ValueError("invalid landing plane convention/radius")
    if not np.isfinite(xyz[valid]).all():
        raise ValueError("available XYZ must be finite")
    records = []
    for row in labels.annotations:
        i = row.frame_index
        if row.t_s != float(times[i]):
            raise ValueError("annotation timestamp does not match the source frame")
        if row.event != "bounce":
            continue
        record: dict = {
            "frame_index": i,
            "t_s": row.t_s,
            "reference_xy_m": None,
            "error_xy_m": None,
            "reason": None,
        }
        camera = cameras[i]
        if row.uv is None:
            record["reason"] = "missing_annotation_location"
        elif camera is None:
            record["reason"] = "invalid_camera"
        else:
            if (camera.image_size.width, camera.image_size.height) != (
                labels.width,
                labels.height,
            ):
                raise ValueError("camera dimensions differ from annotation pixels")
            try:
                reference = bounce_reference(
                    camera, row.uv, convention=convention, ball_radius_m=ball_radius_m
                )
            except ValueError:
                record["reason"] = "invalid_bounce_ray"
            else:
                record["reference_xy_m"] = reference[:2].tolist()
                if valid[i]:
                    record["error_xy_m"] = float(
                        np.linalg.norm(xyz[i, :2] - reference[:2])
                    )
                else:
                    record["reason"] = "unavailable_prediction"
        records.append(record)
    errors = [row["error_xy_m"] for row in records if row["error_xy_m"] is not None]
    return {
        "reference": "calibration_dependent_annotation_ray",
        "camera_source": camera_source,
        "source_sha256": source_sha256,
        "labels_sha256": labels.labels_sha256,
        "convention": convention,
        "plane_z_m": ball_radius_m if convention == "ball_center" else 0.0,
        "aggregation": "each bounce-labelled frame; repeated labels retained; no predicted-event matching",
        "annotated_bounce_rows": len(records),
        "scored_rows": len(errors),
        "coverage": len(errors) / len(records) if records else None,
        "landing_xy_error_m": error_summary(errors),
        "records": records,
        "limitation": "No airborne XYZ accuracy; references inherit calibration and annotation error.",
    }
