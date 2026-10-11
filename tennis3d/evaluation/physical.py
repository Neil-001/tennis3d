"""Numeric-artifact physical consistency, with explicit contact and reference provenance."""

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict

import numpy as np

from tennis3d.evaluation.annotations import AnnotationManifest
from tennis3d.evaluation.landing import bounce_reference
from tennis3d.evaluation.metrics import summarize
from tennis3d.evaluation.physical_ode import (
    DEFAULT_CONFIG,
    DEFAULT_PHYSICS,
    FitConfig,
    FlightParameters,
    fit_flight,
    integrate_flight,
)
from tennis3d.evaluation.real import error_summary
from tennis3d.evaluation.runs import timeline
from tennis3d.geometry.camera import Camera, ImageSize
from tennis3d.geometry.projection import project

__all__ = ["FitConfig", "FlightParameters", "evaluate_physical"]

DEFINITIONS = {
    "residual": "fitted ODE minus predicted XYZ; consistency, never true XYZ accuracy",
    "rmse_m": "sqrt(mean(sum((fitted_m-predicted_m)^2, axis=XYZ))) over every available segment row",
    "max_discrepancy_m": "maximum Euclidean fitted-minus-predicted distance over the same rows",
    "fit": "coordinate-wise soft_l1 least squares; loss_scale_m is not a quality threshold",
    "aggregation": "segment-weighted residual distributions; frame-weighted errors within each segment; no view/time deduplication",
    "observed": "input_observed counts accepted 2D input observations; all available predicted XYZ participate in fitting",
    "contacts": "exclude closed contact episodes expanded by guard and supplied precision; never fit across a contact, availability hole or excessive time gap",
    "landing": "preceding flight extrapolated only to contact onset at exact source time, never a searched predicted bounce or a continuation through contact",
    "identifiability": "scaled robust Jacobian rank is a local numerical diagnostic, not evidence of unique or accurate spin",
}


def _mask(arrays, key, n):
    value = np.asarray(arrays[key])
    if value.shape != (n,) or value.dtype != np.bool_:
        raise ValueError(f"{key} must be a boolean vector matching timestamps")
    return value


def _xyz(arrays, key, n, mask):
    value = np.asarray(arrays[key], dtype=np.float64)
    if value.shape != (n, 3) or not np.isfinite(value[mask]).all():
        raise ValueError(f"{key} must be N by 3 with finite supported XYZ")
    return value


def _metadata(metadata):
    # Copies provenance and rejects NaN, numpy values and opaque objects.
    result = json.loads(json.dumps(dict(metadata), allow_nan=False))
    for key in ("artifact_id", "split", "timing_note", "segmentation_method"):
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ValueError(f"metadata needs nonempty {key}")
    for key in ("source_identity", "prediction_identity"):
        if not isinstance(result.get(key), dict) or not result[key]:
            raise ValueError(f"metadata needs nonempty {key} mapping")
    if result.get("domain") not in ("synthetic", "real", "controlled_fixture"):
        raise ValueError("declare synthetic, real or controlled_fixture domain")
    if result.get("time_basis") not in ("simulation", "source_pts", "declared_nominal"):
        raise ValueError(
            "declare actual simulation/source_pts or explicit nominal diagnostic times"
        )
    if result["domain"] == "real" and result["time_basis"] == "simulation":
        raise ValueError("real footage cannot claim simulation timestamps")
    if result.get("segmentation_source") not in (
        "annotated",
        "automatic",
        "simulator",
        "manual_events",
    ):
        raise ValueError(
            "declare annotated, automatic, simulator or manual_events contact segmentation"
        )
    if result["segmentation_source"] == "manual_events":
        identity = result.get("manual_event_identity", {})
        if not isinstance(identity, dict):
            raise ValueError("manual_events requires a manual_event_identity mapping")
        for key, mapping in (
            ("source_sha256", result["source_identity"]),
            ("annotation_sha256", identity),
        ):
            value = mapping.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"manual_events requires {key} SHA-256")
        if (
            result["time_basis"] != "source_pts"
            or result["segmentation_method"] != "manual-events/v1"
            or identity.get("source_sha256")
            != result["source_identity"]["source_sha256"]
        ):
            raise ValueError(
                "manual_events requires source_pts, matching source hash and manual-events/v1"
            )
    return result


def _annotation_contacts(labels):
    """Consecutive same-event source rows form one closed episode; absent rows break it."""
    episodes = []
    previous = None
    for row in labels.annotations:
        if row.event in ("hit", "bounce"):
            if (
                previous is not None
                and row.frame_index == previous.frame_index + 1
                and row.event == previous.event
            ):
                episodes[-1]["end_t_s"] = row.t_s
            else:
                episodes.append(
                    {
                        "start_t_s": row.t_s,
                        "end_t_s": row.t_s,
                        "kind": row.event,
                        "precision_s": 0.0,
                    }
                )
        previous = row
    return episodes


def _automatic_contacts(times, xyz, available, config):
    """Declared velocity-jump heuristic, using actual cadence and only adjacent valid rows."""
    if len(times) < 3:
        return []
    events: list[dict] = []
    for i in range(1, len(times) - 1):
        dt = np.diff(times[i - 1 : i + 2])
        if not available[i - 1 : i + 2].all() or np.max(dt) > config.max_time_gap_s:
            continue
        before, after = np.diff(xyz[i - 1 : i + 2], axis=0) / dt[:, None]
        if np.linalg.norm(after - before) < config.automatic_jump_m_s:
            continue
        kind = "bounce" if before[2] < 0 < after[2] else "hit"
        event = {
            "start_t_s": float(times[i]),
            "end_t_s": float(times[i]),
            "kind": kind,
            "precision_s": float(max(dt) / 2),
        }
        if (
            events
            and event["start_t_s"] - events[-1]["end_t_s"] <= config.automatic_merge_s
        ):
            events[-1]["end_t_s"] = event["end_t_s"]
            events[-1]["precision_s"] = max(
                events[-1]["precision_s"], event["precision_s"]
            )
        else:
            events.append(event)
    return events


def _contacts(arrays, metadata, times, xyz, available, labels, config):
    if metadata["segmentation_source"] == "manual_events":
        if labels is not None:
            raise ValueError(
                "manual_events uses numeric contacts, not AnnotationManifest"
            )
        if not all(
            k in arrays for k in ("event_t_s", "event_code", "event_precision_s")
        ):
            raise ValueError(
                "manual_events requires explicit numeric times, codes and precision"
            )
    if metadata["segmentation_source"] == "annotated":
        if labels is None:
            raise ValueError(
                "annotated segmentation requires the normalized AnnotationManifest"
            )
        if any(k in arrays for k in ("event_t_s", "event_end_t_s", "event_code")):
            raise ValueError(
                "supply annotations or event arrays, not two contact authorities"
            )
        return (
            _annotation_contacts(labels),
            "consecutive same-event annotated source indices; gaps/noncontact rows break episodes",
        )
    if "event_t_s" not in arrays:
        if any(
            k in arrays for k in ("event_end_t_s", "event_code", "event_precision_s")
        ):
            raise ValueError("event arrays require event_t_s")
        if (
            metadata["segmentation_source"] != "automatic"
            or metadata["segmentation_method"] != "velocity_jump_v1"
        ):
            raise ValueError(
                "supply event arrays, even empty, or declare automatic velocity_jump_v1"
            )
        return _automatic_contacts(
            times, xyz, available, config
        ), "adjacent XYZ velocity jump; candidates within automatic_merge_s merged"
    starts = np.asarray(arrays["event_t_s"], dtype=np.float64)
    ends = np.asarray(arrays.get("event_end_t_s", starts), dtype=np.float64)
    precision = np.asarray(
        arrays.get("event_precision_s", np.zeros(len(starts))), dtype=np.float64
    )
    codes = np.asarray(arrays["event_code"])
    if (
        starts.ndim != 1
        or ends.shape != starts.shape
        or codes.shape != starts.shape
        or precision.shape != starts.shape
        or not np.issubdtype(codes.dtype, np.integer)
    ):
        raise ValueError(
            "event arrays must be matching one-dimensional vectors with integer codes"
        )
    if (
        not np.isfinite(starts).all()
        or not np.isfinite(ends).all()
        or not np.isfinite(precision).all()
        or np.any(starts < 0)
        or np.any(ends < starts)
        or np.any(precision < 0)
    ):
        raise ValueError(
            "contact times/precision must be finite, nonnegative and ordered"
        )
    events = [
        {
            "start_t_s": float(s),
            "end_t_s": float(e),
            "kind": "bounce" if c == 1 else "hit",
            "precision_s": float(p),
        }
        for s, e, c, p in zip(starts, ends, codes, precision)
        if c in (1, 2)
    ]
    if metadata["segmentation_source"] == "manual_events" and (
        not np.isin(codes, [1, 2]).all()
        or (
            len(starts)
            and (not len(times) or starts.min() < times[0] or ends.max() > times[-1])
        )
    ):
        raise ValueError(
            "manual_events requires hit/bounce codes within source timeline"
        )
    events.sort(key=lambda e: e["start_t_s"])
    return (
        events,
        "supplied event episodes; event_code 1 bounce/2 hit; other codes ignored; no implicit row grouping",
    )


def _camera_list(arrays, metadata, cameras, n):
    if cameras is not None:
        if len(cameras) != n or any(
            c is not None and not isinstance(c, Camera) for c in cameras
        ):
            raise ValueError(
                "cameras must contain one accepted Camera or None per source row"
            )
        return cameras
    keys = ("camera_K", "camera_R", "camera_t", "camera_valid")
    if not any(k in arrays for k in keys):
        return [None] * n
    if not all(k in arrays for k in keys):
        raise ValueError(
            "numeric cameras require camera_K/R/t/valid and metadata.image_size"
        )
    dims = metadata.get("image_size")
    if not isinstance(dims, list) or len(dims) != 2:
        raise ValueError("metadata.image_size must be [original width, height]")
    size = ImageSize(*dims)
    values = [np.asarray(arrays[k], dtype=np.float64) for k in keys[:3]]
    valid = _mask(arrays, "camera_valid", n)
    if any(
        v.shape != shape for v, shape in zip(values, ((n, 3, 3), (n, 3, 3), (n, 3)))
    ):
        raise ValueError("numeric cameras must retain each source row")
    return [
        Camera(values[0][i], values[1][i], values[2][i], size) if valid[i] else None
        for i in range(n)
    ]


def _references(arrays, metadata, labels, times, cameras):
    n = len(times)
    uv, valid = np.full((n, 2), np.nan), np.zeros(n, dtype=bool)
    if labels is not None:
        if "reference_uv_px" in arrays or "reference_valid" in arrays:
            raise ValueError("supply labels or numeric image reference, not both")
        if (
            labels.time_basis != "source_pts"
            or metadata["time_basis"] != "source_pts"
            or metadata["source_identity"].get("source_sha256") != labels.source_sha256
            or labels.source_frame_count != n
        ):
            raise ValueError(
                "labels require verified matching source identity, frame count and source PTS"
            )
        for row in labels.annotations:
            if row.t_s != float(times[row.frame_index]):
                raise ValueError(
                    "annotation timestamps differ from numeric source rows"
                )
            if row.uv is not None:
                uv[row.frame_index], valid[row.frame_index] = row.uv, True
        identity = {
            "kind": "annotations",
            "labels_sha256": labels.labels_sha256,
            "pixel_convention": labels.pixel_convention,
            "visibility": "includes class 3 inferred coordinates; no independent XYZ truth",
        }
        for camera in cameras:
            if camera is not None and (
                camera.image_size.width,
                camera.image_size.height,
            ) != (labels.width, labels.height):
                raise ValueError(
                    "camera dimensions differ from original annotation pixels"
                )
    elif "reference_uv_px" in arrays or "reference_valid" in arrays:
        uv = np.asarray(arrays["reference_uv_px"], dtype=np.float64)
        valid = _mask(arrays, "reference_valid", n)
        identity = metadata.get("image_reference")
        if (
            uv.shape != (n, 2)
            or not np.isfinite(uv[valid]).all()
            or not isinstance(identity, dict)
            or identity.get("kind") not in ("annotations", "detections")
            or not identity.get("identity")
        ):
            raise ValueError(
                "numeric UV references need finite original pixels and explicit kind/identity"
            )
    else:
        identity = {"kind": "unavailable", "reason": "no_supplied_image_reference"}
    if (
        valid.any()
        and any(c is not None for c in cameras)
        and not metadata.get("camera_source")
    ):
        raise ValueError("image agreement requires declared camera_source")
    return uv, valid, identity


def _agreement(segment, times, uv, valid, cameras, contacts, physics, config, metadata):
    indices = segment["frame_indices"]
    image = []
    for i, p in zip(indices, segment["fitted_xyz_m"]):
        if not valid[i]:
            continue
        record = {
            "frame_index": i,
            "t_s": float(times[i]),
            "error_px": None,
            "reason": None,
        }
        if cameras[i] is None:
            record["reason"] = "invalid_camera"
        else:
            projected, positive = project(cameras[i], p)
            if not positive:
                record["reason"] = "nonpositive_depth"
            else:
                record["error_px"] = float(np.linalg.norm(projected - uv[i]))
        image.append(record)
    landing = []
    last = indices[-1]
    # Only the immediately following contact is eligible. A missing interval is not filled.
    following = [c for c in contacts if c["start_t_s"] > times[last]]
    if following:
        contact = following[0]
        if contact["kind"] == "bounce" and segment["is_contact_predecessor"]:
            event_time = contact["start_t_s"]
            rows = np.flatnonzero(times == event_time)
            record = {
                "t_s": event_time,
                "frame_index": int(rows[0]) if len(rows) else None,
                "error_xy_m": None,
                "reference_xy_m": None,
                "reason": None,
                "extrapolation_s": float(event_time - times[last]),
            }
            if metadata["time_basis"] != "source_pts":
                record["reason"] = "landing_requires_source_pts"
            elif event_time - times[last] > config.max_time_gap_s or (
                len(rows)
                and rows[0] > last + 1
                and not segment["contact_bridge_available"]
            ):
                record["reason"] = "gap_before_contact"
            elif not len(rows) or not valid[rows[0]]:
                record["reason"] = "no_exact_contact_reference"
            elif cameras[rows[0]] is None:
                record["reason"] = "invalid_camera"
            elif metadata.get("bounce_convention") not in ("ball_center", "ground"):
                record["reason"] = "undeclared_bounce_convention"
            else:
                i = int(rows[0])
                try:
                    reference = bounce_reference(
                        cameras[i],
                        uv[i],
                        convention=metadata["bounce_convention"],
                        ball_radius_m=physics.radius_m,
                    )
                except ValueError:
                    record["reason"] = "invalid_bounce_ray"
                else:
                    state = segment["initial_state"]
                    curve = integrate_flight(
                        [event_time - times[indices[0]]],
                        np.r_[
                            state["position_m"],
                            state["velocity_m_s"],
                            state["spin_rad_s"],
                        ],
                        physics,
                        config,
                    )[0, :3]
                    record["reference_xy_m"] = reference[:2].tolist()
                    record["error_xy_m"] = float(
                        np.linalg.norm(curve[:2] - reference[:2])
                    )
            landing.append(record)
    return {
        "reprojection": {
            "reference_rows": len(image),
            "scored_rows": sum(r["error_px"] is not None for r in image),
            "error_px": error_summary(
                [r["error_px"] for r in image if r["error_px"] is not None]
            ),
            "records": image,
        },
        "landing": {
            "reference": "calibration_dependent_contact_ray",
            "convention": metadata.get("bounce_convention"),
            "records": landing,
            "error_xy_m": error_summary(
                [r["error_xy_m"] for r in landing if r["error_xy_m"] is not None]
            ),
        },
    }


def evaluate_physical(
    arrays: Mapping,
    *,
    metadata: Mapping,
    physics=DEFAULT_PHYSICS,
    config=DEFAULT_CONFIG,
    labels: AnnotationManifest | None = None,
    cameras: list[Camera | None] | None = None,
) -> dict:
    """Fit an already released numeric artifact. No model/data-source reads or quality filtering."""
    metadata = _metadata(metadata)
    times = timeline(arrays["t_s"])
    n = len(times)
    available, observed = (_mask(arrays, k, n) for k in ("available", "input_observed"))
    xyz = _xyz(arrays, "xyz_m", n, available)
    truth, truth_valid = None, np.zeros(n, dtype=bool)
    if "truth_xyz_m" in arrays or "label_valid" in arrays:
        if not all(k in arrays for k in ("truth_xyz_m", "label_valid")):
            raise ValueError(
                "synthetic controls require both truth_xyz_m and label_valid"
            )
        if metadata["domain"] == "real":
            raise ValueError("real artifacts must not claim synthetic XYZ ground truth")
        truth_valid = _mask(arrays, "label_valid", n)
        truth = _xyz(arrays, "truth_xyz_m", n, truth_valid)
    camera_list = _camera_list(arrays, metadata, cameras, n)
    uv, reference_valid, reference_identity = _references(
        arrays, metadata, labels, times, camera_list
    )
    contacts, grouping = _contacts(
        arrays, metadata, times, xyz, available, labels, config
    )
    contact_mask = np.zeros(n, dtype=bool)
    # Number of episode starts before each row also splits contacts between sampled rows.
    region = np.zeros(n, dtype=int)
    for contact in contacts:
        guard = config.contact_guard_s + contact["precision_s"]
        contact_mask |= (times >= contact["start_t_s"] - guard) & (
            times <= contact["end_t_s"] + guard
        )
        region += times > contact["start_t_s"]
    eligible = available & ~contact_mask
    groups, current = [], []
    for i in range(n):
        if current and (
            not eligible[i]
            or times[i] - times[current[-1]] > config.max_time_gap_s
            or region[i] != region[current[-1]]
        ):
            groups.append(current)
            current = []
        if eligible[i]:
            current.append(i)
    if current:
        groups.append(current)
    segments: list[dict] = []
    for indices in groups:
        start, stop = indices[0], indices[-1]
        count_observed = int(observed[indices].sum())
        record: dict = {
            "segment_id": len(segments),
            "frame_indices": indices,
            "start_t_s": float(times[start]),
            "end_t_s": float(times[stop]),
            "duration_s": float(times[stop] - times[start]),
            "frames": len(indices),
            "observed_frames": count_observed,
        }
        reasons = []
        if len(indices) < config.min_frames:
            reasons.append("too_few_frames")
        if count_observed < config.min_observed_frames:
            reasons.append("too_few_observed_frames")
        if times[stop] - times[start] < config.min_duration_s:
            reasons.append("too_short")
        if reasons:
            record.update(
                status="excluded", reason=reasons[0], exclusion_reasons=reasons
            )
        else:
            record.update(fit_flight(times[indices], xyz[indices], physics, config))
            if record["status"] == "fitted":
                if truth is not None:
                    supported = truth_valid[indices]
                    record["true_xyz_error_m"] = summarize(
                        (xyz[indices] - truth[indices])[supported]
                    )
                    record["true_xyz_supported_fraction"] = float(supported.mean())
                following = [c for c in contacts if c["start_t_s"] > times[stop]]
                before_contact = (
                    (times > times[stop]) & (times < following[0]["start_t_s"])
                    if following
                    else np.zeros(n, dtype=bool)
                )
                record["is_contact_predecessor"] = not bool(
                    eligible[before_contact].any()
                )
                # The contact guard is intentionally excluded, including rows
                # where the exporter refuses a prediction near the bounce.
                record["contact_bridge_available"] = bool(
                    available[before_contact & ~contact_mask].all()
                )
                record["agreement"] = _agreement(
                    record,
                    times,
                    uv,
                    reference_valid,
                    camera_list,
                    contacts,
                    physics,
                    config,
                    metadata,
                )
        segments.append(record)
    statuses = Counter(s["status"] for s in segments)
    fitted = [s for s in segments if s["status"] == "fitted"]
    reasons = Counter(
        r
        for s in segments
        for r in s.get("exclusion_reasons", [s["reason"]] if s["reason"] else [])
    )
    landing_records = []
    for contact in contacts:
        if contact["kind"] != "bounce":
            continue
        match = [
            (s["segment_id"], r)
            for s in fitted
            for r in s["agreement"]["landing"]["records"]
            if r["t_s"] == contact["start_t_s"]
        ]
        if match:
            segment_id, row = match[0]
            landing_records.append({"segment_id": segment_id, **row})
        else:
            landing_records.append(
                {
                    "segment_id": None,
                    "t_s": contact["start_t_s"],
                    "error_xy_m": None,
                    "reason": "no_fitted_precontact_segment",
                }
            )
    image_records = [
        r for s in fitted for r in s["agreement"]["reprojection"]["records"]
    ]
    result = {
        "schema_version": "1.0",
        "metadata": metadata,
        "scientific_status": "nominal_time_diagnostic"
        if metadata["time_basis"] == "declared_nominal"
        else "physical_consistency_only",
        "physics": {
            **asdict(physics),
            "derived_coefficients": physics.coefficients,
            "model": "mujoco_sphere_equivalent_v1",
        },
        "config": asdict(config),
        "definitions": DEFINITIONS,
        "segmentation": {
            "source": metadata["segmentation_source"],
            "method": metadata["segmentation_method"],
            "grouping": grouping,
            "contacts": contacts,
        },
        "reference": {
            "image": reference_identity,
            "camera_source": metadata.get("camera_source"),
            "dependence": "Detector reference may share the lifter input; neither image nor landing agreement establishes airborne XYZ accuracy.",
        },
        "counts": {
            "source_frames": n,
            "available_frames": int(available.sum()),
            "unavailable_frames": int((~available).sum()),
            "contact_guard_frames": int(contact_mask.sum()),
            "eligible_frames": int(eligible.sum()),
            "candidate_segments": len(segments),
            "fitted_segments": statuses["fitted"],
            "excluded_segments": statuses["excluded"],
            "failed_segments": statuses["failed"],
            "scored_frames": sum(s["frames"] for s in fitted),
        },
        "reason_counts": dict(reasons),
        "segments": segments,
        "independent_agreement": {
            "source_reference_rows": int(reference_valid.sum()),
            "same_segment_reference_rows": len(image_records),
            "reprojection_error_px": error_summary(
                [r["error_px"] for r in image_records if r["error_px"] is not None]
            ),
            "bounce_episodes": len(landing_records),
            "landing_error_xy_m": error_summary(
                [
                    r["error_xy_m"]
                    for r in landing_records
                    if r["error_xy_m"] is not None
                ]
            ),
            "landing_records": landing_records,
        },
        "residual_distributions_m": {
            key: {
                **error_summary([s[key] for s in fitted]),
                "max": max((s[key] for s in fitted), default=None),
            }
            for key in ("rmse_m", "max_discrepancy_m")
        },
        "unconditional_xyz_error_m": None,
    }
    if truth is not None:
        supported = truth_valid & available
        result["unconditional_xyz_error_m"] = {
            **summarize(xyz[supported] - truth[supported]),
            "labelled_frames": int(truth_valid.sum()),
            "unavailable_labelled_frames": int((truth_valid & ~available).sum()),
            "complete": bool(available[truth_valid].all()),
        }
    json.dumps(result, allow_nan=False)
    return result
