"""Synthetic truth export using the shared viewer contract, never a model claim."""

import numpy as np

from tennis3d.schemas import Result


def result_payload(
    arrays: dict[str, np.ndarray],
    *,
    name: str,
    source_hash: str,
    config_hash: str,
    revision: str,
) -> dict:
    times = arrays["t_s"]
    observed = arrays["ball_observed"]
    frames, projections = [], []
    for i, t in enumerate(times):
        seen = bool(observed[i])
        frames.append(
            {
                "frame_index": i,
                "t_s": float(t),
                "xyz_m": arrays["xyz_m"][i].tolist(),
                "observation": "detected"
                if seen
                else ("missing" if arrays["ball_in_frame"][i] else "out_of_frame"),
                "reconstruction": "estimated" if seen else "inferred",
                "camera_valid": True,
                "reason_codes": [],
            }
        )
        projections.append(
            {
                "frame_index": i,
                "t_s": float(t),
                "ball_clean_uv": nullable_uv(arrays["ball_clean_uv"][i]),
                "ball_observed_uv": nullable_uv(arrays["ball_uv"][i]),
                "court_uv": [nullable_uv(p) for p in arrays["court_uv"][i]],
            }
        )
    kinds = {1: "bounce", 2: "hit", 4: "out"}
    events = [
        {"kind": kinds[int(code)], "t_s": float(t), "xyz_m": p.tolist()}
        for code, t, p in zip(
            arrays["event_code"], arrays["event_t_s"], arrays["event_xyz_m"]
        )
        if int(code) in kinds
    ]
    return {
        "status": "ok",
        "source": {
            "name": name,
            "sha256": source_hash,
            "width": int(arrays["image_size"][0]),
            "height": int(arrays["image_size"][1]),
            "frame_count": len(times),
            "time_origin_s": 0.0,
            "time_base": [1, 1000000],
        },
        "frames": frames,
        "summary": {
            "valid_frame_fraction": 1.0,
            "detected_frame_fraction": float(observed.mean()),
        },
        "provenance": {
            "lifter_checkpoint_sha256": None,
            "config_sha256": config_hash,
            "calibration_mode": "synthetic",
            "calibration_assumptions": [
                "Exact simulated XYZ; corrupted observations and projected camera context."
            ],
            "trajectory_kind": "synthetic_ground_truth",
            "code_revision": revision,
        },
        "synthetic_diagnostics": {
            "kind": "synthetic",
            "frames": projections,
            "events": events,
        },
    }


def nullable_uv(point: np.ndarray):
    return point.tolist() if np.isfinite(point).all() else None


def truth_result(arrays: dict[str, np.ndarray], **metadata) -> Result:
    return Result.model_validate(result_payload(arrays, **metadata))
