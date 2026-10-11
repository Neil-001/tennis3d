"""Apply quality rules to raw predictions and write the viewer result and diagnostics."""

from pathlib import Path

import numpy as np

from tennis3d.artifacts import json_sha256, read_json, write_json
from tennis3d.config import record_config
from tennis3d.evaluation.real import execution_report
from tennis3d.evaluation.runs import timeline
from tennis3d.geometry.camera import Camera, ImageSize
from tennis3d.inference.inputs import (
    OBSERVATIONS,
    camera_inputs,
    detection_inputs,
    hashes,
    source_record,
)
from tennis3d.inference.lift import prediction_inputs
from tennis3d.inference.quality import quality_reasons
from tennis3d.perception.provenance import code_identity
from tennis3d.schemas import (
    Frame,
    Observation,
    Provenance,
    ReasonCode,
    Result,
    Source,
    Summary,
    read_result,
    write_result,
)

EXPORT_INPUTS = (
    "detect.json",
    "frames.npz",
    "ball_track.npz",
    "calibrate.json",
    "cameras.npz",
    "lift.json",
    "predictions_raw.npz",
)


def export(run: Path, config, *, overwrite=False):
    detection, frames, track = detection_inputs(run)
    calibration, cameras = camera_inputs(run, detection, frames)
    lifting, raw = prediction_inputs(run, detection, frames)
    source = source_record(detection, frames)
    identity = {
        "source_sha256": source.sha256,
        "config_sha256": json_sha256(config.model_dump(mode="json")),
        "upstream_sha256": hashes(run, EXPORT_INPUTS),
        "code_identity": code_identity(),
    }
    if (run / "export.json").exists() and not overwrite:
        saved = read_json(run / "export.json")
        if saved["identity"] != identity or saved["artifacts"] != hashes(
            run, ("result.json", "metrics.json", "export.config.json")
        ):
            raise ValueError("Export cache identity changed; use --overwrite")
        read_result(run / "result.json")
        return saved
    n = source.frame_count
    depth = np.zeros(n, dtype=bool)
    size = ImageSize(source.width, source.height)
    for i in np.flatnonzero(cameras["camera_valid"] & raw["covered"]):
        if np.isfinite(raw["xyz_m"][i]).all():
            camera = Camera(cameras["K"][i], cameras["R"][i], cameras["t"][i], size)
            depth[i] = camera.world_to_camera(raw["xyz_m"][i])[2] > 0
    observations = [OBSERVATIONS[int(value)] for value in track["observation"]]
    reasons = quality_reasons(
        frames["t_s"],
        observations,
        cameras["camera_valid"],
        raw["xyz_m"],
        raw["covered"],
        raw["disagreement_m"],
        config,
        positive_depth=depth,
    )
    failures: list[ReasonCode] = []
    if detection["shot_cut"]:
        failures.append("unsupported_shot")
    if detection["failure"] or not n:
        failures.append("decoder_error")
    # An input-contract failure is visible and unplayable even if a prefix had
    # numerical predictions. Preserve the raw prefix separately for diagnosis.
    for row in reasons:
        row.extend(reason for reason in failures if reason not in row)
    result = build_result(
        source=source,
        provenance=Provenance(
            model_label=lifting["model_label"],
            lifter_checkpoint_sha256=lifting["checkpoint_sha256"],
            config_sha256=identity["config_sha256"],
            calibration_mode=calibration["mode"],
            calibration_assumptions=calibration["assumptions"],
            code_revision=identity["code_identity"],
        ),
        t_s=frames["t_s"],
        observations=observations,
        duplicate_of=frames["duplicate_of"],
        camera_valid=cameras["camera_valid"],
        xyz_m=raw["xyz_m"],
        reasons=reasons,
        failure_reasons=failures,
    )
    residual = cameras["filtered_residual_px"]
    # Report residuals only where raw independent landmark evidence exists.
    samples = residual[cameras["camera_valid"][:, None] & np.isfinite(residual)]
    metrics = execution_report(
        result,
        calibration_residual_px=samples,
        runtime_s={
            "detect": detection["elapsed_s"],
            "lift": lifting["elapsed_s"],
            **(
                {"calibrate": calibration["elapsed_s"]}
                if "elapsed_s" in calibration
                else {}
            ),
        },
    )
    metrics["calibration_residual_selection"] = (
        "All finite raw-landmark residuals at accepted cameras; includes non-inliers, no homography pseudo-labels."
    )
    metrics["identity"] = identity
    metrics["raw_covered_fraction"] = lifting["raw_covered_fraction"]
    metrics["declared_detection_runtime"] = {
        key: detection[key]
        for key in ("device", "runtime", "elapsed_s", "peak_cuda_bytes")
    }
    metrics["lifting_runtime"] = lifting["runtime"]
    record_config(run / "export.config.json", config, overwrite=overwrite)
    write_result(run / "result.json", result, overwrite=overwrite)
    write_json(run / "metrics.json", metrics, overwrite=overwrite)
    summary = {
        "stage": "export",
        "exit_code": detection.get("exit_code", 0)
        or (2 if failures else 4 if result.status == "failed" else 0),
        "status": result.status,
        "frame_count": n,
        "coverage": result.summary.model_dump(),
        "failure_reasons": failures,
        "identity": identity,
        "artifacts": hashes(run, ("result.json", "metrics.json", "export.config.json")),
        "result": str((run / "result.json").resolve()),
    }
    write_json(run / "export.json", summary, overwrite=overwrite)
    return summary


def build_result(
    *,
    source: Source,
    provenance: Provenance,
    t_s,
    observations: list[Observation],
    duplicate_of,
    camera_valid,
    xyz_m,
    reasons: list[list[ReasonCode]],
    failure_reasons: list[ReasonCode] | None = None,
) -> Result:
    times = timeline(t_s)
    n = source.frame_count
    xyz = np.asarray(xyz_m, dtype=np.float64)
    cameras = np.asarray(camera_valid)
    duplicates = np.asarray(duplicate_of)
    if len(times) != n or len(observations) != n or len(reasons) != n:
        raise ValueError("export must preserve every source frame")
    if xyz.shape != (n, 3) or cameras.shape != (n,) or cameras.dtype != np.bool_:
        raise ValueError(
            "export prediction/camera arrays must match source frame count"
        )
    if duplicates.shape != (n,) or duplicates.dtype.kind not in "iu":
        raise ValueError(
            "duplicate_of must be an integer source-frame array; -1 means absent"
        )
    frames = []
    for i in range(n):
        available = not reasons[i]
        frames.append(
            Frame(
                frame_index=i,
                t_s=float(times[i]),
                xyz_m=xyz[i].tolist() if available else None,
                observation=observations[i],
                reconstruction=(
                    "estimated" if observations[i] == "detected" else "inferred"
                )
                if available
                else "unavailable",
                camera_valid=bool(cameras[i]),
                reason_codes=reasons[i],
                duplicate_of=int(duplicates[i]) if duplicates[i] >= 0 else None,
            )
        )
    valid = sum(frame.xyz_m is not None for frame in frames)
    detected = sum(frame.observation == "detected" for frame in frames)
    failures = failure_reasons or []
    status = "failed" if failures or not valid else "ok" if valid == n else "partial"
    return Result(
        status=status,
        source=source,
        provenance=provenance,
        frames=frames,
        summary=Summary(
            valid_frame_fraction=valid / n if n else 0.0,
            detected_frame_fraction=detected / n if n else 0.0,
        ),
        failure_reasons=failures,
    )
