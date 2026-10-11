"""Compose the existing stages, reusing only validated current-code caches."""

import time
from pathlib import Path

from tennis3d.artifacts import file_sha256, read_json, write_json
from tennis3d.inference.export import export
from tennis3d.inference.inputs import camera_inputs, detection_inputs, hashes
from tennis3d.inference.lift import lift
from tennis3d.perception.assets import manifest, verify


def reconstruct(video: Path, run: Path, config, *, overwrite=False, progress=None):
    from tennis3d.inference.progress import report

    report(progress, "checking", unit="stages")
    if not video.is_file():
        raise FileNotFoundError(video)
    started = time.perf_counter()
    source_hash = file_sha256(video)
    if (run / "manifest.json").exists() and not overwrite:
        previous = read_json(run / "manifest.json")
        if (
            previous.get("stage") != "reconstruct"
            or previous.get("source_sha256") != source_hash
        ):
            raise ValueError(
                "Existing manifest belongs to an unrelated run; use --overwrite"
            )
    assets = manifest(config.detection.assets_manifest)
    for name in ("ball", "court"):
        verify(assets.assets[name])
    timings, reused = {}, []
    if (run / "detect.json").exists() and not overwrite:
        detection, frames, _ = detection_inputs(
            run, expected_config=config.detection, source_sha256=source_hash
        )
        if detection["assets"] != assets.model_dump(mode="json"):
            raise ValueError("Detection asset manifest changed; use --overwrite")
        reused.append("detect")
    else:
        from tennis3d.perception.detect import detect

        then = time.perf_counter()
        detect(
            video,
            run,
            config.detection,
            overwrite=overwrite,
            **({"progress": progress} if progress else {}),
        )
        timings["detect_wall_s"] = time.perf_counter() - then
        detection, frames, _ = detection_inputs(
            run, expected_config=config.detection, source_sha256=source_hash
        )
    if not len(frames["t_s"]):
        summary = {
            "stage": "reconstruct",
            "status": "failed",
            "exit_code": detection.get("exit_code", 2),
            "failure_reasons": ["decoder_error"],
            "error": detection["failure"] or "No decoded frames",
            "frame_count": 0,
            "coverage": {"valid_frame_fraction": 0.0},
            "source_sha256": source_hash,
            "detect_sha256": file_sha256(run / "detect.json"),
        }
        write_json(run / "manifest.json", summary, overwrite=True)
        return summary
    if (run / "calibrate.json").exists() and not overwrite:
        camera_inputs(run, detection, frames, expected_config=config.calibration)
        reused.append("calibrate")
    else:
        from tennis3d.perception.calibrate import calibrate

        then = time.perf_counter()
        calibrate(
            run,
            config.calibration,
            overwrite=overwrite,
            **({"progress": progress} if progress else {}),
        )
        timings["calibrate_wall_s"] = time.perf_counter() - then
    for name, operation, options in (
        ("lift", lift, config.lifting),
        ("export", export, config.quality),
    ):
        cached = (run / f"{name}.json").exists() and not overwrite
        then = time.perf_counter()
        report(progress, name, unit="stages")
        extra = {"progress": progress} if name == "lift" and progress else {}
        summary = operation(run, options, overwrite=overwrite, **extra)
        timings[f"{name}_wall_s"] = time.perf_counter() - then
        if cached:
            reused.append(name)
    result = {
        "stage": "reconstruct",
        "status": summary["status"],
        "exit_code": summary["exit_code"],
        "source_sha256": source_hash,
        "frame_count": summary["frame_count"],
        "coverage": summary["coverage"],
        "failure_reasons": summary["failure_reasons"],
        "result": summary["result"],
        "reused_stages": reused,
        "invocation_seconds": time.perf_counter() - started,
        "invocation_timings": timings,
        "config": config.model_dump(mode="json"),
        "artifacts": hashes(
            run,
            (
                "detect.json",
                "calibrate.json",
                "lift.json",
                "export.json",
                "result.json",
                "metrics.json",
            ),
        ),
    }
    if file_sha256(video) != source_hash:
        raise ValueError("Source video changed during reconstruction")
    report(progress, "complete", 1, 1, "stages")
    write_json(run / "manifest.json", result, overwrite=True)
    return result
