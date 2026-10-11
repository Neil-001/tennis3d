"""Lifting-only real evaluation on detection/calibration bytes released by an earlier run.

This script is deliberately outside tennis3d: the numerical package it imports
keeps its own source identity, and the adapter is identified by its file hash
and commit. Detection and calibration never run here. Every reused file is
verified against a hash-pinned artifact manifest and copied into a new output
root; the source gallery is only read. Results are not a fresh end-to-end
detection run.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from fractions import Fraction
from importlib import import_module
from pathlib import Path

import numpy as np

from tennis3d.artifacts import (
    ArtifactProvenance,
    atomic_file,
    canonical_json,
    file_sha256,
    json_sha256,
    read_json,
    read_npz,
    write_json,
    write_npz,
)
from tennis3d.ball.associate import DETECTED, DUPLICATE
from tennis3d.config import load_config, record_config
from tennis3d.evaluation.delivery import (
    checked_release,
    fresh_output,
    write_gallery,
    write_report,
)
from tennis3d.evaluation.inventory import read_inventory, verify_clip
from tennis3d.evaluation.real import execution_report
from tennis3d.evaluation.runs import timeline
from tennis3d.geometry.camera import Camera, ImageSize
from tennis3d.inference.export import EXPORT_INPUTS, build_result
from tennis3d.inference.inputs import (
    DETECTION_FILES,
    OBSERVATIONS,
    check_runtime,
    checked_config,
    hashes,
    source_record,
)
from tennis3d.inference.lift import LIFT_INPUTS, prediction_inputs
from tennis3d.inference.quality import quality_reasons
from tennis3d.inference.settings import ReconstructionConfig
from tennis3d.perception.provenance import (
    CALIBRATION_RUNTIME,
    detection_runtime,
)
from tennis3d.perception.provenance import (
    code_identity as stage_identity,
)
from tennis3d.provenance import code_identity as package_identity
from tennis3d.schemas import Provenance, ReasonCode, read_result, write_result

MANIFEST_SCHEMA = "real-artifact-interface/v1"
UPSTREAM_FILES = (
    "detect.json",
    "detect.config.json",
    *DETECTION_FILES,
    "calibrate.json",
    "calibrate.config.json",
    "cameras.npz",
)
STATEMENT = (
    "Detection, tracking and calibration are inherited cached inputs and were "
    "NOT rerun. Their artifacts were verified against the released artifact "
    "manifest and copied byte-for-byte from an earlier gallery; only lifting, "
    "export and physical consistency are new. Detection and calibration "
    "runtimes, devices and environments recorded in those artifacts are "
    "historical preprocessing."
)


class CachedInputError(Exception):
    """An input or persisted artifact cannot be trusted; the whole run stops."""


def intact(run, sealed):
    """Require every already verified file of a clip to still have its bytes."""
    try:
        changed = sorted(
            name
            for name, digest in hashes(run, sealed).items()
            if digest != sealed[name]
        )
    except OSError as error:
        raise CachedInputError(f"{run.name}: verified artifact vanished") from error
    if changed:
        raise CachedInputError(f"{run.name}: verified artifacts changed: {changed}")


def sealed_json(run, name, value, sealed):
    """Write a report and seal the bytes intended, so a later change is caught."""
    write_json(run / name, value)
    sealed[name] = _sha256(canonical_json(value) + b"\n")


def adapter_identity(commit=None):
    """This file's bytes and commit, separate from the numerical package."""
    script = Path(__file__).resolve()
    digest = file_sha256(script)

    def git(*arguments):
        try:
            return subprocess.check_output(
                ["git", "-C", str(script.parent), *arguments],
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            return None

    source = "argument"
    if commit is None:
        head = git("rev-parse", "HEAD")
        commit, source = (head.decode().strip(), "git_head") if head else (None, None)
    committed = git("show", f"{commit}:scripts/{script.name}") if commit else None
    verified = committed is not None and _sha256(committed) == digest
    if source == "argument" and committed is not None and not verified:
        raise CachedInputError("adapter script differs from --adapter-commit")
    if source == "git_head" and not verified:
        # Uncommitted or untracked edits belong to no commit.
        commit, source = None, None
    return {
        "script": f"scripts/{script.name}",
        "script_sha256": digest,
        "commit": commit,
        "commit_source": source,
        # False for an argument that no local Git checkout could confirm.
        "commit_verified_against_script": verified,
    }


def checked_manifest(path, expected_sha256, inventory_path):
    """Pin the manifest itself, its inventory and its internal agreement."""
    if file_sha256(path) != expected_sha256:
        raise CachedInputError("artifact manifest hash mismatch")
    manifest = read_json(path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise CachedInputError(f"artifact manifest must use {MANIFEST_SCHEMA}")
    if file_sha256(inventory_path) != manifest["original_inventory_sha256"]:
        raise CachedInputError(
            "inventory differs from the artifact manifest's inventory"
        )
    files = manifest["files"]
    for row in manifest["clips"]:
        for name in UPSTREAM_FILES:
            listed = files.get(f"clips/{row['clip_id']}/{name}")
            recorded = row["artifacts"].get(name)
            if (
                listed is not None
                and recorded is not None
                and any(listed[key] != recorded[key] for key in ("bytes", "sha256"))
            ):
                raise CachedInputError(
                    "artifact manifest file and clip records disagree"
                )
    return manifest


def checked_bytes(path, entry, name):
    if entry is None:
        raise CachedInputError(f"artifact manifest does not list {name}")
    try:
        data = Path(path).read_bytes()
    except OSError as error:
        raise CachedInputError(f"cached artifact is unreadable: {name}") from error
    if len(data) != entry["bytes"] or _sha256(data) != entry["sha256"]:
        raise CachedInputError(f"cached artifact differs from manifest: {name}")
    return data


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def verified_upstream(gallery, manifest, clip_id):
    return {
        name: checked_bytes(
            Path(gallery) / "clips" / clip_id / name,
            manifest["files"].get(f"clips/{clip_id}/{name}"),
            f"{clip_id}/{name}",
        )
        for name in UPSTREAM_FILES
    }


def copy_upstream(gallery, manifest, clip_id, run):
    """Verify all nine upstream files, then publish exactly the verified bytes."""
    verified = verified_upstream(gallery, manifest, clip_id)
    for name, data in verified.items():
        with atomic_file(run / name) as stream:
            stream.write(data)
    return hashes(run, UPSTREAM_FILES)


def cached_detection(run, expected_config):
    """The checks of detection_inputs, bound to the recorded producer.

    Only the code-revision equality differs: the producer recorded in the
    arrays is reported verbatim and must be one identity, because the bytes
    are pinned by the manifest and detection does not rerun.
    """
    summary = read_json(run / "detect.json")
    config = checked_config(run, "detect", summary)
    if config != expected_config.model_dump(mode="json"):
        raise ValueError("cached detection configuration differs from the gallery")
    if hashes(run, DETECTION_FILES) != summary["artifacts"]:
        raise ValueError("Detection artifact hash mismatch")
    check_runtime(summary, detection_runtime(summary.get("device")), "Detection")
    arrays, producers = {}, set()
    for name in DETECTION_FILES:
        arrays[name], provenance = read_npz(run / name)
        if provenance != ArtifactProvenance(
            source_sha256=summary["source_sha256"],
            config_sha256=summary["config_sha256"],
            upstream_sha256={
                name: summary["assets"]["assets"][name]["sha256"]
                for name in ("ball", "court")
            },
            code_revision=provenance.code_revision,
        ):
            raise ValueError("cached detection provenance differs from its summary")
        producers.add(provenance.code_revision)
    if len(producers) != 1:
        raise ValueError("cached detection artifacts have different producers")
    frames, track = arrays["frames.npz"], arrays["ball_track.npz"]
    times = timeline(frames["t_s"])
    n = len(times)
    if n != summary["frame_count"] or not np.array_equal(
        frames["frame_index"], np.arange(n)
    ):
        raise ValueError("Detection source indices/frame count disagree")
    if n and times[0] != 0:
        raise ValueError("Normalized source timeline must start at zero")
    for name in ("duplicate_of", "cut", "pts"):
        if frames[name].shape != (n,):
            raise ValueError(f"Invalid source shape: {name}")
    if frames["time_base"].shape != (n, 2) or np.any(frames["time_base"] <= 0):
        raise ValueError("Invalid source time bases")
    if n:
        decoded = frames["pts"] * frames["time_base"][:, 0] / frames["time_base"][:, 1]
        if not np.allclose(
            decoded - float(frames["time_origin_s"]), times, rtol=0, atol=1e-9
        ):
            raise ValueError("Source PTS and normalized timestamps disagree")
    if (
        track["ball_uv"].shape != (n, 2)
        or track["ball_observed"].shape != (n,)
        or track["observation"].shape != (n,)
    ):
        raise ValueError("Track shapes do not match source frames")
    states = track["observation"]
    if states.dtype.kind not in "iu" or np.any(
        (states < 0) | (states >= len(OBSERVATIONS))
    ):
        raise ValueError("Unknown observation code")
    if track["ball_observed"].dtype != np.bool_ or not np.array_equal(
        track["ball_observed"], states == DETECTED
    ):
        raise ValueError("Observation states and accepted mask disagree")
    if not np.isfinite(track["ball_uv"][track["ball_observed"]]).all():
        raise ValueError("Detected pixels must be finite")
    duplicates = frames["duplicate_of"]
    if (
        duplicates.dtype.kind not in "iu"
        or np.any(duplicates < -1)
        or not np.array_equal(duplicates >= 0, states == DUPLICATE)
        or np.any(duplicates >= np.arange(n))
    ):
        raise ValueError("Duplicate mapping disagrees with source indices/states")
    producer = {
        "code_revision": producers.pop(),
        "assets_sha256": {
            name: summary["assets"]["assets"][name]["sha256"]
            for name in ("ball", "court")
        },
        **{
            key: summary.get(key)
            for key in ("device", "runtime", "elapsed_s", "peak_cuda_bytes")
        },
    }
    return summary, frames, track, producer


def cached_cameras(run, detection, frames, expected_config):
    """The checks of camera_inputs, bound to the recorded producer."""
    summary = read_json(run / "calibrate.json")
    config = checked_config(run, "calibrate", summary)
    check_runtime(summary, CALIBRATION_RUNTIME, "Calibration")
    if config != expected_config.model_dump(mode="json"):
        raise ValueError("cached calibration configuration differs from the gallery")
    if file_sha256(run / "cameras.npz") != summary["cameras_sha256"]:
        raise ValueError("Camera artifact hash mismatch")
    arrays, provenance = read_npz(run / "cameras.npz")
    if provenance != ArtifactProvenance(
        source_sha256=detection["source_sha256"],
        config_sha256=summary["config_sha256"],
        upstream_sha256=hashes(run, ("frames.npz", "court.npz", "detect.json")),
        code_revision=provenance.code_revision,
    ):
        raise ValueError("cached camera provenance differs from its inputs")
    n = len(frames["t_s"])
    if not np.array_equal(arrays["t_s"], frames["t_s"]) or summary["frame_count"] != n:
        raise ValueError("Camera and source timestamps/count disagree")
    if list(map(int, frames["image_size"])) != summary["image_size"]:
        raise ValueError("Camera/source dimensions disagree")
    for name, shape in (
        ("camera_valid", (n,)),
        ("K", (n, 3, 3)),
        ("R", (n, 3, 3)),
        ("t", (n, 3)),
        ("court_uv", (n, 14, 2)),
    ):
        if arrays[name].shape != shape:
            raise ValueError(f"Invalid camera array shape: {name}")
    valid = arrays["camera_valid"]
    if valid.dtype != np.bool_ or any(
        not np.isfinite(arrays[key][valid]).all() for key in ("K", "R", "t", "court_uv")
    ):
        raise ValueError("Accepted camera arrays must be finite with boolean validity")
    producer = {
        "code_revision": provenance.code_revision,
        **{key: summary.get(key) for key in ("runtime", "elapsed_s")},
    }
    return summary, arrays, producer


def lift_cached(run, config, detection, frames, track, cameras, cached, identity):
    """The lift stage on already validated cached inputs; numerics are shared."""
    import torch

    from tennis3d.inference.predict import predict_view
    from tennis3d.lifting.checkpoint import load_lifter
    from tennis3d.lifting.runtime import environment, resolve_device, synchronize
    from tennis3d.perception.assets import MissingAssetError

    if not config.checkpoint.is_file():
        raise MissingAssetError(f"Missing lifter checkpoint: {config.checkpoint}")
    device = resolve_device(config.device)
    torch.set_num_threads(config.threads)
    checkpoint_hash = file_sha256(config.checkpoint)
    inputs = hashes(run, LIFT_INPUTS)
    upstream = inputs | {"checkpoint": checkpoint_hash}
    started = time.perf_counter()
    try:
        model, state = load_lifter(config.checkpoint, device=device)
    except (ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f"Incompatible lifter checkpoint: {error}") from error
    load_s = time.perf_counter() - started
    n = len(frames["t_s"])
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    then = time.perf_counter()
    xyz, covered, disagreement = predict_view(
        model,
        {
            "t_s": frames["t_s"],
            "image_size": frames["image_size"],
            "ball_uv": track["ball_uv"],
            "ball_observed": track["ball_observed"],
            "court_uv": cameras["court_uv"],
            "camera_valid": cameras["camera_valid"],
        },
        device=device,
        max_frames=config.max_frames,
        stride=config.stride,
    )
    synchronize(device)
    predict_s = time.perf_counter() - then
    if (
        file_sha256(config.checkpoint) != checkpoint_hash
        or hashes(run, LIFT_INPUTS) != inputs
    ):
        raise CachedInputError(f"{run.name}: Lifting inputs changed during execution")
    config_hash = record_config(run / "lift.config.json", config)
    arrays = {
        "t_s": frames["t_s"],
        "frame_index": frames["frame_index"],
        "xyz_m": xyz,
        "covered": covered,
        "disagreement_m": disagreement,
    }
    write_npz(
        run / "predictions_raw.npz",
        arrays,
        ArtifactProvenance(
            source_sha256=detection["source_sha256"],
            config_sha256=config_hash,
            upstream_sha256=upstream,
            code_revision=identity,
        ),
    )
    count = int(covered.sum())
    summary = {
        "stage": "lift",
        "exit_code": detection.get("exit_code", 0) or (4 if not count else 0),
        "status": "failed"
        if not count or detection["status"] == "failed"
        else "ok"
        if count == n
        else "partial",
        "frame_count": n,
        "raw_covered_fraction": count / n if n else 0.0,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": state["step"],
        "checkpoint_training_identity": {
            key: state.get(key) for key in ("run_id", "manifest_sha256", "provenance")
        },
        "model_label": config.model_label,
        "checkpoint_config": state["config"],
        "weights": "EMA",
        "runtime": environment(device),
        "elapsed_s": time.perf_counter() - started,
        "timings": {"load_s": load_s, "predict_s": predict_s},
        "peak_cuda_bytes": torch.cuda.max_memory_allocated()
        if device == "cuda"
        else None,
        "config_sha256": config_hash,
        "predictions_sha256": file_sha256(run / "predictions_raw.npz"),
        "cached_inputs": cached,
    }
    write_json(run / "lift.json", summary)


def persisted_predictions(run, config, detection, frames, sealed):
    """What is scored is what is on disk: the stock checks on the written lift.

    The new arrays carry this package's stage identity, so the unchanged
    prediction_inputs applies in full, including its code-revision binding.
    """
    try:
        summary, arrays = prediction_inputs(
            run, detection, frames, expected_config=config
        )
        sealed |= hashes(run, ("lift.json", "lift.config.json"))
        # Bind the summary just hashed, not a later rewrite of it.
        if read_json(run / "lift.json") != summary:
            raise ValueError("lift summary changed while it was read")
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise CachedInputError(f"{run.name}: {error}") from error
    sealed["predictions_raw.npz"] = summary["predictions_sha256"]
    return summary, arrays


def export_cached(
    run,
    config,
    detection,
    frames,
    track,
    calibration,
    cameras,
    lifting,
    raw,
    cached,
    identity,
):
    """The export stage on already validated cached inputs; numerics are shared."""
    source = source_record(detection, frames)
    export_identity = {
        "source_sha256": source.sha256,
        "config_sha256": json_sha256(config.model_dump(mode="json")),
        "upstream_sha256": hashes(run, EXPORT_INPUTS),
        "code_identity": identity,
    }
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
    for row in reasons:
        row.extend(reason for reason in failures if reason not in row)
    result = build_result(
        source=source,
        provenance=Provenance(
            model_label=lifting["model_label"],
            lifter_checkpoint_sha256=lifting["checkpoint_sha256"],
            config_sha256=export_identity["config_sha256"],
            calibration_mode=calibration["mode"],
            calibration_assumptions=calibration["assumptions"],
            code_revision=identity,
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
    samples = residual[cameras["camera_valid"][:, None] & np.isfinite(residual)]
    # Only lifting was measured now; inherited stage times stay apart from it.
    metrics = execution_report(
        result,
        calibration_residual_px=samples,
        runtime_s={"lift": lifting["elapsed_s"]},
    )
    metrics["calibration_residual_selection"] = (
        "All finite raw-landmark residuals at accepted cameras; includes non-inliers, no homography pseudo-labels."
    )
    metrics["identity"] = export_identity
    metrics["raw_covered_fraction"] = lifting["raw_covered_fraction"]
    metrics["historical_preprocessing"] = {
        "note": "Recorded by the original detection/calibration run; not measured in this invocation.",
        "runtime_s": {
            "detect": detection["elapsed_s"],
            **(
                {"calibrate": calibration["elapsed_s"]}
                if "elapsed_s" in calibration
                else {}
            ),
        },
        "declared_detection_runtime": {
            key: detection[key]
            for key in ("device", "runtime", "elapsed_s", "peak_cuda_bytes")
        },
    }
    metrics["lifting_runtime"] = lifting["runtime"]
    metrics["cached_inputs"] = cached
    record_config(run / "export.config.json", config)
    write_result(run / "result.json", result)
    write_json(run / "metrics.json", metrics)
    summary = {
        "stage": "export",
        "exit_code": detection.get("exit_code", 0)
        or (2 if failures else 4 if result.status == "failed" else 0),
        "status": result.status,
        "frame_count": n,
        "coverage": result.summary.model_dump(),
        "failure_reasons": failures,
        "identity": export_identity,
        "artifacts": hashes(run, ("result.json", "metrics.json", "export.config.json")),
        "result": str((run / "result.json").resolve()),
        "cached_inputs": cached,
    }
    write_json(run / "export.json", summary)
    return summary


def evaluate_cached_clip(clip, config, gallery, manifest, identities, run, sealed):
    """evaluate_real_clip with reconstruct replaced by verified cached inputs.

    `sealed` receives the hash of each file once it is verified, so the caller
    can prove the clip intact after a failure and again before publication.
    """
    evaluate_physical = import_module("tennis3d.evaluation.physical").evaluate_physical

    started = time.perf_counter()
    # Changed source bytes or timeline are an input failure, never the model's.
    try:
        frozen = verify_clip(clip)
        if clip.source_frame_start != 0:
            raise ValueError(
                "runner requires a preselected complete clip file, not an implicit source crop"
            )
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise CachedInputError(f"{clip.clip_id}: {error}") from error
    assert (
        clip.source_path is not None
        and clip.width is not None
        and clip.height is not None
    )
    # An untrusted cache is never reported as a clip failure of the new model.
    try:
        upstream = copy_upstream(gallery, manifest, clip.clip_id, run)
        sealed |= upstream
        detection, frames, track, detector = cached_detection(run, config.detection)
        if detection["source_sha256"] != clip.source_sha256:
            raise ValueError("Detection cache belongs to a different source")
        calibration = cameras = calibrator = None
        if len(frames["t_s"]):
            # Before any model runs: the stage rows are the frozen timeline's.
            rows = frozen["frames"]
            pts = [
                Fraction(row["pts"] * row["time_base"][0], row["time_base"][1])
                for row in rows
            ]
            expected = np.array([float(value - pts[0]) for value in pts])
            if (
                len(frames["t_s"]) != len(rows)
                or list(map(int, frames["image_size"])) != [clip.width, clip.height]
                or not np.allclose(frames["t_s"], expected, atol=1e-9, rtol=0)
                or not np.array_equal(frames["pts"], [row["pts"] for row in rows])
                or not np.array_equal(
                    frames["time_base"], [row["time_base"] for row in rows]
                )
            ):
                raise ValueError("stage rows differ from the frozen source timeline")
            calibration, cameras, calibrator = cached_cameras(
                run, detection, frames, config.calibration
            )
            if calibrator["code_revision"] != detector["code_revision"]:
                raise ValueError("cached stages have different producers")
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise CachedInputError(f"{clip.clip_id}: {error}") from error
    cached = {
        "declaration": STATEMENT,
        "detection_rerun": False,
        "calibration_rerun": False,
        "artifact_manifest_sha256": identities["artifact_manifest_sha256"],
        "upstream_sha256": upstream,
        # Verbatim from the cached arrays; a different scheme from the package's.
        "upstream_producer": {
            "code_revision": detector["code_revision"],
            "detection": detector,
            "calibration": calibrator,
        },
        "numerical_package": identities["numerical_package"],
        "adapter": identities["adapter"],
    }
    identity = identities["numerical_package"]["stage_code_identity"]
    if calibration is None or cameras is None:
        # reconstruct stops here as well: no rows exist to calibrate or lift.
        summary = {
            "stage": "reconstruct_cached_inputs",
            "status": "failed",
            "exit_code": detection.get("exit_code", 2),
            "failure_reasons": ["decoder_error"],
            "error": detection["failure"] or "No decoded frames",
            "frame_count": 0,
            "coverage": {"valid_frame_fraction": 0.0},
            "source_sha256": clip.source_sha256,
            "detect_sha256": upstream["detect.json"],
            "cached_inputs": cached,
        }
        sealed_json(run, "manifest.json", summary, sealed)
        return {
            "status": "failed",
            "pipeline": summary,
            "failure_reasons": summary["failure_reasons"],
        }, None
    timings = {}
    then = time.perf_counter()
    lift_cached(
        run, config.lifting, detection, frames, track, cameras, cached, identity
    )
    timings["lift_wall_s"] = time.perf_counter() - then
    lifting, raw = persisted_predictions(run, config.lifting, detection, frames, sealed)
    timings |= lifting["timings"]
    then = time.perf_counter()
    exported = export_cached(
        run,
        config.quality,
        detection,
        frames,
        track,
        calibration,
        cameras,
        lifting,
        raw,
        cached,
        identity,
    )
    timings["export_wall_s"] = time.perf_counter() - then
    # The export must have read the sealed bytes and its own files must be whole.
    written = tuple(exported["artifacts"])
    if exported["identity"]["upstream_sha256"] != {
        name: sealed[name] for name in EXPORT_INPUTS
    }:
        raise CachedInputError(f"{run.name}: export inputs changed during execution")
    sealed |= exported["artifacts"]
    sealed["export.json"] = _sha256(canonical_json(exported) + b"\n")
    intact(run, {name: sealed[name] for name in (*written, "export.json")})
    result = read_result(run / "result.json")
    if (
        result.source.sha256 != clip.source_sha256
        or result.source.frame_count != clip.frame_count
        or (result.source.width, result.source.height) != (clip.width, clip.height)
    ):
        raise CachedInputError(
            f"{clip.clip_id}: result does not preserve the frozen source identity"
        )
    camera_objects = [
        Camera(
            cameras["K"][i],
            cameras["R"][i],
            cameras["t"][i],
            ImageSize(clip.width, clip.height),
        )
        if valid
        else None
        for i, valid in enumerate(cameras["camera_valid"])
    ]
    metadata = {
        "artifact_id": clip.clip_id,
        "mode": "end_to_end",
        "domain": "real",
        "split": "test",
        "time_basis": "source_pts",
        "timing_note": clip.timing_note,
        "source_identity": {
            "source_sha256": clip.source_sha256,
            "width": clip.width,
            "height": clip.height,
            "frame_count": clip.frame_count,
            "recording_group": clip.recording_group,
            "rights": clip.rights_evidence,
        },
        "prediction_identity": {
            "checkpoint_sha256": lifting["checkpoint_sha256"],
            "weights": "EMA",
            "filtering": "raw predictions before export quality rules",
        },
        "segmentation_source": "automatic",
        "segmentation_method": "velocity_jump_v1",
        "event_grouping": "prediction-derived velocity jumps; engineering thresholds fixed before outputs; no contact truth",
        "image_reference": {
            "kind": "detections",
            "identity": detection["artifacts"]["ball_track.npz"],
            "dependence": "same input detector; dependent consistency measure",
        },
        "camera_source": calibration["mode"],
        "bounce_convention": "ball_center",
    }
    numeric = {
        "t_s": frames["t_s"],
        "xyz_m": raw["xyz_m"],
        "available": raw["covered"],
        "input_observed": track["ball_observed"],
        "reference_uv_px": track["ball_uv"],
        "reference_valid": track["ball_observed"],
        "camera_K": cameras["K"],
        "camera_R": cameras["R"],
        "camera_t": cameras["t"],
        "camera_valid": cameras["camera_valid"],
    }
    then = time.perf_counter()
    fit = evaluate_physical(numeric, metadata=metadata, cameras=camera_objects)
    timings["physical_fit_s"] = time.perf_counter() - then
    sealed_json(run, "physical.json", fit, sealed)
    timings["clip_wall_s"] = time.perf_counter() - started
    summary = {
        "stage": "reconstruct_cached_inputs",
        "status": exported["status"],
        "exit_code": exported["exit_code"],
        "source_sha256": clip.source_sha256,
        "frame_count": exported["frame_count"],
        "coverage": exported["coverage"],
        "failure_reasons": exported["failure_reasons"],
        "result": exported["result"],
        "reused_stages": ["detect", "calibrate"],
        "invocation_seconds": timings["clip_wall_s"],
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
        "cached_inputs": cached,
    }
    sealed_json(run, "manifest.json", summary, sealed)
    execution = read_json(run / "metrics.json")
    intact(run, sealed)
    return {
        "status": result.status,
        "failure_reasons": list(
            dict.fromkeys(
                [
                    *summary["failure_reasons"],
                    *result.failure_reasons,
                    *execution.get("failure_reasons", {}),
                ]
            )
        ),
        "pipeline": summary,
        "execution": execution,
        "physical_path": f"clips/{clip.clip_id}/physical.json",
        "result_path": f"clips/{clip.clip_id}/result.json",
        "tracking": None,
        "landing": None,
        "labelled_lifting": None,
    }, fit


def cached_real(
    release_path,
    inventory_path,
    gallery,
    manifest_path,
    manifest_sha256,
    output,
    *,
    device="cpu",
    model_label=None,
    package_source_sha256=None,
    adapter_commit=None,
):
    """The `real` command with reconstruct replaced by manifest-verified caches.

    An untrusted manifest, inventory, gallery configuration, upstream file,
    checkpoint or persisted new artifact raises CachedInputError and stops the
    whole run; no clip is silently dropped or downgraded. A failure of the new
    lifting/export/physics work is recorded for its clip only after every
    verified file and the checkpoint are proven unchanged. Failed or
    unsupported states recorded by the original detection stay in the report
    exactly as `real` keeps them. A stopped run has no metrics.json and an
    aborted.json naming the cause.
    """
    started = time.perf_counter()
    package = package_identity()
    if package_source_sha256 not in (None, package["source_sha256"]):
        raise CachedInputError(
            "imported tennis3d source differs from --package-source-sha256"
        )
    gallery = Path(gallery).resolve()
    manifest = checked_manifest(manifest_path, manifest_sha256, inventory_path)
    release, _, provenance = checked_release(release_path)
    inventory = read_inventory(inventory_path)
    if any(clip.eligible and clip.labels_path for clip in inventory.clips):
        raise CachedInputError(
            "cached-input evaluation does not implement labelled clips; use `real`"
        )
    # The gallery's resolved config is itself a pinned artifact. Detection and
    # calibration sections describe the copied bytes; only lifting changes.
    config_entry = manifest["files"].get("config.json")
    config_bytes = checked_bytes(gallery / "config.json", config_entry, "config.json")
    lifting = {"device": device, "checkpoint": release.checkpoint.path}
    if model_label is not None:
        lifting["model_label"] = model_label
    config = load_config(
        ReconstructionConfig, gallery / "config.json", overrides={"lifting": lifting}
    )
    if (gallery / "config.json").read_bytes() != config_bytes:
        raise CachedInputError("gallery configuration changed while it was read")
    # Every reused byte is checked before the output root exists.
    for clip in inventory.clips:
        if clip.eligible:
            verified_upstream(gallery, manifest, clip.clip_id)
    identities = {
        "artifact_manifest_sha256": manifest_sha256,
        "numerical_package": {
            "source_sha256": package["source_sha256"],
            "code_revision": package["code_revision"],
            "expected_source_sha256": package_source_sha256,
            # The scheme stock stages stamp into arrays and results.
            "stage_code_identity": stage_identity(),
        },
        "adapter": adapter_identity(adapter_commit),
    }
    if Path(output).resolve() == gallery or gallery in Path(output).resolve().parents:
        raise CachedInputError("output must be outside the source gallery")
    output = fresh_output(output)
    try:
        return _evaluate(
            release,
            provenance,
            inventory,
            inventory_path,
            config,
            gallery,
            manifest,
            manifest_path,
            identities | {"package": package, "adapter_commit": adapter_commit},
            output,
            device,
            started,
        )
    except BaseException as error:
        # No metrics.json exists yet; say why so the root cannot pass as complete.
        write_json(
            output / "aborted.json",
            {
                "status": "aborted",
                "error": f"{type(error).__name__}: {error}",
                "note": "Incomplete cached-input evaluation; not a gallery. Start again with a new output root.",
            },
            overwrite=True,
        )
        raise


def _evaluate(
    release,
    provenance,
    inventory,
    inventory_path,
    config,
    gallery,
    manifest,
    manifest_path,
    identities,
    output,
    device,
    started,
):
    summarize_physical = import_module(
        "tennis3d.evaluation.physical_controls"
    ).summarize_physical
    from tennis3d.lifting.runtime import environment, resolve_device

    package, manifest_sha256 = (
        identities["package"],
        identities["artifact_manifest_sha256"],
    )

    def checkpoint_intact():
        try:
            changed = file_sha256(release.checkpoint.path) != release.checkpoint.sha256
        except OSError as error:
            raise CachedInputError("released checkpoint vanished") from error
        if changed:
            raise CachedInputError("checkpoint changed during real evaluation")

    # Root metadata is sealed like clip files: metrics.json records the config
    # hash and consumers reread both files.
    records, physics, seals = [], [], {output: {}}
    sealed_json(output, "inventory.json", inventory.model_dump(), seals[output])
    sealed_json(output, "config.json", config.model_dump(mode="json"), seals[output])
    for clip in inventory.clips:
        record = {
            "clip_id": clip.clip_id,
            "recording_group": clip.recording_group,
            "source": clip.model_dump(),
        }
        if not clip.eligible:
            record.update(status="ineligible", failure_reasons=clip.barriers)
        else:
            run = output / "clips" / clip.clip_id
            sealed = seals.setdefault(run, {})
            try:
                diagnostics, fit = evaluate_cached_clip(
                    clip, config, gallery, manifest, identities, run, sealed
                )
                record.update(diagnostics)
                if fit is not None:
                    physics.append(fit)
            except (ValueError, OSError, RuntimeError, ImportError) as error:
                # Only a failure on proven-intact inputs belongs to the model.
                intact(run, sealed)
                checkpoint_intact()
                record.update(
                    status="failed",
                    failure_reasons=[f"{type(error).__name__}: {error}"],
                )
        records.append(record)
        write_json(
            output / "progress.json",
            {
                "processed": len(records),
                "preselected": len(inventory.clips),
                "records": records,
            },
            overwrite=True,
        )
    then = time.perf_counter()
    physical = summarize_physical(physics) if physics else None
    summary_s = time.perf_counter() - then
    measured = [
        record["pipeline"]["invocation_timings"]
        for record in records
        if "invocation_timings" in record.get("pipeline", {})
    ]
    producers = sorted(
        {
            record["pipeline"]["cached_inputs"]["upstream_producer"]["code_revision"]
            for record in records
            if "pipeline" in record
        }
    )
    cached = {
        "declaration": STATEMENT,
        "detection_rerun": False,
        "calibration_rerun": False,
        "new_stages": ["lift", "export", "physical"],
        "inherited_stages": ["detect", "calibrate"],
        "artifact_manifest": {
            "path": str(Path(manifest_path).resolve()),
            "sha256": manifest_sha256,
            "schema": manifest["schema"],
            "original_inventory_sha256": manifest["original_inventory_sha256"],
            "original_release_sha256": manifest.get("release_sha256"),
            "original_checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "original_metrics_sha256": manifest.get("metrics_sha256"),
        },
        "source_gallery": str(gallery),
        "source_gallery_config_sha256": manifest["files"]["config.json"]["sha256"],
        "upstream_files": list(UPSTREAM_FILES),
        "upstream_producer_code_revisions": producers,
        "numerical_package": identities["numerical_package"]
        | {"source_files": package["source_files"]},
        "adapter": identities["adapter"],
        "environment": {
            **environment(resolve_device(device)),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "lifting_threads": config.lifting.threads,
            **{
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
        # Each value is its own timer or a sum of such timers, never a difference.
        "measured_timings_s": {
            key: sum(row[key] for row in measured)
            for key in (
                "load_s",
                "predict_s",
                "lift_wall_s",
                "export_wall_s",
                "physical_fit_s",
                "clip_wall_s",
            )
        }
        | {"physical_summary_s": summary_s},
    }
    result = {
        "schema_version": "1.0",
        "kind": "end_to_end_real",
        "status": (
            "incomplete_inventory"
            if not inventory.summary()["required_gallery_ready"]
            else "measured"
            if any(record.get("result_path") for record in records)
            else "no_results"
        ),
        "inventory_sha256": file_sha256(inventory_path),
        "inventory": inventory.summary(),
        "provenance": provenance | {"cached_inputs": cached},
        "config_sha256": json_sha256(config.model_dump(mode="json")),
        "records": records,
        "physical": physical,
        "limitations": inventory.limitations
        + [
            "Dense real XYZ truth unavailable",
            "Unknown detector training overlap prevents held-out detector claims",
            "Detection-reference reprojection shares detector errors",
            "Physical consistency is not independent XYZ truth",
            "Lifting-only evaluation on cached inputs, not a fresh end-to-end run. "
            + STATEMENT,
        ],
    }
    write_report(output, result)
    with (output / "report.md").open("a") as stream:
        stream.write(f"Cached inputs: {STATEMENT}\n")
    write_gallery(output, result)
    # metrics.json is the completion mark: it is written last, and only if every
    # verified byte of the root metadata and every clip, the checkpoint and the
    # code still stand.
    for run, sealed in seals.items():
        intact(run, sealed)
    checkpoint_intact()
    if (
        package_identity()["source_sha256"] != package["source_sha256"]
        or adapter_identity(identities["adapter_commit"]) != identities["adapter"]
    ):
        raise CachedInputError("package or adapter source changed during evaluation")
    cached["process_wall_s"] = time.perf_counter() - started
    write_json(output / "metrics.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "inventory", "gallery", "artifact-manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--artifact-manifest-sha256", required=True)
    parser.add_argument(
        "--package-source-sha256",
        help="required tennis3d.provenance source_sha256 of the imported package",
    )
    parser.add_argument(
        "--adapter-commit", help="commit this script was merged at; checked if possible"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--model-label", help="viewer label; defaults to the source gallery's label"
    )
    args = parser.parse_args(argv)
    try:
        result = cached_real(
            args.release,
            args.inventory,
            args.gallery,
            args.artifact_manifest,
            args.artifact_manifest_sha256,
            args.output,
            device=args.device,
            model_label=args.model_label,
            package_source_sha256=args.package_source_sha256,
            adapter_commit=args.adapter_commit,
        )
    except (
        CachedInputError,
        ValueError,
        OSError,
        RuntimeError,
        ImportError,
        KeyError,
        TypeError,
    ) as error:
        print(f"Evaluation unavailable: {error}")
        return 2
    # Records repeat every clip's diagnostics; print the run identity instead.
    cached = dict(result["provenance"]["cached_inputs"])
    cached["numerical_package"] = {
        key: value
        for key, value in cached["numerical_package"].items()
        if key != "source_files"
    }
    print(
        json.dumps(
            {
                "status": result["status"],
                "records": {
                    record["clip_id"]: record["status"] for record in result["records"]
                },
                "cached_inputs": cached,
            },
            allow_nan=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
