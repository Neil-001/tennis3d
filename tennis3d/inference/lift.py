"""Persist raw predictions from the shared lifter, preserving every source row."""

import importlib.metadata
import time
from pathlib import Path

import numpy as np

from tennis3d.artifacts import (
    ArtifactProvenance,
    file_sha256,
    read_json,
    read_npz,
    write_json,
    write_npz,
)
from tennis3d.config import record_config
from tennis3d.inference.inputs import (
    camera_inputs,
    checked_config,
    detection_inputs,
    hashes,
)
from tennis3d.perception.assets import MissingAssetError
from tennis3d.perception.provenance import code_identity

LIFT_INPUTS = (
    "detect.json",
    "frames.npz",
    "ball_track.npz",
    "calibrate.json",
    "cameras.npz",
)


def prediction_inputs(run, detection, frames, *, expected_config=None):
    summary = read_json(run / "lift.json")
    config = checked_config(run, "lift", summary)
    for name in ("torch", "numpy"):
        if importlib.metadata.version(name) != summary["runtime"][name]:
            raise ValueError(f"Lifting runtime changed: {name}; use --overwrite")
    if expected_config is not None and config != expected_config.model_dump(
        mode="json"
    ):
        raise ValueError("Lifting cache configuration changed; use --overwrite")
    checkpoint = Path(config["checkpoint"])
    if not checkpoint.is_file():
        raise MissingAssetError(f"Missing lifter checkpoint: {checkpoint}")
    upstream = hashes(run, LIFT_INPUTS) | {"checkpoint": file_sha256(checkpoint)}
    if summary["checkpoint_sha256"] != upstream["checkpoint"]:
        raise ValueError("Lifter checkpoint changed; use --overwrite")
    if file_sha256(run / "predictions_raw.npz") != summary["predictions_sha256"]:
        raise ValueError("Raw prediction hash mismatch")
    expected = ArtifactProvenance(
        source_sha256=detection["source_sha256"],
        config_sha256=summary["config_sha256"],
        upstream_sha256=upstream,
        code_revision=code_identity(),
    )
    arrays, _ = read_npz(run / "predictions_raw.npz", expected=expected)
    n = len(frames["t_s"])
    if not np.array_equal(arrays["t_s"], frames["t_s"]) or not np.array_equal(
        arrays["frame_index"], frames["frame_index"]
    ):
        raise ValueError("Raw predictions do not preserve source frame/time identity")
    for key, shape in (("xyz_m", (n, 3)), ("covered", (n,)), ("disagreement_m", (n,))):
        if arrays[key].shape != shape:
            raise ValueError(f"Invalid prediction shape: {key}")
    if arrays["covered"].dtype != np.bool_:
        raise ValueError("Prediction coverage must be boolean")
    return summary, arrays


def lift(run: Path, config, *, overwrite=False, progress=None):
    detection, frames, track = detection_inputs(run)
    _, cameras = camera_inputs(run, detection, frames)
    if (run / "lift.json").exists() and not overwrite:
        return prediction_inputs(run, detection, frames, expected_config=config)[0]
    if not config.checkpoint.is_file():
        raise MissingAssetError(f"Missing lifter checkpoint: {config.checkpoint}")
    import torch

    from tennis3d.inference.predict import predict_view
    from tennis3d.lifting.checkpoint import load_lifter
    from tennis3d.lifting.runtime import environment, resolve_device, synchronize

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
    n = len(frames["t_s"])
    # Features/tensors are assembled by predict_view one <=250-row window at a
    # time. Only source-aligned numeric arrays and merged results scale with T.
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
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
        progress=progress,
    )
    synchronize(device)
    if (
        file_sha256(config.checkpoint) != checkpoint_hash
        or hashes(run, LIFT_INPUTS) != inputs
    ):
        raise ValueError("Lifting inputs changed during execution")
    config_hash = record_config(run / "lift.config.json", config, overwrite=overwrite)
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
            code_revision=code_identity(),
        ),
        overwrite=overwrite,
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
        "checkpoint_step": state.get("step"),
        "checkpoint_training_identity": {
            key: state.get(key) for key in ("run_id", "manifest_sha256", "provenance")
        },
        "model_label": config.model_label,
        "checkpoint_config": state["config"],
        "weights": "EMA",
        "runtime": environment(device),
        "elapsed_s": time.perf_counter() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated()
        if device == "cuda"
        else None,
        "config_sha256": config_hash,
        "predictions_sha256": file_sha256(run / "predictions_raw.npz"),
    }
    write_json(run / "lift.json", summary, overwrite=overwrite)
    return summary
