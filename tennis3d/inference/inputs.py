"""Validate stage identities and source alignment before cached inference/export."""

import importlib.metadata
from pathlib import Path

import numpy as np

from tennis3d.artifacts import (
    ArtifactProvenance,
    file_sha256,
    json_sha256,
    read_json,
    read_npz,
)
from tennis3d.ball.associate import DETECTED, DUPLICATE
from tennis3d.evaluation.runs import timeline
from tennis3d.perception.provenance import (
    CALIBRATION_RUNTIME,
    code_identity,
    detection_runtime,
)
from tennis3d.schemas import Observation, Source

DETECTION_FILES = ("frames.npz", "court.npz", "ball_candidates.npz", "ball_track.npz")
OBSERVATIONS: tuple[Observation, ...] = (
    "missing",
    "detected",
    "ambiguous",
    "duplicate",
)


def hashes(run: Path, names) -> dict:
    return {name: file_sha256(run / name) for name in names}


def checked_config(run, stage, summary):
    config = read_json(run / f"{stage}.config.json")
    if json_sha256(config) != summary["config_sha256"]:
        raise ValueError(f"{stage} resolved configuration hash mismatch")
    return config


def check_runtime(summary, dependencies, stage):
    recorded = summary.get("runtime")
    if not isinstance(recorded, dict) or set(dependencies) - recorded.keys():
        raise ValueError(f"{stage} runtime identity is missing; use --overwrite")
    for name in dependencies:
        if importlib.metadata.version(name) != recorded[name]:
            raise ValueError(f"{stage} runtime changed: {name}; use --overwrite")


def detection_inputs(run: Path, *, expected_config=None, source_sha256=None):
    summary = read_json(run / "detect.json")
    config = checked_config(run, "detect", summary)
    if expected_config is not None and config != expected_config.model_dump(
        mode="json"
    ):
        raise ValueError("Detection cache configuration changed; use --overwrite")
    if source_sha256 is not None and source_sha256 != summary["source_sha256"]:
        raise ValueError(
            "Detection cache belongs to a different source; use --overwrite"
        )
    if hashes(run, DETECTION_FILES) != summary["artifacts"]:
        raise ValueError("Detection artifact hash mismatch; use --overwrite")
    check_runtime(summary, detection_runtime(summary.get("device")), "Detection")
    expected = ArtifactProvenance(
        source_sha256=summary["source_sha256"],
        config_sha256=summary["config_sha256"],
        upstream_sha256={
            name: summary["assets"]["assets"][name]["sha256"]
            for name in ("ball", "court")
        },
        code_revision=code_identity(),
    )
    # Read each artifact once, never accumulate raw video images or heatmaps.
    arrays = {
        name: read_npz(run / name, expected=expected)[0] for name in DETECTION_FILES
    }
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
    return summary, frames, track


def camera_inputs(run, detection, frames, *, expected_config=None):
    summary = read_json(run / "calibrate.json")
    config = checked_config(run, "calibrate", summary)
    check_runtime(summary, CALIBRATION_RUNTIME, "Calibration")
    if expected_config is not None and config != expected_config.model_dump(
        mode="json"
    ):
        raise ValueError("Calibration cache configuration changed; use --overwrite")
    if file_sha256(run / "cameras.npz") != summary["cameras_sha256"]:
        raise ValueError("Camera artifact hash mismatch")
    expected = ArtifactProvenance(
        source_sha256=detection["source_sha256"],
        config_sha256=summary["config_sha256"],
        upstream_sha256=hashes(run, ("frames.npz", "court.npz", "detect.json")),
        code_revision=code_identity(),
    )
    arrays, _ = read_npz(run / "cameras.npz", expected=expected)
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
    return summary, arrays


def source_record(detection, frames):
    if not len(frames["t_s"]):
        raise ValueError(
            "Empty decode has no known image dimensions/time base; see detect.json"
        )
    return Source(
        name=Path(detection["source"]).name,
        sha256=detection["source_sha256"],
        width=int(frames["image_size"][0]),
        height=int(frames["image_size"][1]),
        frame_count=len(frames["t_s"]),
        time_origin_s=float(frames["time_origin_s"]),
        time_base=frames["time_base"][0].tolist(),
    )
