"""Audit detector continuity and uncertain cameras on development recordings.

Run with ``python -m tennis3d.evaluation.development --help``. This reads numeric
artifacts only. Detector predictions are never localization or camera truth.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tennis3d.artifacts import file_sha256, json_sha256, read_json, read_npz, write_json
from tennis3d.court.model import court_landmarks
from tennis3d.evaluation.runs import (
    ALL_MISSING_WINDOW_FRAMES,
    all_missing_windows,
    run_report,
)
from tennis3d.schemas import Observation

STATES: tuple[Observation, ...] = ("missing", "detected", "ambiguous", "duplicate")
DURATION_KEYS = ("visible_run_lengths_s", "missing_run_lengths_s")
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def distribution(values, weights=None):
    """Empirical inverse-CDF quantiles, with explicit optional sample weights."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("distribution samples must be finite and one-dimensional")
    if not len(values):
        return {"count": 0, "mean": None, "min": None, "max": None, "quantiles": None}
    weights = np.ones(len(values)) if weights is None else np.asarray(weights, float)
    if (
        weights.shape != values.shape
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise ValueError("one finite positive weight required per sample")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order]) / weights.sum()
    quantiles = values[order][
        np.minimum(np.searchsorted(cumulative, QUANTILES), len(values) - 1)
    ]
    return {
        "count": len(values),
        "mean": float(np.average(values, weights=weights)),
        "min": float(values.min()),
        "max": float(values.max()),
        "quantiles": dict(zip(map(str, QUANTILES), map(float, quantiles))),
    }


def _check_hash(path, expected):
    if file_sha256(path) != expected:
        raise ValueError(f"Artifact hash mismatch: {path}")


def _camera_report(cameras, court, times, size, prefix):
    valid = cameras["raw_valid" if prefix else "camera_valid"]
    n = len(times)
    if valid.shape != (n,) or valid.dtype != np.bool_:
        raise ValueError("invalid camera mask")
    matrices = [cameras[prefix + key] for key in ("K", "R", "t")]
    for array, shape in zip(matrices, ((n, 3, 3), (n, 3, 3), (n, 3))):
        if array.shape != shape or not np.isfinite(array[valid]).all():
            raise ValueError("invalid accepted camera matrices")
    K, R, t = matrices
    centers = -np.einsum("nji,nj->ni", R[valid], t[valid])
    focal = K[valid, 0, 0] / size[0]
    xyz = np.einsum("nij,kj->nki", R[valid], court_landmarks()) + t[valid, None, :]
    pixels = np.einsum("nij,nkj->nki", K[valid], xyz)
    uv = pixels[..., :2] / pixels[..., 2:]
    # Outer doubles quadrilateral, in perimeter order; projected area can exceed image.
    corners = uv[:, [0, 1, 3, 2]] / size
    area = (
        abs(
            np.sum(
                corners[..., 0] * np.roll(corners[..., 1], -1, axis=1)
                - corners[..., 1] * np.roll(corners[..., 0], -1, axis=1),
                axis=1,
            )
        )
        / 2
    )
    observed = court["observed"][valid]
    residual = np.linalg.norm((uv - court["raw_uv"][valid]) / size, axis=-1)
    inliers = cameras["inliers"][valid] & observed
    indices = np.flatnonzero(valid)
    adjacent = np.flatnonzero(np.diff(indices) == 1)
    dt = np.diff(times[indices])[adjacent]
    translation = np.linalg.norm(np.diff(centers, axis=0)[adjacent], axis=1) / dt
    relative = np.einsum("nij,nkj->nik", R[indices[adjacent + 1]], R[indices[adjacent]])
    angle = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1))
    return {
        "valid_frames": int(valid.sum()),
        "total_frames": n,
        "valid_fraction": float(valid.mean()) if n else 0,
        "focal_over_width": distribution(focal),
        "center_m": {axis: distribution(centers[:, i]) for i, axis in enumerate("xyz")},
        "projected_court_area_over_image": distribution(area),
        "residual_normalized_all_observed": distribution(residual[observed]),
        "residual_normalized_inliers": distribution(residual[inliers]),
        "adjacent_valid_intervals": len(adjacent),
        "center_speed_m_s": distribution(translation),
        "rotation_speed_deg_s": distribution(np.rad2deg(angle) / dt),
        "relative_focal_speed_s": distribution(
            abs(np.diff(np.log(focal)))[adjacent] / dt
        ),
    }


def summarize_clip(run_path, clip_identity):
    """Verify hashes/envelopes and retain every selected timestamp and failure."""
    run_path = Path(run_path)
    detection = read_json(run_path / "detect.json")
    arrays, provenance = {}, None
    for name in ("frames.npz", "ball_track.npz", "court.npz", "ball_candidates.npz"):
        _check_hash(run_path / name, detection["artifacts"][name])
        arrays[name], envelope = read_npz(run_path / name, expected=provenance)
        provenance = envelope
    if provenance.source_sha256 != clip_identity["clip_sha256"]:
        raise ValueError("clip source hash differs from artifact")
    if (
        json_sha256(read_json(run_path / "detect.config.json"))
        != provenance.config_sha256
    ):
        raise ValueError("detector configuration hash differs")
    frames, ball, court = (
        arrays[name] for name in ("frames.npz", "ball_track.npz", "court.npz")
    )
    times, codes = frames["t_s"], ball["observation"]
    if (
        codes.shape != times.shape
        or codes.dtype.kind not in "iu"
        or np.any((codes < 0) | (codes > 3))
    ):
        raise ValueError("invalid observation codes")
    if len(times) > clip_identity["frame_count"]:
        raise ValueError("decode exceeds selected source denominator")
    incomplete = len(times) != clip_identity["frame_count"]
    mapping_hash = None
    if "source_pts_mapping" in clip_identity:
        mapping_path = Path(clip_identity["source_pts_mapping"])
        mapping = read_json(mapping_path)
        if mapping["clip_sha256"] != provenance.source_sha256:
            raise ValueError("timestamp mapping source differs")
        rows = mapping["frames"][: len(times)]
        if not np.array_equal(
            frames["pts"], [r["clip_pts"] for r in rows]
        ) or not np.array_equal(
            frames["time_base"], [r["clip_time_base"] for r in rows]
        ):
            raise ValueError("source PTS mapping differs")
        mapping_hash = file_sha256(mapping_path)
    states = [STATES[int(code)] for code in codes]
    report = run_report(times, states)
    for row in report["runs"]:
        row["left_censored"] = row["start_frame"] == 0
        row["right_censored"] = row["stop_frame"] == len(times)
    missing_s = sum(row["duration_s"] for row in report["runs"] if not row["visible"])
    report.update(
        identity=dict(clip_identity),
        selected_frame_count=clip_identity["frame_count"],
        incomplete_decode=incomplete,
        source_pts_mapping_sha256=mapping_hash,
        detection=detection,
        artifact_provenance=provenance.model_dump(),
        artifact_hashes=detection["artifacts"],
        missing_time_s=missing_s,
        missing_time_fraction=missing_s / report["duration_s"]
        if report["duration_s"]
        else None,
        missing_frame_fraction=float(np.mean(codes != 1)) if len(codes) else None,
        all_missing=not bool(np.any(codes == 1)),
        all_visible=bool(len(codes) and np.all(codes == 1)),
        initial_missing=bool(len(codes) and codes[0] != 1),
        initial_missing_duration_s=report["runs"][0]["duration_s"]
        if len(codes) and codes[0] != 1
        else 0,
        warmup_states=states[:2],
        cadence_s=distribution(np.diff(times)),
        duplicate_frame_count=int(np.sum(frames["duplicate_of"] >= 0)),
        shot_cut_frames=np.flatnonzero(frames["cut"]).tolist(),
    )
    report["all_missing_windows"] = {
        str(width): dict(
            zip(("total", "all_missing"), all_missing_windows(codes == 1, width))
        )
        for width in ALL_MISSING_WINDOW_FRAMES
    }
    report["empirical_ready"] = report["empirical_ready"] and not incomplete
    calibration_path = run_path / "calibrate.json"
    if not calibration_path.exists():
        report["calibration"] = {"status": "absent"}
        return report
    calibration = read_json(calibration_path)
    _check_hash(run_path / "cameras.npz", calibration["cameras_sha256"])
    cameras, camera_provenance = read_npz(run_path / "cameras.npz")
    if (
        json_sha256(read_json(run_path / "calibrate.config.json"))
        != camera_provenance.config_sha256
    ):
        raise ValueError("camera configuration hash differs")
    if camera_provenance.source_sha256 != provenance.source_sha256:
        raise ValueError("camera source differs")
    for name in ("frames.npz", "court.npz", "detect.json"):
        _check_hash(run_path / name, camera_provenance.upstream_sha256[name])
    if not np.array_equal(cameras["t_s"], times):
        raise ValueError("camera timestamps differ")
    report["calibration"] = calibration
    report["camera_artifact_provenance"] = camera_provenance.model_dump()
    report["camera"] = {
        name: _camera_report(cameras, court, times, frames["image_size"], prefix)
        for name, prefix in (("raw", "raw_"), ("filtered", ""))
    }
    report["camera"]["filter_reason_counts"] = dict(
        Counter(calibration["reason_codes"][int(x)] for x in cameras["filter_reason"])
    )
    return report


def _aggregate(reports):
    grouped = defaultdict(list)
    for report in reports:
        grouped[report["identity"]["recording_group"]].append(report)
    result: dict[str, Any] = {
        "clip_count": len(reports),
        "recording_group_count": len(grouped),
        "frame_count": sum(r["frame_count"] for r in reports),
        "selected_frame_count": sum(
            r.get("selected_frame_count", r["frame_count"]) for r in reports
        ),
        "incomplete_decode_clip_ids": [
            r["identity"]["clip_id"] for r in reports if r.get("incomplete_decode")
        ],
        "duration_s": sum(r["duration_s"] for r in reports),
        "all_missing_clips": [
            r["identity"]["clip_id"] for r in reports if r["all_missing"]
        ],
        "unfit_clip_ids": [
            r["identity"]["clip_id"] for r in reports if not r["empirical_ready"]
        ],
    }
    result["pooled_missing_time_fraction"] = (
        sum(r["missing_time_s"] for r in reports) / result["duration_s"]
        if result["duration_s"]
        else None
    )
    group_fractions = [
        sum(r["missing_time_s"] for r in rows) / sum(r["duration_s"] for r in rows)
        for rows in grouped.values()
        if sum(r["duration_s"] for r in rows)
    ]
    result["recording_balanced_missing_time_fraction"] = (
        float(np.mean(group_fractions)) if group_fractions else None
    )
    result["initial_missing_clip_fraction"] = (
        sum(r["initial_missing"] for r in reports) / len(reports) if reports else None
    )
    result["initial_missing_duration_s"] = distribution(
        [r["initial_missing_duration_s"] for r in reports if r["initial_missing"]]
    )
    result["empirical_corruption"], result["durations"] = {}, {}
    for key in DURATION_KEYS:
        samples, weights, omitted = [], [], []
        for group, rows in sorted(grouped.items()):
            values = [v for r in rows for v in r["empirical_corruption"][key]]
            if not values:
                omitted.append(group)
                continue
            samples.extend(values)
            weights.extend([1 / len(values)] * len(values))
        result["empirical_corruption"][key] = samples
        result["empirical_corruption"][key.replace("lengths_s", "weights")] = (
            (np.asarray(weights) / sum(weights)).tolist() if weights else []
        )
        result["durations"][key] = {
            "pooled": distribution(samples),
            "recording_balanced": distribution(samples, weights),
            "groups_without_samples": omitted,
        }
    result["empirical_ready"] = (
        all(result["empirical_corruption"][key] for key in DURATION_KEYS)
        and not result["incomplete_decode_clip_ids"]
    )
    result["pooling_rule"] = (
        "Each recording contributes equal weight within each positive-duration state sample set; groups without that state have no conditional samples and are reported. Missing-time fractions retain all groups. Boundary durations are retained as censored observations."
    )
    return result


def aggregate_reports(reports):
    """Pooled/group-balanced durations and leave-one-recording-group-out results."""
    result = _aggregate(reports)
    groups = sorted({r["identity"]["recording_group"] for r in reports})
    result["leave_one_recording_out"] = {
        group: _aggregate(
            [r for r in reports if r["identity"]["recording_group"] != group]
        )
        for group in groups
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-provenance", type=Path, help="Frozen inference source sidecar"
    )
    args = parser.parse_args(argv)
    inventory = read_json(args.inventory)
    reports = [
        summarize_clip(args.runs_root / clip["clip_id"], clip)
        for clip in inventory["clips"]
    ]
    write_json(
        args.output,
        {
            "inventory_sha256": file_sha256(args.inventory),
            "utility_sha256": file_sha256(__file__),
            "source_provenance": read_json(args.source_provenance)
            if args.source_provenance
            else None,
            "source_provenance_sha256": file_sha256(args.source_provenance)
            if args.source_provenance
            else None,
            "clips": reports,
            "aggregate": aggregate_reports(reports),
            "limitations": "Convenience development recordings; uploaded-source PTS, unknown original exposure timing and checkpoint overlap. No localization labels or camera truth.",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
