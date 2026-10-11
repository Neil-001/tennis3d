"""Development diversity diagnostics for an existing synthetic dataset."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from heapq import nsmallest
from itertools import pairwise
from pathlib import Path

import numpy as np

from tennis3d.artifacts import file_sha256, read_json, read_npz, write_json
from tennis3d.evaluation.runs import (
    ALL_MISSING_WINDOW_FRAMES,
    all_missing_windows,
    run_report,
)
from tennis3d.inference.windows import make_windows
from tennis3d.synthetic import KINDS
from tennis3d.synthetic.data import load_view, read_manifest

PAIR_TRACKING_LIMIT = 100_000
PAIR_DETAIL_LIMIT = 100


@dataclass
class ScalarStats:
    """Constant-memory population moments."""

    count: int = 0
    total: float = 0.0
    total_squares: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, values) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if not len(array):
            return
        if not np.isfinite(array).all():
            raise ValueError("diversity statistics require finite values")
        self.count += len(array)
        self.total += float(array.sum(dtype=np.float64))
        self.total_squares += float(np.square(array).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(array.min()))
        self.maximum = max(self.maximum, float(array.max()))

    def result(self) -> dict:
        if not self.count:
            return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
        mean = self.total / self.count
        variance = max(0.0, self.total_squares / self.count - mean * mean)
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": mean,
            "std": math.sqrt(variance),
        }


@dataclass
class VectorStats:
    axes: tuple[ScalarStats, ScalarStats, ScalarStats] = field(
        default_factory=lambda: (ScalarStats(), ScalarStats(), ScalarStats())
    )

    def add(self, values) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1:] != (3,):
            raise ValueError("vector diversity statistics require XYZ triples")
        for index, accumulator in enumerate(self.axes):
            accumulator.add(array[..., index])

    def result(self) -> dict:
        return {
            axis: accumulator.result()
            for axis, accumulator in zip(("x", "y", "z"), self.axes)
        }


@dataclass
class SegmentStats:
    count: int = 0
    initial_velocity_m_s: VectorStats = field(default_factory=VectorStats)
    initial_spin_rad_s: VectorStats = field(default_factory=VectorStats)
    initial_speed_m_s: ScalarStats = field(default_factory=ScalarStats)
    maximum_height_m: ScalarStats = field(default_factory=ScalarStats)
    duration_s: ScalarStats = field(default_factory=ScalarStats)

    def add(self, arrays: dict[str, np.ndarray], segment_id: int) -> None:
        indices = np.flatnonzero(arrays["segment_id"] == segment_id)
        if not len(indices):
            raise ValueError(f"physical segment {segment_id} has no frames")
        first, last = int(indices[0]), int(indices[-1])
        velocity = arrays["velocity_m_s"][first]
        spin = arrays["spin_rad_s"][first]
        self.count += 1
        self.initial_velocity_m_s.add(velocity)
        self.initial_spin_rad_s.add(spin)
        self.initial_speed_m_s.add(np.linalg.norm(velocity))
        self.maximum_height_m.add(np.max(arrays["xyz_m"][indices, 2]))
        self.duration_s.add(arrays["t_s"][last] - arrays["t_s"][first])

    def result(self) -> dict:
        return {
            "count": self.count,
            "initial_velocity_m_s": self.initial_velocity_m_s.result(),
            "initial_spin_rad_s": self.initial_spin_rad_s.result(),
            "initial_speed_m_s": self.initial_speed_m_s.result(),
            "maximum_height_m": self.maximum_height_m.result(),
            "sampled_duration_s": self.duration_s.result(),
        }


@dataclass
class SplitAudit:
    physical_rallies: int = 0
    views: int = 0
    physical_frames: int = 0
    view_frames: int = 0
    valid_label_rows: int = 0
    shot_counts: Counter = field(default_factory=Counter)
    fps_counts: Counter = field(default_factory=Counter)
    assembly_attempt_counts: Counter = field(default_factory=Counter)
    adjacent_prior_pairs: Counter = field(default_factory=Counter)
    evaluation_windows: dict[str, Counter] = field(
        default_factory=lambda: {mode: Counter() for mode in ("clean", "corrupted")}
    )
    evaluation_failures: list[dict] = field(default_factory=list)
    views_without_supported_labels: int = 0
    segments_per_rally: ScalarStats = field(default_factory=ScalarStats)
    rally_duration_s: ScalarStats = field(default_factory=ScalarStats)
    frame_interval_s: ScalarStats = field(default_factory=ScalarStats)
    segment_stats: dict[str, SegmentStats] = field(
        default_factory=lambda: {kind: SegmentStats() for kind in KINDS}
    )
    camera_center_frames_m: VectorStats = field(default_factory=VectorStats)
    camera_center_mean_per_view_m: VectorStats = field(default_factory=VectorStats)
    focal_over_width_frames: ScalarStats = field(default_factory=ScalarStats)
    focal_over_width_mean_per_view: ScalarStats = field(default_factory=ScalarStats)
    visibility_fraction_per_view: ScalarStats = field(default_factory=ScalarStats)
    in_frame_fraction_per_view: ScalarStats = field(default_factory=ScalarStats)
    camera_valid_fraction_per_view: ScalarStats = field(default_factory=ScalarStats)
    observation_counts: Counter = field(default_factory=Counter)
    visible_run_s: ScalarStats = field(default_factory=ScalarStats)
    missing_run_s: ScalarStats = field(default_factory=ScalarStats)
    zero_duration_runs: int = 0
    windows: dict[int, Counter] = field(
        default_factory=lambda: {
            width: Counter() for width in ALL_MISSING_WINDOW_FRAMES
        }
    )


def _checked(path: Path, expected: str) -> Path:
    if file_sha256(path) != expected:
        raise ValueError(f"Artifact checksum mismatch: {path}")
    return path


def _pool_rows(pool: dict) -> int:
    size = pool.get("config", {}).get("size")
    if not isinstance(size, int) or size <= 0:
        raise ValueError(f"pool {pool.get('family_id')} has no positive config.size")
    return size


def _pool_report(pools: list[dict]) -> list[dict]:
    rows = []
    for pool in pools:
        accepted = _pool_rows(pool)
        attempts = pool.get("attempts", {})
        for kind in KINDS:
            proposed = attempts.get(kind)
            rows.append(
                {
                    "split": pool["split"],
                    "family_id": pool["family_id"],
                    "kind": kind,
                    "accepted": accepted,
                    "proposal_attempts": proposed,
                    "acceptance_fraction": accepted / proposed
                    if isinstance(proposed, int) and proposed > 0
                    else None,
                }
            )
    return rows


def _physical_arrays(path: Path, expected: str) -> dict[str, np.ndarray]:
    arrays, _ = read_npz(_checked(path, expected))
    required = {
        "t_s",
        "xyz_m",
        "velocity_m_s",
        "spin_rad_s",
        "label_valid",
        "segment_id",
        "segment_kind",
        "prior_id",
    }
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"physical rally is missing arrays: {sorted(missing)}")
    segment_count = len(arrays["segment_kind"])
    if arrays["prior_id"].shape != (segment_count,):
        raise ValueError("physical segment and prior tables differ in length")
    if not np.array_equal(np.unique(arrays["segment_id"]), np.arange(segment_count)):
        raise ValueError("physical segment ids must be dense and zero based")
    return arrays


def _record_physical(
    result: SplitAudit,
    arrays: dict[str, np.ndarray],
    record: dict,
    family_uses: dict[tuple[str, str, int], int],
) -> None:
    result.physical_rallies += 1
    result.physical_frames += len(arrays["t_s"])
    result.valid_label_rows += int(np.count_nonzero(arrays["label_valid"]))
    result.fps_counts[int(record["fps"])] += 1
    result.assembly_attempt_counts[int(record["attempts"])] += 1
    result.segments_per_rally.add(len(arrays["segment_kind"]))
    result.rally_duration_s.add(arrays["t_s"][-1] - arrays["t_s"][0])
    result.frame_interval_s.add(np.diff(arrays["t_s"]))
    sequence = list(zip(arrays["segment_kind"].tolist(), arrays["prior_id"].tolist()))
    for before, after in pairwise(sequence):
        key = (record["family_id"], *before, *after)
        if key not in result.adjacent_prior_pairs and (
            len(result.adjacent_prior_pairs) >= PAIR_TRACKING_LIMIT
        ):
            raise ValueError(
                f"development audit exceeds {PAIR_TRACKING_LIMIT} distinct adjacent "
                "pairs per split; use a partitioned main-scale audit"
            )
        result.adjacent_prior_pairs[key] += 1
    for segment_id, (kind_index, prior_id) in enumerate(
        zip(arrays["segment_kind"], arrays["prior_id"])
    ):
        kind_number = int(kind_index)
        if not 0 <= kind_number < len(KINDS):
            raise ValueError(f"unknown physical segment kind {kind_number}")
        kind = KINDS[kind_number]
        prior = int(prior_id)
        if prior < 0:
            raise ValueError(f"negative prior id {prior}")
        result.shot_counts[kind] += 1
        family_uses[(record["family_id"], kind, prior)] += 1
        result.segment_stats[kind].add(arrays, segment_id)


def _record_view(
    result: SplitAudit, arrays: dict[str, np.ndarray], record: dict
) -> None:
    observed = arrays["ball_observed"]
    in_frame = arrays["ball_in_frame"]
    valid_camera = arrays["camera_valid"]
    n = len(observed)
    result.views += 1
    result.view_frames += n
    result.visibility_fraction_per_view.add(np.mean(observed))
    result.in_frame_fraction_per_view.add(np.mean(in_frame))
    result.camera_valid_fraction_per_view.add(np.mean(valid_camera))
    result.views_without_supported_labels += int(
        not (valid_camera & arrays["label_valid"]).any()
    )
    for mode, mask in (("clean", in_frame), ("corrupted", observed)):
        for window in make_windows(valid_camera):
            missing = not mask[window.start : window.stop].any()
            result.evaluation_windows[mode].update(total=1, all_missing=int(missing))
            if missing:
                result.evaluation_failures.append(
                    {
                        "rally_id": record["rally_id"],
                        "view_id": record["view_id"],
                        "mode": mode,
                        "start": window.start,
                        "stop": window.stop,
                    }
                )

    centers = -np.einsum(
        "nji,nj->ni", arrays["camera_R"][valid_camera], arrays["camera_t"][valid_camera]
    )
    ratios = arrays["camera_K"][valid_camera, 0, 0] / arrays["image_size"][0]
    result.camera_center_frames_m.add(centers)
    result.focal_over_width_frames.add(ratios)
    if len(ratios):
        result.camera_center_mean_per_view_m.add(centers.mean(axis=0))
        result.focal_over_width_mean_per_view.add(ratios.mean())

    durations = run_report(
        arrays["t_s"], ["detected" if value else "missing" for value in observed]
    )
    result.observation_counts.update(durations["observation_counts"])
    result.visible_run_s.add(durations["empirical_corruption"]["visible_run_lengths_s"])
    result.missing_run_s.add(durations["empirical_corruption"]["missing_run_lengths_s"])
    result.zero_duration_runs += sum(
        run["duration_s"] == 0 for run in durations["runs"]
    )
    for width in ALL_MISSING_WINDOW_FRAMES:
        total, missing = all_missing_windows(observed, width)
        result.windows[width].update(total=total, all_missing=missing)


def _split_result(
    result: SplitAudit,
    family_uses: dict[tuple[str, str, int], int],
    pools: dict[str, dict],
    split: str,
) -> dict:
    repeated_pair_count = sum(
        count > 1 for count in result.adjacent_prior_pairs.values()
    )
    repeated_pairs = nsmallest(
        PAIR_DETAIL_LIMIT,
        (
            (key, count)
            for key, count in result.adjacent_prior_pairs.items()
            if count > 1
        ),
        key=lambda item: (-item[1], item[0]),
    )
    occupancy = []
    reused = []
    grouped: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for (family, kind, prior), count in family_uses.items():
        grouped[(family, kind)][prior] = count
        if count > 1:
            reused.append(
                {
                    "family_id": family,
                    "kind": kind,
                    "prior_id": prior,
                    "uses": count,
                    "reuses_after_first": count - 1,
                }
            )
    groups = {
        (family, kind)
        for family, pool in pools.items()
        if pool["split"] == split
        for kind in KINDS
    }
    groups.update(grouped)
    for family, kind in sorted(groups):
        counts = grouped[(family, kind)]
        pool_rows = _pool_rows(pools[family])
        if counts and max(counts) >= pool_rows:
            raise ValueError(
                f"{family}/{kind} uses prior id {max(counts)} outside {pool_rows} rows"
            )
        occupancy.append(
            {
                "family_id": family,
                "kind": kind,
                "pool_rows": pool_rows,
                "used_rows": len(counts),
                "occupancy_fraction": len(counts) / pool_rows,
                "segment_uses": sum(counts.values()),
                "prior_use_counts": {
                    str(prior): count for prior, count in sorted(counts.items())
                },
            }
        )
    windows = {}
    for width, counts in result.windows.items():
        total = counts["total"]
        missing = counts["all_missing"]
        windows[str(width)] = {
            "total_windows": total,
            "all_missing_windows": missing,
            "all_missing_fraction": missing / total if total else None,
        }
    return {
        "physical_rallies": result.physical_rallies,
        "views": result.views,
        "physical_frames": result.physical_frames,
        "view_frames": result.view_frames,
        "valid_label_rows": result.valid_label_rows,
        "shot_counts": {kind: result.shot_counts[kind] for kind in KINDS},
        "cadence": {
            "fps_rally_counts": {
                str(key): value for key, value in sorted(result.fps_counts.items())
            },
            "segments_per_rally": result.segments_per_rally.result(),
            "rally_duration_s": result.rally_duration_s.result(),
            "physical_frame_interval_s": result.frame_interval_s.result(),
        },
        "assembly_attempts": {
            "rally_counts": {
                str(key): value
                for key, value in sorted(result.assembly_attempt_counts.items())
            },
            "total_attempts": sum(
                attempts * count
                for attempts, count in result.assembly_attempt_counts.items()
            ),
        },
        "prior_occupancy": occupancy,
        "adjacent_prior_pairs": {
            "total": sum(result.adjacent_prior_pairs.values()),
            "distinct": len(result.adjacent_prior_pairs),
            "reuses_after_first": sum(
                count - 1 for count in result.adjacent_prior_pairs.values()
            ),
            "tracking_limit": PAIR_TRACKING_LIMIT,
            "repeated_identity_count": repeated_pair_count,
            "detail_limit": PAIR_DETAIL_LIMIT,
            "details_truncated": repeated_pair_count > PAIR_DETAIL_LIMIT,
            "repeated": [
                {
                    "family_id": key[0],
                    "from_kind": KINDS[key[1]],
                    "from_prior_id": key[2],
                    "to_kind": KINDS[key[3]],
                    "to_prior_id": key[4],
                    "uses": count,
                }
                for key, count in repeated_pairs
            ],
        },
        "reused_transitions": sorted(
            reused, key=lambda row: (row["family_id"], row["kind"], row["prior_id"])
        ),
        "physical_segment_distributions": {
            kind: result.segment_stats[kind].result() for kind in KINDS
        },
        "view_distributions": {
            "camera_center_m_all_frames": result.camera_center_frames_m.result(),
            "camera_center_mean_m_per_view": result.camera_center_mean_per_view_m.result(),
            "focal_over_width_all_frames": result.focal_over_width_frames.result(),
            "focal_over_width_mean_per_view": result.focal_over_width_mean_per_view.result(),
            "visibility_fraction_per_view": result.visibility_fraction_per_view.result(),
            "in_frame_fraction_per_view": result.in_frame_fraction_per_view.result(),
            "camera_valid_fraction_per_view": result.camera_valid_fraction_per_view.result(),
        },
        "observation_runs": {
            "method": "tennis3d.evaluation.runs.run_report",
            "observation_counts": dict(sorted(result.observation_counts.items())),
            "visible_duration_s": result.visible_run_s.result(),
            "missing_duration_s": result.missing_run_s.result(),
            "zero_duration_boundary_runs": result.zero_duration_runs,
        },
        "all_missing_windows": windows,
        "evaluation_windows": {
            "method": "make_windows(camera_valid, max_frames=250, stride=125)",
            "modes": {
                mode: dict(counts) for mode, counts in result.evaluation_windows.items()
            },
            "views_without_supported_labels": result.views_without_supported_labels,
            "failures": result.evaluation_failures,
        },
    }


def audit_dataset(root: str | Path) -> dict:
    """Read one numeric rally/view at a time; retain manifests and capped pair counts."""
    root = Path(root)
    checksums = read_json(root / "manifest.sha256.json")
    manifest_path = _checked(root / "manifest.json", checksums["manifest.json"])
    manifest = read_json(manifest_path)
    pools_by_family = {pool["family_id"]: pool for pool in manifest["pools"]}
    split_reports = {}
    seen_physical: set[tuple[str, str]] = set()
    physical_owners: dict[str, str] = {}
    for split in ("train", "val", "test"):
        entry = manifest["splits"][split]
        split_path = _checked(root / entry["path"], entry["sha256"])
        split_manifest = read_manifest(split_path)
        accumulator = SplitAudit()
        family_uses: dict[tuple[str, str, int], int] = defaultdict(int)
        for record in split_manifest["records"]:
            family = record["family_id"]
            if family not in pools_by_family:
                raise ValueError(f"record refers to unknown pool family {family}")
            physical_identity = (record["rally_path"], record["rally_sha256"])
            if physical_owners.setdefault(record["rally_sha256"], split) != split:
                raise ValueError("physical rally artifact is reused across splits")
            if physical_identity not in seen_physical:
                arrays = _physical_arrays(
                    split_path.parent / record["rally_path"], record["rally_sha256"]
                )
                _record_physical(accumulator, arrays, record, family_uses)
                seen_physical.add(physical_identity)
            _record_view(
                accumulator,
                load_view(split_path, record, manifest=split_manifest),
                record,
            )
        split_reports[split] = _split_result(
            accumulator, family_uses, pools_by_family, split
        )
    return {
        "schema_version": "1.0",
        "source_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": checksums["manifest.json"],
            "generator_sha256": manifest.get("generator_sha256"),
            "config_sha256": manifest.get("config_sha256"),
        },
        "window_widths_frames": list(ALL_MISSING_WINDOW_FRAMES),
        "pool_acceptance": _pool_report(manifest["pools"]),
        "stitch_counters": {
            "available": False,
            "reason": "the generated manifest does not record in-memory stitch counters",
        },
        "splits": split_reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="generated synthetic dataset root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = audit_dataset(args.root)
        write_json(args.output, report)
    except (KeyError, OSError, ValueError) as error:
        parser.exit(2, f"Synthetic diversity audit failed: {error}\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "splits": {
                    split: {
                        key: report["splits"][split][key]
                        for key in ("physical_rallies", "views")
                    }
                    for split in ("train", "val", "test")
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
