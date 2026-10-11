"""Paired summary of finished real evaluations at two resolutions.

Each low-resolution clip is a copy of one full-resolution clip, frame for
frame. This script joins the finished evaluation roots of several models
through the frozen pairing map and counts, for every pair, how observation,
camera validity, raw coverage and exported XYZ changed on the same source
frames, and how status and physical-fit counts changed. It computes nothing
from XYZ values: no real XYZ truth exists, so neither cohort is treated as the
more accurate one. The cohorts are never pooled. Nothing is rerun or modified.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from tennis3d.artifacts import file_sha256, read_json, read_npz, write_json
from tennis3d.evaluation.delivery import fresh_output
from tennis3d.evaluation.inventory import read_inventory
from tennis3d.schemas import read_result

COHORTS = ("full", "low")
FLAGS = ("observed", "camera_valid", "raw_covered", "exported_xyz")
FIT_COUNTS = (
    "candidate_segments",
    "fitted_segments",
    "excluded_segments",
    "failed_segments",
    "eligible_frames",
    "contact_guard_frames",
    "scored_frames",
)
LIMITS = [
    "Each pair holds the same source frames at two resolutions; the low-resolution clip is a derived copy, not separately recorded footage.",
    "No real XYZ, ball UV or camera truth exists. A change in coverage or fit counts is not a change in accuracy.",
    "Physical-fit counts measure consistency with each predicted curve, on the time basis recorded in each report.",
    "Detection and calibration ran once per cohort and are shared by all models; differences between models within a cohort come from the lifter alone.",
]


class PairingError(Exception):
    """The inputs do not form the frozen pairs; nothing is written."""


def pinned(path, expected, name):
    if file_sha256(path) != expected:
        raise PairingError(f"{name} differs from its pinned SHA-256")
    return Path(path).resolve()


def clip_rows(gallery, clip, record, checkpoint, bound):
    """Per-frame flags of one clip of one model, or None when it has no result."""
    summary = {
        "clip_id": clip.clip_id,
        "status": record["status"],
        "failure_reasons": record.get("failure_reasons", []),
        "frame_count": clip.frame_count,
    }
    if (bound["status"], bool(bound["result"])) != (
        record["status"],
        bool(record.get("result_path")),
    ):
        raise PairingError(f"{clip.clip_id}: record differs from the release binding")
    if not record.get("result_path"):
        return summary, None
    run = gallery / "clips" / clip.clip_id
    pipeline = record["pipeline"]
    # Every released file of the clip still has its bound bytes.
    released = {"result.json": bound["result"]} | bound["artifacts"]
    for name, entry in released.items():
        if file_sha256(run / name) != entry["sha256"]:
            raise PairingError(
                f"{clip.clip_id}: {name} differs from the release binding"
            )
    for name in ("result.json", "lift.json", "detect.json", "calibrate.json"):
        if file_sha256(run / name) != pipeline["artifacts"][name]:
            raise PairingError(f"{clip.clip_id}: {name} differs from its run summary")
    lifting = read_json(run / "lift.json")
    if (
        lifting["checkpoint_sha256"] != checkpoint
        or file_sha256(run / "predictions_raw.npz") != lifting["predictions_sha256"]
    ):
        raise PairingError(f"{clip.clip_id}: predictions are not this checkpoint's")
    result = read_result(run / "result.json")
    if not (result.status == record["status"] == pipeline["status"]):
        raise PairingError(f"{clip.clip_id}: record, run summary and result disagree")
    # The stage files are those the run summary and detect.json recorded.
    detection = read_json(run / "detect.json")
    recorded = pipeline.get("cached_inputs", {}).get("upstream_sha256", {})
    for name, digest in (detection["artifacts"] | recorded).items():
        if file_sha256(run / name) != digest:
            raise PairingError(f"{clip.clip_id}: {name} differs from its recorded hash")
    raw, _ = read_npz(run / "predictions_raw.npz")
    frames, _ = read_npz(run / "frames.npz")
    if file_sha256(clip.timeline_path) != clip.timeline_sha256:
        raise PairingError(f"{clip.clip_id}: frozen timeline changed")
    timeline = read_json(clip.timeline_path)["frames"]
    if (
        result.source.sha256 != clip.source_sha256
        or detection["source_sha256"] != clip.source_sha256
        or len(result.frames) != clip.frame_count
        or len(raw["covered"]) != clip.frame_count
        or len(timeline) != clip.frame_count
        or not np.array_equal(frames["pts"], [row["pts"] for row in timeline])
        or not np.array_equal(
            frames["time_base"], [row["time_base"] for row in timeline]
        )
    ):
        raise PairingError(f"{clip.clip_id}: result does not match the frozen clip")
    rows = {
        "pts": frames["pts"],
        "time_base": frames["time_base"],
        "observed": np.array([f.observation == "detected" for f in result.frames]),
        "camera_valid": np.array([f.camera_valid for f in result.frames]),
        "raw_covered": raw["covered"].astype(bool),
        "exported_xyz": np.array([f.xyz_m is not None for f in result.frames]),
    }
    summary |= {
        "result_sha256": pipeline["artifacts"]["result.json"],
        "upstream_sha256": pipeline.get("cached_inputs", {}).get("upstream_sha256"),
        "rows": {name: int(rows[name].sum()) for name in FLAGS},
        "observation": dict(
            sorted(Counter(f.observation for f in result.frames).items())
        ),
    }
    fit = record.get("physical_path")
    if fit:
        report = read_json(gallery / fit)
        metadata, counts = report["metadata"], report["counts"]
        states = Counter(segment["status"] for segment in report["segments"])
        if (
            "physical.json" not in released
            or set(states) - {"fitted", "excluded", "failed"}
            or any(
                counts[f"{state}_segments"] != states[state]
                for state in ("fitted", "excluded", "failed")
            )
            or fit != f"clips/{clip.clip_id}/physical.json"
            or metadata["artifact_id"] != clip.clip_id
            or metadata["source_identity"]["source_sha256"] != clip.source_sha256
            or metadata["prediction_identity"]["checkpoint_sha256"] != checkpoint
            or counts["source_frames"] != clip.frame_count
            or counts["available_frames"] != summary["rows"]["raw_covered"]
            or counts["candidate_segments"] != len(report["segments"])
        ):
            raise PairingError(f"{clip.clip_id}: physical report is not this clip's")
        summary["physical"] = {
            "time_basis": metadata["time_basis"],
            "segmentation_method": metadata["segmentation_method"],
            "counts": {name: counts[name] for name in FIT_COUNTS},
            "reason_counts": report["reason_counts"],
        }
    return summary, rows


def transitions(full, low):
    """Source frames by their state in the full and the low resolution result."""
    return {
        name: {
            "both": int((full[name] & low[name]).sum()),
            "full_only": int((full[name] & ~low[name]).sum()),
            "low_only": int((~full[name] & low[name]).sum()),
            "neither": int((~full[name] & ~low[name]).sum()),
        }
        for name in FLAGS
    }


def model_pairs(name, roots, inventories, pairs):
    """All frozen pairs of one model; every pair is kept, with or without results."""
    metrics, records, bound, checkpoints = {}, {}, {}, set()
    for cohort in COHORTS:
        gallery = Path(roots[cohort]).resolve()
        metrics[cohort] = read_json(gallery / "metrics.json")
        if (
            metrics[cohort].get("kind") != "end_to_end_real"
            or (gallery / "aborted.json").exists()
            or metrics[cohort]["inventory_sha256"] != inventories[cohort]["sha256"]
        ):
            raise PairingError(f"{name}: {cohort} root is not a finished evaluation")
        checkpoints.add(
            metrics[cohort]["provenance"]["release"]["checkpoint"]["sha256"]
        )
        records[cohort] = {row["clip_id"]: row for row in metrics[cohort]["records"]}
        clips = {clip.clip_id for clip in inventories[cohort]["inventory"].clips}
        if len(records[cohort]) != len(metrics[cohort]["records"]) or (
            set(records[cohort]) != clips
        ):
            raise PairingError(f"{name}: {cohort} root and inventory differ in clips")
        binding = read_json(roots["bindings"][cohort])
        bound[cohort] = {row["clip_id"]: row for row in binding["clips"]}
        if (
            binding["gallery_metrics"]["sha256"]
            != file_sha256(gallery / "metrics.json")
            or binding["inventory"]["sha256"] != inventories[cohort]["sha256"]
            or set(bound[cohort]) != clips
            or len(bound[cohort]) != len(binding["clips"])
        ):
            raise PairingError(f"{name}: {cohort} release binding is for another root")
    if len(checkpoints) != 1:
        raise PairingError(f"{name}: the two roots used different checkpoints")
    checkpoint = checkpoints.pop()
    rows, totals = [], {flag: Counter() for flag in FLAGS}
    for pair in pairs:
        sides, flags = {}, {}
        for cohort, clip in zip(COHORTS, pair, strict=True):
            sides[cohort], flags[cohort] = clip_rows(
                Path(roots[cohort]).resolve(),
                clip,
                records[cohort][clip.clip_id],
                checkpoint,
                bound[cohort][clip.clip_id],
            )
        row = {
            "full_clip_id": pair[0].clip_id,
            "low_clip_id": pair[1].clip_id,
            "source_frames": pair[0].frame_count,
            **sides,
            "status_change": [sides["full"]["status"], sides["low"]["status"]],
            "frames": None,
        }
        if flags["full"] is not None and flags["low"] is not None:
            if not np.array_equal(
                flags["full"]["pts"], flags["low"]["pts"]
            ) or not np.array_equal(
                flags["full"]["time_base"], flags["low"]["time_base"]
            ):
                raise PairingError(f"{pair[1].clip_id}: paired rows differ in PTS")
            row["frames"] = transitions(flags["full"], flags["low"])
            for flag in FLAGS:
                totals[flag].update(row["frames"][flag])
        rows.append(row)
    physical_counts = {
        cohort: {
            key: sum(
                row[cohort]["physical"]["counts"][key]
                for row in rows
                if "physical" in row[cohort]
            )
            for key in FIT_COUNTS
        }
        for cohort in COHORTS
    }
    # The evaluation's own summary covers exactly these per-clip reports.
    for cohort in COHORTS:
        summarized = metrics[cohort].get("physical")
        fitted = [row for row in rows if "physical" in row[cohort]]
        if not fitted and not summarized:
            continue
        cohorts = (summarized or {}).get("cohorts", [])
        if (
            len(cohorts) != 1
            or sorted(cohorts[0]["artifact_ids"])
            != sorted(row[cohort]["clip_id"] for row in fitted)
            or any(
                cohorts[0]["counts"][key] != physical_counts[cohort][key]
                for key in FIT_COUNTS
            )
        ):
            raise PairingError(
                f"{name}: {cohort} physical reports differ from the evaluation summary"
            )
    return {
        "model": name,
        "checkpoint_sha256": checkpoint,
        "roots": {
            cohort: {
                "path": str(Path(roots[cohort]).resolve()),
                "metrics_sha256": file_sha256(Path(roots[cohort]) / "metrics.json"),
                "release_binding": {
                    "path": str(Path(roots["bindings"][cohort]).resolve()),
                    "sha256": file_sha256(roots["bindings"][cohort]),
                },
                "config_sha256": metrics[cohort]["config_sha256"],
                "status_counts": dict(
                    sorted(
                        Counter(r["status"] for r in metrics[cohort]["records"]).items()
                    )
                ),
            }
            for cohort in COHORTS
        },
        "pairs": rows,
        "pairs_with_both_results": sum(row["frames"] is not None for row in rows),
        "source_frames": sum(row["source_frames"] for row in rows),
        "frames_in_pairs_with_both_results": {
            flag: dict(sorted(totals[flag].items())) for flag in FLAGS
        },
        "status_changes": dict(
            sorted(Counter(" -> ".join(row["status_change"]) for row in rows).items())
        ),
        "physical_counts": physical_counts,
    }


def same_stages(sides):
    """Whether every model that has a result for a clip read the same stage bytes.

    A model without a result for the clip records no stages and does not count;
    the pair stays in every model's report either way.
    """
    stages = {
        json.dumps(side["upstream_sha256"], sort_keys=True)
        for side in sides
        if "upstream_sha256" in side
    }
    return len(stages) <= 1


def paired(pairing, inventories, models, output, notes=()):
    """Join every model's two roots through the pairing map and write the summary.

    `inventories` maps each cohort to (path, SHA-256); `pairing` is (path,
    SHA-256); `models` maps a model name to its two evaluation roots and, under
    "bindings", the two `release-binding.json` files of `product_manifest.py`
    whose per-file hashes every clip must still match.
    """
    pairing_path = pinned(pairing[0], pairing[1], "pairing map")
    loaded, catalogs = {}, {}
    for cohort in COHORTS:
        path, digest = inventories[cohort]
        path = pinned(path, digest, f"{cohort} inventory")
        catalogs[cohort] = read_inventory(path)
        loaded[cohort] = {
            "path": str(path),
            "sha256": digest,
            "inventory": catalogs[cohort],
        }
    document = read_json(pairing_path)
    if document["parent_inventory_sha256"] != loaded["full"]["sha256"]:
        raise PairingError("pairing map belongs to another full-resolution inventory")
    clips = {
        cohort: {clip.clip_id: clip for clip in catalogs[cohort].clips}
        for cohort in COHORTS
    }
    pairs = []
    for entry in document["clips"]:
        full = clips["full"].get(entry["parent_clip_id"])
        low = clips["low"].get(entry["clip_id"])
        if (
            full is None
            or low is None
            or full.source_sha256 != entry["parent_source_sha256"]
            or full.timeline_sha256 != entry["parent_timeline_sha256"]
            or low.source_sha256 != entry["source_sha256"]
            or low.timeline_sha256 != entry["timeline_sha256"]
            or not (full.frame_count == low.frame_count == entry["frame_count"])
        ):
            raise PairingError(f"{entry['clip_id']}: pair differs from the inventories")
        pairs.append((full, low))
    for side, cohort in enumerate(COHORTS):
        named = [pair[side].clip_id for pair in pairs]
        if len(named) != len(set(named)) or set(named) != set(clips[cohort]):
            raise PairingError("pairing map does not cover every clip exactly once")
    summaries = [
        model_pairs(name, roots, loaded, pairs) for name, roots in models.items()
    ]
    for cohort in COHORTS:
        for index, pair in enumerate(pairs):
            if not same_stages([s["pairs"][index][cohort] for s in summaries]):
                raise PairingError(
                    f"{pair[COHORTS.index(cohort)].clip_id}: models read different stages"
                )
    output = fresh_output(output)
    result = {
        "schema_version": 1,
        "kind": "paired_resolution_summary",
        "pairing": {"path": str(pairing_path), "sha256": pairing[1]},
        "inventories": {
            cohort: {key: loaded[cohort][key] for key in ("path", "sha256")}
            for cohort in COHORTS
        },
        "pairs": len(pairs),
        "source_frames": sum(pair[0].frame_count or 0 for pair in pairs),
        "definitions": {
            "observed": "frame whose observation is 'detected'",
            "camera_valid": "frame with an accepted camera",
            "raw_covered": "frame inside a prediction window, before quality rules",
            "exported_xyz": "frame with exported XYZ after the quality rules",
            "both/full_only/low_only/neither": "the same source frame in the full and the low resolution result",
        },
        "models": summaries,
        "script_sha256": file_sha256(__file__),
        "notes": list(notes),
        "limitations": LIMITS,
    }
    write_json(output / "paired.json", result)
    lines = [
        "# Paired resolution summary",
        "",
        (
            f"{len(pairs)} frozen pairs, {result['source_frames']} source frames per "
            "cohort. Counts of the same source frames in the full and the low "
            "resolution result; no accuracy is implied."
        ),
        "",
    ]
    for summary in summaries:
        lines += [
            f"## {summary['model']}",
            "",
            f"Pairs with both results: {summary['pairs_with_both_results']} of {len(pairs)}.",
            "",
            "| frames | both | full only | low only | neither |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for flag in FLAGS:
            row = summary["frames_in_pairs_with_both_results"][flag]
            lines.append(
                f"| {flag} | "
                + " | ".join(
                    str(row.get(key, 0))
                    for key in ("both", "full_only", "low_only", "neither")
                )
                + " |"
            )
        lines += ["", "| status full -> low | pairs |", "| --- | ---: |"]
        lines += [
            f"| {key} | {value} |" for key, value in summary["status_changes"].items()
        ]
        lines += ["", "| physical fit | full | low |", "| --- | ---: | ---: |"]
        lines += [
            f"| {key} | {summary['physical_counts']['full'][key]} | "
            f"{summary['physical_counts']['low'][key]} |"
            for key in FIT_COUNTS
        ]
        lines.append("")
    lines += ["## Limits", ""] + [f"- {text}" for text in [*notes, *LIMITS]]
    (output / "paired.md").write_text("\n".join(lines) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pairing", "full-inventory", "low-inventory", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("pairing-sha256", "full-inventory-sha256", "low-inventory-sha256"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=FULL_ROOT:LOW_ROOT",
        help="one model's two finished evaluation roots; repeat per model",
    )
    parser.add_argument(
        "--binding",
        action="append",
        required=True,
        metavar="NAME=FULL_BINDING:LOW_BINDING",
        help="the model's two release-binding.json files; one per --model",
    )
    parser.add_argument(
        "--note", action="append", default=[], help="limitation recorded verbatim"
    )
    args = parser.parse_args(argv)
    try:
        models = {}
        for item in args.model:
            name, roots = item.split("=", 1)
            full, low = roots.split(":", 1)
            if name in models:
                raise PairingError(f"model named twice: {name}")
            models[name] = {"full": full, "low": low}
        for item in args.binding:
            name, files = item.split("=", 1)
            full, low = files.split(":", 1)
            if name not in models or "bindings" in models[name]:
                raise PairingError(f"binding without one model: {name}")
            models[name]["bindings"] = {"full": full, "low": low}
        if any("bindings" not in roots for roots in models.values()):
            raise PairingError("every model needs its release bindings")
        result = paired(
            (args.pairing, args.pairing_sha256),
            {
                "full": (args.full_inventory, args.full_inventory_sha256),
                "low": (args.low_inventory, args.low_inventory_sha256),
            },
            models,
            args.output,
            args.note,
        )
    except (PairingError, ValueError, OSError, KeyError, TypeError) as error:
        print(f"Paired summary unavailable: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                summary["model"]: {
                    "pairs_with_both_results": summary["pairs_with_both_results"],
                    "status_changes": summary["status_changes"],
                    "exported_xyz": summary["frames_in_pairs_with_both_results"][
                        "exported_xyz"
                    ],
                }
                for summary in result["models"]
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
