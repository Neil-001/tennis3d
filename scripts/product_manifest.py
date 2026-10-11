"""App gallery manifest and release binding for one finished real evaluation.

Reads a completed `cached_real.py` or `delivery real` root and writes two
files into a new directory. `product-manifest.json` is the app's gallery
manifest: every inventory clip with its source, timeline and unchanged
`result.json`. `release-binding.json` records the hashes and identities those
entries rest on, and counts raw and exported gaps per clip. Nothing is
recomputed, repaired or relabelled, and a clip without a result stays listed
with its status.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from tennis3d.artifacts import file_sha256, read_json, read_npz, write_json
from tennis3d.evaluation.delivery import checked_release, fresh_output
from tennis3d.evaluation.inventory import read_inventory
from tennis3d.schemas import read_result

LIMITS = [
    "No independent real XYZ, ball UV or camera truth exists for these clips.",
    "Raw coverage counts rows inside a prediction window; exported XYZ also passes the quality rules.",
    "Footage and detector weight redistribution rights are unverified.",
]


class ReleaseError(Exception):
    """The evaluation root does not support a product manifest."""


def entry(path):
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def clip_binding(gallery, clip, record):
    """One clip's files and row counts, checked against its own summaries."""
    run = gallery / "clips" / clip.clip_id
    row = {
        "clip_id": clip.clip_id,
        "source_sha256": clip.source_sha256,
        "timeline_sha256": clip.timeline_sha256,
        "frame_count": clip.frame_count,
        "status": record["status"],
        "failure_reasons": record.get("failure_reasons", []),
        "result": None,
    }
    if not record.get("result_path"):
        return row
    pipeline = record["pipeline"]
    result_path = gallery / record["result_path"]
    if (
        result_path != run / "result.json"
        or file_sha256(result_path) != pipeline["artifacts"]["result.json"]
        or file_sha256(run / "export.json") != pipeline["artifacts"]["export.json"]
        or file_sha256(run / "detect.json") != pipeline["artifacts"]["detect.json"]
    ):
        raise ReleaseError(f"{clip.clip_id}: result differs from its run summary")
    result = read_result(result_path)
    lifting = read_json(run / "lift.json")
    raw, _ = read_npz(run / "predictions_raw.npz")
    if (
        result.source.sha256 != clip.source_sha256
        or result.source.frame_count != clip.frame_count
        or result.status != record["status"]
        or len(raw["covered"]) != clip.frame_count
        or file_sha256(run / "predictions_raw.npz") != lifting["predictions_sha256"]
        or file_sha256(run / "lift.json") != pipeline["artifacts"]["lift.json"]
    ):
        raise ReleaseError(f"{clip.clip_id}: result does not match the frozen clip")
    exported = sum(frame.xyz_m is not None for frame in result.frames)
    covered = int(raw["covered"].sum())
    detection = read_json(run / "detect.json")
    row |= {
        "result": entry(result_path),
        "run_path": str(run),
        "artifacts": {
            name: entry(run / name)
            for name in (
                "predictions_raw.npz",
                "lift.json",
                "export.json",
                "metrics.json",
                "manifest.json",
                "physical.json",
            )
            if (run / name).is_file()
        },
        "upstream_sha256": pipeline.get("cached_inputs", {}).get("upstream_sha256"),
        "detector_assets_sha256": {
            name: detection["assets"]["assets"][name]["sha256"]
            for name in ("ball", "court")
        },
        "checkpoint_sha256": lifting["checkpoint_sha256"],
        "rows": {
            "source": clip.frame_count,
            "raw_covered": covered,
            "raw_gap": clip.frame_count - covered,
            "exported_xyz": exported,
            "exported_gap": clip.frame_count - exported,
            "camera_valid": sum(frame.camera_valid for frame in result.frames),
            "observation": dict(
                sorted(Counter(frame.observation for frame in result.frames).items())
            ),
            "reconstruction": dict(
                sorted(Counter(frame.reconstruction for frame in result.frames).items())
            ),
            "reason_codes": dict(
                sorted(
                    Counter(
                        reason
                        for frame in result.frames
                        for reason in frame.reason_codes
                    ).items()
                )
            ),
        },
        "coverage": pipeline["coverage"],
    }
    return row


def note(row):
    """Status and coverage in words, for the manifest's clip note."""
    if row["result"] is None:
        reasons = "; ".join(row["failure_reasons"]) or "no reason recorded"
        return f"{row['status']}: no result ({reasons})"
    rows = row["rows"]
    text = (
        f"{row['status']}: {rows['exported_xyz']} of {rows['source']} source frames "
        f"have exported XYZ; {rows['raw_covered']} have raw model output"
    )
    if row["failure_reasons"]:
        text += f"; {', '.join(row['failure_reasons'])}"
    return text


def product(gallery, inventory_path, inventory_sha256, clips_path, release_path, out):
    gallery = Path(gallery).resolve()
    if file_sha256(inventory_path) != inventory_sha256:
        raise ReleaseError("inventory differs from --inventory-sha256")
    inventory = read_inventory(inventory_path)
    metrics = read_json(gallery / "metrics.json")
    if (
        metrics.get("kind") != "end_to_end_real"
        or metrics["inventory_sha256"] != inventory_sha256
        or (gallery / "aborted.json").exists()
    ):
        raise ReleaseError("gallery is not a completed evaluation of this inventory")
    release, _, _ = checked_release(release_path)
    recorded = metrics["provenance"]["release"]["checkpoint"]["sha256"]
    if recorded != release.checkpoint.sha256:
        raise ReleaseError("gallery was evaluated with another checkpoint")
    records = {record["clip_id"]: record for record in metrics["records"]}
    listed = {clip["id"]: clip for clip in read_json(clips_path)["clips"]}
    ids = [clip.clip_id for clip in inventory.clips]
    if len(records) != len(metrics["records"]) or not (
        set(records) == set(listed) == set(ids)
    ):
        raise ReleaseError("gallery, inventory and clip list name different clips")
    bindings, clips = [], []
    for clip in inventory.clips:
        shown = listed[clip.clip_id]
        if (shown["source_sha256"], shown["timeline_sha256"]) != (
            clip.source_sha256,
            clip.timeline_sha256,
        ):
            raise ReleaseError(f"{clip.clip_id}: clip list names another source")
        row = clip_binding(gallery, clip, records[clip.clip_id])
        if row["result"] and row["checkpoint_sha256"] != release.checkpoint.sha256:
            raise ReleaseError(f"{clip.clip_id}: lifted with another checkpoint")
        bindings.append(row)
        item = {
            key: shown[key]
            for key in ("id", "title", "rights", "media_offset_s")
            if key in shown
        } | {
            "source_sha256": clip.source_sha256,
            "timeline_sha256": clip.timeline_sha256,
            "note": note(row),
        }
        # The app resolves relative paths from the manifest's own directory.
        for key in ("source_path", "timeline_path"):
            if getattr(clip, key):
                item[key] = str(Path(getattr(clip, key)).resolve())
        if row["result"]:
            item |= {
                "result_path": row["result"]["path"],
                "result_sha256": row["result"]["sha256"],
            }
        clips.append(item)
    out = fresh_output(out)
    write_json(out / "product-manifest.json", {"schema_version": 1, "clips": clips})
    totals = Counter()
    for row in bindings:
        totals.update(
            {
                key: value
                for key, value in (row.get("rows") or {}).items()
                if isinstance(value, int)
            }
        )
    binding = {
        "schema_version": 1,
        "kind": "real_product_release_binding",
        "gallery": str(gallery),
        "gallery_metrics": entry(gallery / "metrics.json"),
        "gallery_config_sha256": metrics["config_sha256"],
        "inventory": entry(Path(inventory_path).resolve()),
        "clip_list": entry(Path(clips_path).resolve()),
        "release": entry(Path(release_path).resolve()),
        "checkpoint": {
            "sha256": release.checkpoint.sha256,
            "run_id": release.run_id,
            "step": release.step,
        },
        "identities": metrics["provenance"].get("cached_inputs"),
        "product_manifest": entry(out / "product-manifest.json"),
        "script_sha256": file_sha256(__file__),
        "status_counts": dict(
            sorted(Counter(row["status"] for row in bindings).items())
        ),
        "source_rows": sum(row["frame_count"] or 0 for row in bindings),
        "row_totals_over_clips_with_results": dict(sorted(totals.items())),
        "clips": bindings,
        "limitations": LIMITS,
    }
    write_json(out / "release-binding.json", binding)
    return binding


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gallery", "inventory", "clips", "release", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        binding = product(
            args.gallery,
            args.inventory,
            args.inventory_sha256,
            args.clips,
            args.release,
            args.output,
        )
    except (ReleaseError, ValueError, OSError, KeyError, TypeError) as error:
        print(f"Product manifest unavailable: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                key: binding[key]
                for key in (
                    "product_manifest",
                    "status_counts",
                    "source_rows",
                    "row_totals_over_clips_with_results",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
