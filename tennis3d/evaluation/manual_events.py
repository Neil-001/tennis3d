"""Read-only source-bound manual-event snapshots, without fabricated UV labels."""

from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from tennis3d.artifacts import file_sha256, read_json
from tennis3d.evaluation.contacts import MANUAL_EXCLUSION_S, TOLERANCES_S
from tennis3d.evaluation.runs import timeline


def load_contact_plan(path, expected_sha256):
    if file_sha256(path) != expected_sha256:
        raise ValueError("frozen contact plan hash mismatch")
    plan = read_json(path)
    if (
        plan.get("frozen_before_model_output_scoring") is not True
        or plan["primary_tolerance_seconds"] != MANUAL_EXCLUSION_S
        or plan["sensitivity_tolerance_seconds"] != list(TOLERANCES_S)
        or plan["manual_segmentation_contract"]["primary_exclusion_precision_seconds"]
        != MANUAL_EXCLUSION_S
        or plan["manual_segmentation_contract"]["additive_segmentation_source"]
        != "manual_events"
    ):
        raise ValueError("frozen plan differs from implemented contact contract")
    return plan


def load_manual_snapshot(root, expected_receipt_sha256):
    """Verify every frozen file, timeline row and rational selected event time.

    Timeline paths are those of the immutable inventory. No model outputs,
    checkpoint, browser state or production label files are read here.
    """
    root = Path(root)
    receipt_path = root / "receipt.json"
    if file_sha256(receipt_path) != expected_receipt_sha256:
        raise ValueError("manual snapshot receipt hash mismatch")
    receipt = read_json(receipt_path)
    inventory_path = root / "inventory.private.json"
    if file_sha256(inventory_path) != receipt["inventory_sha256"]:
        raise ValueError("manual snapshot inventory hash mismatch")
    inventory = read_json(inventory_path)
    clips = inventory["clips"]
    names = {f"{c['clip_id']}.json" for c in clips}
    if (
        len(names) != len(clips)
        or names != set(receipt["files"])
        or names != {p.name for p in (root / "labels").glob("*.json")}
        or receipt["clips"] != len(clips)
    ):
        raise ValueError("manual snapshot membership differs from receipt/inventory")
    result, counts = [], Counter()
    for clip in clips:
        label_path = root / "labels" / f"{clip['clip_id']}.json"
        label_hash = file_sha256(label_path)
        entry = receipt["files"][label_path.name]
        if label_hash != entry["sha256"] or label_path.stat().st_size != entry["bytes"]:
            raise ValueError("manual annotation hash/size mismatch")
        label = read_json(label_path)
        reviewed = label["reviewed_event_inventory"]
        if (
            label["schema"] != "tennis3d.manual-events/v1"
            or label["submitted"] is not True
            or label["clip_id"] != clip["clip_id"]
            or label["frame_count"] != clip["frame_count"]
            or label["revision"] != entry["revision"]
            or len(label["events"]) != entry["events"]
            or reviewed["frame_start"] != 0
            or reviewed["frame_end_exclusive"] != clip["frame_count"]
            or set(reviewed["types"]) != {"hit", "bounce"}
        ):
            raise ValueError("manual annotation submission/coverage contract mismatch")
        identity = label["identity"]
        if identity != {
            "inventory_sha256": receipt["inventory_sha256"],
            "source_sha256": clip["source_sha256"],
            "timeline_sha256": clip["timeline_sha256"],
        }:
            raise ValueError("manual annotation source identity mismatch")
        timeline_path = Path(clip["timeline_path"])
        if file_sha256(timeline_path) != clip["timeline_sha256"]:
            raise ValueError("frozen timeline hash mismatch")
        source = read_json(timeline_path)
        frames = source["frames"]
        if (
            source["source_sha256"] != clip["source_sha256"]
            or len(frames) != clip["frame_count"]
        ):
            raise ValueError("timeline source/frame count mismatch")
        times = []
        for i, frame in enumerate(frames):
            if (
                frame["frame_index"] != i
                or frame["original_frame_index"]
                != clip["original_source_frame_start"] + i
                or type(frame["pts"]) is not int
                or len(frame["time_base"]) != 2
                or any(type(v) is not int or v <= 0 for v in frame["time_base"])
            ):
                raise ValueError("frozen timeline integer PTS/index contract mismatch")
            times.append(
                Fraction(frame["pts"] * frame["time_base"][0], frame["time_base"][1])
            )
        t_s = timeline([float(t) for t in times])
        refs, seen = [], set()
        for event in label["events"]:
            i = event["frame_index"]
            kind = event["type"]
            if (
                type(i) is not int
                or not 0 <= i < len(frames)
                or kind not in ("hit", "bounce")
                or (kind, i) in seen
                or event["source"] != "manual"
            ):
                raise ValueError("invalid/duplicate manual event selection")
            seen.add((kind, i))
            frame = frames[i]
            if (
                any(
                    event[k] != frame[k]
                    for k in (
                        "pts",
                        "time_base",
                        "original_frame_index",
                        "decoded_rgb_sha256",
                    )
                )
                or event["source_frame_index"] != i
            ):
                raise ValueError("manual event differs from frozen decoded row")
            if (
                Fraction(
                    event["source_time"]["numerator"],
                    event["source_time"]["denominator"],
                )
                != times[i]
                or event["source_time_seconds"] != float(times[i])
                or event["elapsed_clip_seconds"] != float(times[i] - times[0])
            ):
                raise ValueError("manual event rational source time mismatch")
            refs.append(
                {
                    "frame_index": i,
                    "t_s": float(times[i]),
                    "kind": kind,
                    "annotation": event,
                }
            )
            counts[kind] += 1
        result.append(
            {
                "clip": clip,
                "references": refs,
                "t_s": t_s,
                "frames": frames,
                "annotation_sha256": label_hash,
                "label": label,
            }
        )
    if dict(counts) != receipt["event_counts"]:
        raise ValueError("manual event totals differ from receipt")
    return result, {
        "receipt_sha256": expected_receipt_sha256,
        "inventory_sha256": receipt["inventory_sha256"],
        "clips": len(result),
        "event_counts": dict(counts),
        "human_timing_caveat": receipt["human_timing_caveat"],
        "timing_interpretation": receipt["timing_interpretation"],
    }


def manual_event_arrays(references):
    return {
        "event_t_s": np.array([r["t_s"] for r in references], dtype=float),
        "event_code": np.array(
            [1 if r["kind"] == "bounce" else 2 for r in references], dtype=np.int64
        ),
        "event_precision_s": np.full(len(references), MANUAL_EXCLUSION_S),
    }
