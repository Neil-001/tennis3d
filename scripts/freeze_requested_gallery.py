"""Map requested source-second ranges to decoded frames and freeze a gallery.

No network, detector, or lifter is used. ``prepare`` decodes each privately
downloaded complete source once, keeps the frames the declared endpoint rule
selects, and writes verified lossless clips with their timelines. ``freeze``
joins those measured identities with a source-only review and writes the
private and portable inventories. Requested ranges are never adjusted.
"""

import argparse
import hashlib
import shutil
import sys
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import av

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_real_gallery import checked_prepared, encode
from tennis3d.artifacts import file_sha256, read_json, write_json
from tennis3d.evaluation.inventory import (
    Clip,
    Inventory,
    verify_clip,
)

ENDPOINT_RULE = (
    "Source time of a decoded frame is (pts - first decoded frame pts) * time_base "
    "of the complete downloaded file, as an exact rational. A frame is retained "
    "when start_s <= source time < end_s. The start is inclusive and the end is "
    "exclusive."
)
PREPARATION_NOTE = (
    "Exact requested-range trim of the complete downloaded source; lossless RGB "
    "libx264rgb CRF0 ultrafast, one encoder thread, movie_timescale=input "
    "timebase denominator, use_editlist=1, avoid_negative_ts=disabled. Source "
    "RGB hashes and exact retained integer PTS/time bases match every prepared "
    "decoded frame."
)
MAPPING_KEYS = (
    "clip_id",
    "requested_start_s",
    "requested_end_s",
    "original_source_frame_start",
    "original_source_frame_end_exclusive",
    "first_last_integer_pts",
    "time_base",
    "first_last_source_time_s",
    "frame_count",
)


def source_time(pts, time_base, origin_pts):
    return Fraction((pts - origin_pts) * time_base[0], time_base[1])


def map_ranges(source: Path, ranges: dict[str, tuple[int, int]]):
    """Decode once and return the retained timeline rows of every range."""
    rows = {clip_id: [] for clip_id in ranges}
    last = max(end for _, end in ranges.values())
    origin = None
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        size = [stream.width, stream.height]
        previous = None
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("source frame lacks an exact integer PTS")
            basis = [frame.time_base.numerator, frame.time_base.denominator]
            if origin is None:
                origin = frame.pts
            time = source_time(frame.pts, basis, origin)
            if previous is not None and time <= previous:
                raise ValueError("source PTS must strictly increase")
            previous = time
            if time >= last:
                break
            rgb_sha256 = None
            for clip_id, (start, end) in ranges.items():
                if not start <= time < end:
                    continue
                if [frame.width, frame.height] != size:
                    raise ValueError("source frame size changes inside a range")
                if rgb_sha256 is None:
                    rgb = frame.to_ndarray(format="rgb24")
                    rgb_sha256 = hashlib.sha256(rgb.tobytes()).hexdigest()
                rows[clip_id].append(
                    {
                        "frame_index": len(rows[clip_id]),
                        "original_frame_index": index,
                        "pts": frame.pts,
                        "time_base": basis,
                        "source_time_s": f"{time.numerator}/{time.denominator}",
                        "decoded_rgb_sha256": rgb_sha256,
                    }
                )
    return rows, origin, size


def prepare(request, sources: Path, source_files: dict[str, str], output: Path):
    """Write every requested clip, or record why its range gave no frames."""
    clips = request["provided_clips"]
    if len({clip["id"] for clip in clips}) != len(clips):
        raise ValueError("requested clip IDs must be unique")
    (output / "timelines").mkdir(parents=True, exist_ok=True)
    (output / "prepared").mkdir(exist_ok=True)
    records = []
    for video_id in dict.fromkeys(clip["video_id"] for clip in clips):
        source = sources / source_files[video_id]
        source_sha256, source_bytes = file_sha256(source), source.stat().st_size
        wanted = {
            clip["id"]: (clip["start_s"], clip["end_s"])
            for clip in clips
            if clip["video_id"] == video_id
        }
        if any(end <= start or start < 0 for start, end in wanted.values()):
            raise ValueError("requested ranges must be nonempty and nonnegative")
        mapped, origin, (width, height) = map_ranges(source, wanted)
        for clip_id, rows in mapped.items():
            start_s, end_s = wanted[clip_id]
            record = {
                "clip_id": clip_id,
                "video_id": video_id,
                "requested_start_s": start_s,
                "requested_end_s": end_s,
                "original_source_filename": source.name,
                "original_source_sha256": source_sha256,
                "original_source_bytes": source_bytes,
                "source_time_origin_pts": origin,
                "frame_count": len(rows),
            }
            records.append(record)
            if not rows:
                record["barrier"] = "requested range holds no decoded source frame"
                continue
            first, last = rows[0], rows[-1]
            timeline = {
                "source_sha256": None,
                "original_source_sha256": source_sha256,
                "original_source_frame_start": first["original_frame_index"],
                "original_source_frame_end_exclusive": last["original_frame_index"] + 1,
                "image_size": [width, height],
                "requested_source_seconds": [start_s, end_s],
                "source_time_origin_pts": origin,
                "endpoint_rule": ENDPOINT_RULE,
                "index_contract": "original_frame_index is the zero-based decoded "
                "frame of the complete downloaded source file",
                "frames": rows,
            }
            if last["original_frame_index"] + 1 - first["original_frame_index"] != len(
                rows
            ):
                raise ValueError("retained source frames are not contiguous")
            target = output / "prepared" / f"{clip_id}.mp4"
            timeline_path = output / "timelines" / f"{clip_id}.json"
            draft = Clip.model_validate(
                {
                    "clip_id": clip_id,
                    "recording_group": "pending source review",
                    "recording_evidence": "pending source review",
                    "distinct_recording_verified": False,
                    "source_url": f"https://www.youtube.com/watch?v={video_id}",
                    "original_source_sha256": source_sha256,
                    "original_source_bytes": source_bytes,
                    "original_source_frame_start": first["original_frame_index"],
                    "original_source_frame_end_exclusive": last["original_frame_index"]
                    + 1,
                    "frame_count": len(rows),
                    "width": width,
                    "height": height,
                    "time_basis": "source_pts",
                    "timing_note": ENDPOINT_RULE,
                    "local_use": "unknown",
                    "redistribution": "unknown",
                    "rights_evidence": "pending source review",
                    "label_status": "unavailable",
                    "detector_training_independence": "unknown",
                    "categories": [],
                    "eligible": False,
                    "barriers": ["source encoding pending"],
                }
            )
            if not target.exists():
                if shutil.disk_usage(output).free < width * height * 3 * len(rows) * 2:
                    raise ValueError("insufficient free disk for prepared bound")
                # encode() rechecks every source RGB hash and PTS in a second decode.
                encode(draft, source, target, timeline)
            timeline["source_sha256"] = file_sha256(target)
            write_json(timeline_path, timeline, overwrite=True)
            prepared = draft.model_copy(
                update={
                    "source_path": str(target.resolve()),
                    "source_sha256": timeline["source_sha256"],
                    "source_bytes": target.stat().st_size,
                    "timeline_path": str(timeline_path.resolve()),
                    "timeline_sha256": file_sha256(timeline_path),
                }
            )
            checked_prepared(
                prepared.model_copy(update={"eligible": True}), target, timeline
            )
            times = [
                source_time(row["pts"], row["time_base"], origin)
                for row in (first, last)
            ]
            record.update(
                source_path=prepared.source_path,
                source_sha256=prepared.source_sha256,
                source_bytes=prepared.source_bytes,
                timeline_path=prepared.timeline_path,
                timeline_sha256=prepared.timeline_sha256,
                width=width,
                height=height,
                original_source_frame_start=first["original_frame_index"],
                original_source_frame_end_exclusive=last["original_frame_index"] + 1,
                first_last_integer_pts=[first["pts"], last["pts"]],
                time_base=first["time_base"],
                first_last_source_time_s=[float(time) for time in times],
                every_prepared_rgb_and_integer_pts_matches_source=True,
            )
    return records


def freeze(request, prepared, review, provenance, config_name: str, output: Path):
    """Join measured identities with the source-only review; keep every clip."""
    records = {record["clip_id"]: record for record in prepared["records"]}
    requested = [clip["id"] for clip in request["provided_clips"]]
    if len(requested) != request["expected_final_clips"] or set(requested) != set(
        records
    ):
        raise ValueError("prepared records do not cover the complete request")
    if set(review["clips"]) != set(requested):
        raise ValueError("source review must cover every requested clip")
    clips, portable = [], []
    for clip_id in requested:
        record, notes = records[clip_id], review["clips"][clip_id]
        recording = review["recordings"][record["video_id"]]
        barriers = list(notes.get("barriers", []))
        if "barrier" in record:
            barriers.append(record["barrier"])
        values = {
            "clip_id": clip_id,
            "recording_group": recording["recording_group"],
            "recording_evidence": recording["recording_evidence"],
            "distinct_recording_verified": recording["distinct_recording_verified"],
            "source_url": f"https://www.youtube.com/watch?v={record['video_id']}",
            "original_source_sha256": record["original_source_sha256"],
            "original_source_bytes": record["original_source_bytes"],
            "preparation_note": PREPARATION_NOTE,
            "time_basis": "source_pts" if record["frame_count"] else "unavailable",
            "timing_note": " ".join(
                [review["timing_note"], *notes.get("limitations", [])]
            ),
            "local_use": recording["local_use"],
            "redistribution": recording["redistribution"],
            "rights_evidence": recording["rights_evidence"],
            "label_status": "unavailable",
            "detector_training_independence": "unknown",
            "categories": notes["categories"],
            "category_evidence": notes["category_evidence"],
            "eligible": not barriers,
            "barriers": barriers,
        }
        for key in (
            "source_sha256",
            "source_bytes",
            "source_path",
            "timeline_path",
            "timeline_sha256",
            "frame_count",
            "width",
            "height",
            "original_source_frame_start",
            "original_source_frame_end_exclusive",
        ):
            values[key] = record.get(key)
        if not record["frame_count"]:
            values["frame_count"] = None
            values["original_source_sha256"] = values["original_source_bytes"] = None
        clip = Clip.model_validate(values)
        if clip.eligible:
            verify_clip(clip)
        clips.append(clip.model_dump())
        public = clip.model_dump()
        if clip.source_path is not None:
            public["source_path"] = f"data/{config_name}/prepared/{clip_id}.mp4"
            public["timeline_path"] = f"configs/{config_name}/timelines/{clip_id}.json"
        portable.append(public)
    common = {
        "frozen_utc": datetime.now(UTC).isoformat(),
        "intended_role": "representative_end_to_end",
        "selection_rule": review["selection_rule"],
        "excluded_development_recording_groups": review[
            "excluded_development_recording_groups"
        ],
        "excluded_development_source_sha256": review[
            "excluded_development_source_sha256"
        ],
        "exclusion_evidence": review["exclusion_evidence"],
        "limitations": review["limitations"],
    }
    private = Inventory.model_validate({**common, "clips": clips})
    public = Inventory.model_validate({**common, "clips": portable})
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "inventory.private.json", private.model_dump())
    write_json(output / "inventory.json", public.model_dump())
    sources = []
    for source in provenance["sources"]:
        rows = [
            record
            for record in prepared["records"]
            if record["video_id"] == source["youtube_id"]
        ]
        sources.append(
            {
                **source,
                "downloaded_sha256": rows[0]["original_source_sha256"],
                "downloaded_bytes": rows[0]["original_source_bytes"],
                "source_time_origin_pts": rows[0]["source_time_origin_pts"],
                "requested_clips": [
                    {key: row.get(key) for key in MAPPING_KEYS} for row in rows
                ],
            }
        )
    write_json(
        output / "provenance.json",
        {
            **provenance,
            "source_only": True,
            "endpoint_rule": ENDPOINT_RULE,
            "request_sha256": prepared["request_sha256"],
            "sources": sources,
        },
    )
    return private.summary()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    step = commands.add_parser("prepare")
    step.add_argument("--request", type=Path, required=True)
    step.add_argument("--sources", type=Path, required=True)
    step.add_argument(
        "--source-files",
        type=Path,
        required=True,
        help="JSON object mapping each video ID to its file under --sources",
    )
    step.add_argument("--output", type=Path, required=True)
    step = commands.add_parser("freeze")
    step.add_argument("--request", type=Path, required=True)
    step.add_argument("--prepared", type=Path, required=True)
    step.add_argument("--review", type=Path, required=True)
    step.add_argument("--provenance", type=Path, required=True)
    step.add_argument("--config-name", required=True)
    step.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = read_json(args.request)
    if args.command == "prepare":
        if (args.output / "prepared-records.json").exists():
            parser.error("prepared records already exist")
        records = prepare(
            request, args.sources, read_json(args.source_files), args.output
        )
        write_json(
            args.output / "prepared-records.json",
            {
                "request_sha256": file_sha256(args.request),
                "endpoint_rule": ENDPOINT_RULE,
                "pyav": av.__version__,
                "linked_libraries": av.library_versions,
                "source_only": True,
                "records": records,
            },
        )
        return
    prepared = read_json(args.prepared)
    if prepared["request_sha256"] != file_sha256(args.request):
        parser.error("prepared records belong to another request")
    print(
        freeze(
            request,
            prepared,
            read_json(args.review),
            read_json(args.provenance),
            args.config_name,
            args.output,
        )
    )


if __name__ == "__main__":
    main()
