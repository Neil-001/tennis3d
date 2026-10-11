"""Reproduce frozen gallery clips from preserved private downloaded sections.

No network, detector, or lifter is used. Stream frames to bound RGB memory and
retain each downloaded-file integer PTS, time base, and decoded RGB identity.
"""

import argparse
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import av

from tennis3d.artifacts import file_sha256, read_json, write_json
from tennis3d.evaluation.inventory import Clip, read_inventory, verify_clip


def checked_source(clip: Clip, source: Path):
    if (
        source.stat().st_size != clip.original_source_bytes
        or file_sha256(source) != clip.original_source_sha256
    ):
        raise ValueError("downloaded source identity differs from frozen section")


def source_frames(clip: Clip, source: Path, timeline):
    """Yield one retained RGB frame after checking the frozen source mapping."""
    if (
        clip.original_source_frame_start is None
        or clip.original_source_frame_end_exclusive is None
    ):
        raise ValueError("frozen source interval is unavailable")
    checked_source(clip, source)
    count = 0
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            if index >= clip.original_source_frame_end_exclusive:
                break
            if index < clip.original_source_frame_start:
                continue
            if count >= len(timeline["frames"]):
                raise ValueError("source interval contains extra frames")
            row = timeline["frames"][count]
            rgb = frame.to_ndarray(format="rgb24")
            if (
                frame.pts is None
                or frame.time_base is None
                or frame.pts != row["pts"]
                or [frame.time_base.numerator, frame.time_base.denominator]
                != row["time_base"]
                or index != row["original_frame_index"]
                or [frame.width, frame.height] != [clip.width, clip.height]
                or hashlib.sha256(rgb.tobytes()).hexdigest()
                != row["decoded_rgb_sha256"]
            ):
                raise ValueError("source frame pixels or exact timestamp differ")
            count += 1
            yield frame.pts, frame.time_base, rgb, stream.average_rate
    if count != clip.frame_count:
        raise ValueError("source omits frozen retained frames")


def encode(clip: Clip, source: Path, target: Path, timeline):
    """Encode one frame at a time without rebasing or interpolating timestamps."""
    if clip.width is None or clip.height is None:
        raise ValueError("frozen source dimensions are unavailable")
    if target.exists():
        raise FileExistsError(target)
    frames = iter(source_frames(clip, source, timeline))
    pts, basis, rgb, rate = next(frames)
    with av.open(
        str(target),
        "w",
        format="mp4",
        options={
            "movie_timescale": str(basis.denominator),
            "use_editlist": "1",
            "avoid_negative_ts": "disabled",
        },
    ) as destination:
        stream = destination.add_stream("libx264rgb", rate=rate)
        assert isinstance(stream, av.video.stream.VideoStream)
        stream.width, stream.height, stream.pix_fmt = clip.width, clip.height, "rgb24"
        stream.time_base = basis
        stream.codec_context.time_base = basis
        stream.codec_context.thread_count = 1
        stream.options = {"crf": "0", "preset": "ultrafast"}

        def push(pts, time_base, pixels):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = pts, time_base
            for packet in stream.encode(frame):
                destination.mux(packet)

        push(pts, basis, rgb)
        del rgb
        for pts, basis, rgb, _ in frames:
            push(pts, basis, rgb)
        for packet in stream.encode():
            destination.mux(packet)


def checked_prepared(clip: Clip, target: Path, timeline):
    relocated = clip.model_copy(update={"source_path": str(target)})
    verify_clip(relocated)
    count = 0
    with av.open(str(target)) as container:
        container.streams.video[0].codec_context.thread_count = 1
        for frame in container.decode(video=0):
            if count >= len(timeline["frames"]):
                raise ValueError("prepared clip has extra frames")
            row = timeline["frames"][count]
            if (
                frame.pts is None
                or frame.time_base is None
                or frame.pts != row["pts"]
                or [frame.time_base.numerator, frame.time_base.denominator]
                != row["time_base"]
                or [frame.width, frame.height] != [clip.width, clip.height]
                or hashlib.sha256(
                    frame.to_ndarray(format="rgb24").tobytes()
                ).hexdigest()
                != row["decoded_rgb_sha256"]
            ):
                raise ValueError("prepared RGB or integer PTS/time base differs")
            count += 1
    if count != clip.frame_count:
        raise ValueError("prepared clip omits frozen source frames")


def prepare(clip: Clip, source: Path, output: Path):
    if not clip.eligible or clip.original_source_sha256 is None:
        raise ValueError("preparation requires a frozen eligible source interval")
    if (
        clip.timeline_path is None
        or clip.width is None
        or clip.height is None
        or clip.frame_count is None
    ):
        raise ValueError("frozen source timeline or dimensions are unavailable")
    if file_sha256(clip.timeline_path) != clip.timeline_sha256:
        raise ValueError("frozen timeline changed")
    timeline = read_json(clip.timeline_path)
    target = output / f"{clip.clip_id}.mp4"
    output.mkdir(parents=True, exist_ok=True)
    # A conservative uncompressed RGB bound avoids relying on compression ratio.
    decoded_size = clip.width * clip.height * 3 * clip.frame_count
    if not target.exists():
        if shutil.disk_usage(output).free < decoded_size * 2:
            raise ValueError("insufficient free disk for conservative prepared bound")
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output) as temporary:
            candidate = Path(temporary) / "clip.mp4"
            encode(clip, source, candidate, timeline)
            checked_prepared(clip, candidate, timeline)
            os.link(candidate, target)
    else:
        for _ in source_frames(clip, source, timeline):
            pass
    checked_prepared(clip, target, timeline)
    return {
        "clip_id": clip.clip_id,
        "downloaded_source_sha256": clip.original_source_sha256,
        "prepared_sha256": clip.source_sha256,
        "source_path": str(target.resolve()),
        "frames": clip.frame_count,
        "every_rgb_and_integer_pts_verified": True,
        "decoded_rgb_size_bound_bytes": decoded_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-id")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        parser.error("receipt already exists")
    inventory = read_inventory(args.inventory)
    sources = {
        row["downloaded_sha256"]: row["downloaded_filename"]
        for row in read_json(args.provenance)["sources"]
    }
    selected = [
        clip
        for clip in inventory.clips
        if clip.eligible and (args.clip_id is None or clip.clip_id == args.clip_id)
    ]
    if not selected:
        parser.error("eligible clip ID absent from frozen inventory")
    missing = [
        clip.clip_id
        for clip in selected
        if clip.original_source_sha256 is None
        or not sources.get(clip.original_source_sha256)
    ]
    if missing:
        parser.error(f"provenance lacks downloaded source for: {', '.join(missing)}")
    records = [
        prepare(clip, args.sources / sources[clip.original_source_sha256], args.output)
        for clip in selected
    ]
    write_json(
        args.receipt,
        {
            "inventory_sha256": file_sha256(args.inventory),
            "provenance_sha256": file_sha256(args.provenance),
            "pyav": av.__version__,
            "linked_libraries": av.library_versions,
            "source_only": True,
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
