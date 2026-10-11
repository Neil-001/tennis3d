"""Write the browser display copies shipped with a precomputed example gallery.

``derive`` reads each frozen source clip named by a private input manifest,
requires its recorded SHA-256 and the gallery's timeline, and writes
``<clip>/playback.mp4`` with a ``playback.json`` receipt beside it. The copy
keeps every source frame at its exact elapsed source time. ``verify`` decodes
the shipped copies again and checks them against their receipts and timelines;
it needs no source clip.

The copies are lossy and much smaller than their lossless sources. They are for display
only: results and timelines come from the original pixels and stay unchanged.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import av
from av.video.reformatter import ColorRange, Colorspace, Interpolation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis3d.artifacts import file_sha256, read_json, write_json

HEIGHT = 1080
ENCODER: dict[str, object] = {
    "crf": "32",
    "deadline": "good",
    "cpu-used": "2",
    "row-mt": "1",
    "tile-columns": "2",
    "lag-in-frames": "25",
    "auto-alt-ref": "1",
}
ENCODING = (
    "VP9 profile 0 in MP4, libvpx-vp9 constant quality "
    + " ".join(f"{key}={value}" for key, value in ENCODER.items())
    + ", one keyframe a second at most, 8-bit 4:2:0 BT.709 limited range, "
    f"{HEIGHT} lines, no audio"
)


def _timeline_rows(timeline: dict) -> list[tuple[int, Fraction]]:
    return [(row["pts"], Fraction(*row["time_base"])) for row in timeline["frames"]]


def _decoded(path: Path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"{path} has a frame without a timestamp")
            yield frame


def derive_clip(clip: dict, source: Path, folder: Path) -> dict:
    timeline_path = folder / "timeline.json"
    if file_sha256(timeline_path) != clip["timeline_sha256"]:
        raise ValueError(f"{clip['id']}: timeline differs from the gallery manifest")
    timeline = read_json(timeline_path)
    rows = _timeline_rows(timeline)
    if file_sha256(source) != clip["source_sha256"] or (
        timeline["source_sha256"] != clip["source_sha256"]
    ):
        raise ValueError(f"{clip['id']}: source differs from the gallery manifest")
    output = folder / "playback.mp4"
    temporary = folder / f".playback.{os.getpid()}.mp4"
    mapping = []
    try:
        with (
            av.open(str(source)) as original,
            av.open(str(temporary), mode="w", format="mp4") as encoded,
        ):
            video = original.streams.video[0]
            time_base, rate = video.time_base, video.average_rate
            if time_base is None or rate is None:
                raise ValueError(f"{clip['id']}: source has no time base or rate")
            stream = encoded.add_stream("libvpx-vp9", rate=rate)
            assert isinstance(stream, av.VideoStream)
            stream.height = HEIGHT
            stream.width = round(video.width * HEIGHT / video.height / 2) * 2
            stream.pix_fmt = "yuv420p"
            context = stream.codec_context
            context.time_base = time_base
            context.bit_rate = 0
            context.gop_size = round(rate)
            context.thread_count = 4
            context.colorspace = Colorspace.ITU709
            context.color_range = ColorRange.MPEG
            context.color_primaries = 1
            context.color_trc = 1
            stream.options = ENCODER
            for index, frame in enumerate(original.decode(video)):
                if index >= len(rows) or (frame.pts, frame.time_base) != rows[index]:
                    raise ValueError(f"{clip['id']}: frame {index} is not in timeline")
                elapsed = (frame.pts - rows[0][0]) * time_base
                pts = elapsed / time_base
                mapping.append(
                    {
                        "frame_index": index,
                        "source_pts": frame.pts,
                        "source_time_base": [
                            time_base.numerator,
                            time_base.denominator,
                        ],
                        "elapsed_s": float(elapsed),
                        "playback_pts": int(pts),
                    }
                )
                frame = frame.reformat(
                    width=stream.width,
                    height=stream.height,
                    format="yuv420p",
                    dst_colorspace=Colorspace.ITU709,
                    dst_color_range=ColorRange.MPEG,
                    interpolation=Interpolation.LANCZOS,
                )
                frame.pts, frame.time_base = int(pts), time_base
                encoded.mux(stream.encode(frame))
            encoded.mux(stream.encode())
            size = [stream.width, stream.height]
        if len(mapping) != len(rows):
            raise ValueError(f"{clip['id']}: source has fewer frames than timeline")
        if file_sha256(source) != clip["source_sha256"]:
            raise ValueError(f"{clip['id']}: source changed during encoding")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {
        "schema_version": "1.0",
        "kind": "playback_proxy",
        "clip_id": clip["id"],
        "source_sha256": clip["source_sha256"],
        "timeline_sha256": clip["timeline_sha256"],
        "playback_sha256": file_sha256(output),
        "playback_bytes": output.stat().st_size,
        "playback_image_size": size,
        "source_image_size": timeline["image_size"],
        "media_offset_s": 0.0,
        "encoding": ENCODING,
        "pixel_identity": (
            "Lossy display copy. Its pixels differ from the source; "
            "the results and timeline were computed from the source."
        ),
        "frames": mapping,
    }
    write_json(folder / "playback.json", receipt, overwrite=True)
    verify_clip(clip | {"playback_sha256": receipt["playback_sha256"]}, folder)
    return receipt


def verify_clip(clip: dict, folder: Path) -> int:
    """Decode a shipped copy and require its receipt's and timeline's exact times."""
    name = clip["id"]
    receipt = read_json(folder / "playback.json")
    timeline = read_json(folder / "timeline.json")
    digest = file_sha256(folder / "playback.mp4")
    if not (digest == clip["playback_sha256"] == receipt["playback_sha256"]):
        raise ValueError(f"{name}: display copy differs from its recorded SHA-256")
    for key in ("source_sha256", "timeline_sha256"):
        if receipt[key] != clip[key]:
            raise ValueError(f"{name}: receipt names another {key}")
    if file_sha256(folder / "timeline.json") != clip["timeline_sha256"]:
        raise ValueError(f"{name}: timeline differs from the gallery manifest")
    rows, mapping = _timeline_rows(timeline), receipt["frames"]
    frames = list(_decoded(folder / "playback.mp4"))
    if not len(frames) == len(mapping) == len(rows):
        raise ValueError(f"{name}: frame counts differ")
    for index, (frame, row, (pts, base)) in enumerate(zip(frames, mapping, rows)):
        elapsed = (pts - rows[0][0]) * base
        if (
            row["frame_index"] != index
            or (row["source_pts"], Fraction(*row["source_time_base"])) != (pts, base)
            or row["playback_pts"] * base != elapsed
            or frame.pts * frame.time_base != elapsed
            or [frame.width, frame.height] != receipt["playback_image_size"]
        ):
            raise ValueError(f"{name}: frame {index} is not at its source time")
    return len(frames)


def _derive(job):
    clip, source, folder = job
    receipt = derive_clip(clip, source, folder)
    return clip["id"], receipt["playback_sha256"], receipt["playback_bytes"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    derive = commands.add_parser("derive", help="encode copies from frozen sources")
    derive.add_argument("--sources", type=Path, required=True, help="input manifest")
    derive.add_argument("--jobs", type=int, default=4)
    check = commands.add_parser("verify", help="decode and check the shipped copies")
    for command in (derive, check):
        command.add_argument("--gallery", type=Path, required=True)
    args = parser.parse_args(argv)
    document = read_json(args.gallery)
    root = args.gallery.resolve().parent
    if args.command == "verify":
        count = sum(verify_clip(clip, root / clip["id"]) for clip in document["clips"])
        print(f"Verified {len(document['clips'])} display copies, {count} frames.")
        return 0
    sources = {clip["id"]: clip for clip in read_json(args.sources)["clips"]}
    if list(sources) != [clip["id"] for clip in document["clips"]]:
        raise ValueError("Input manifest and gallery list different clips")
    jobs = []
    for clip in document["clips"]:
        source = sources[clip["id"]]
        for key in ("source_sha256", "timeline_sha256"):
            if source[key] != clip[key]:
                raise ValueError(f"{clip['id']}: input manifest names another {key}")
        jobs.append((clip, Path(source["source_path"]), root / clip["id"]))
    with ProcessPoolExecutor(args.jobs) as pool:
        done = {name: (digest, size) for name, digest, size in pool.map(_derive, jobs)}
    for clip in document["clips"]:
        clip["playback_path"] = f"{clip['id']}/playback.mp4"
        clip["playback_sha256"] = done[clip["id"]][0]
        print(clip["id"], *done[clip["id"]])
    # The manifest keeps its indented layout.
    text = json.dumps(document, indent=1, sort_keys=True, ensure_ascii=False)
    args.gallery.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
