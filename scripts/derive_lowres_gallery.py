"""Derive paired low-resolution copies of a frozen gallery, frame for frame.

No network, detector, or lifter is used. ``derive`` checks every parent frame
against its frozen RGB hash and integer PTS, resizes it with an exact integer
area average, and writes lossless clips that keep each parent frame index, PTS
and time base. ``reproduce`` rebuilds the clips from the parents and requires
the frozen bytes. A wrong parent or an incomplete derivation writes no
inventory.
"""

import argparse
import hashlib
import itertools
import platform
import sys
import tempfile
import time
from datetime import UTC, datetime
from importlib.metadata import distribution
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis3d.artifacts import file_sha256, read_json, write_json
from tennis3d.evaluation.inventory import Clip, Inventory, read_inventory, verify_clip

RESIZE_RULE = (
    "Exact area average in integer arithmetic. Source and target sizes must "
    "have the same aspect ratio. Each target pixel is the mean of the source "
    "RGB values over its footprint, with partly covered source pixels weighted "
    "by the covered fraction, computed per channel as an exact integer sum "
    "over the common denominator and rounded half up. No colour conversion, "
    "sharpening, cropping or padding."
)
# Read once at start, so a later edit of this file cannot misname the code that ran.
TOOL_SHA256 = file_sha256(__file__)
ENCODING = (
    "lossless RGB libx264rgb CRF0 ultrafast, one encoder thread, "
    "movie_timescale=parent timebase denominator, use_editlist=1, "
    "avoid_negative_ts=disabled"
)


def _axis_sums(values, target: int, axis: int):
    """Sum along one axis over each target footprint, in units of 1/q pixel."""
    source = values.shape[axis]
    divisor = np.gcd(source, target)
    p, q = source // divisor, target // divisor  # footprint p/q source pixels
    values = np.moveaxis(values, axis, 0).astype(np.uint64)
    total = np.concatenate([np.zeros_like(values[:1]), np.cumsum(values, axis=0)])
    edges = np.arange(target + 1) * p  # in 1/q source pixels
    whole, part = edges // q, edges % q
    padded = np.concatenate([values, np.zeros_like(values[:1])])
    shape = (-1,) + (1,) * (values.ndim - 1)
    integral = q * total[whole] + part.astype(np.uint64).reshape(shape) * padded[whole]
    return np.moveaxis(integral[1:] - integral[:-1], 0, axis), p


def area_resize(rgb, width: int, height: int):
    """Resize H x W x 3 uint8 by the declared exact area average."""
    source_height, source_width = rgb.shape[:2]
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("resize requires an 8-bit RGB image")
    if source_width * height != source_height * width:
        raise ValueError("target size does not preserve the source aspect ratio")
    if width > source_width:
        raise ValueError("target size must not exceed the source size")
    rows, row_weight = _axis_sums(rgb, height, 0)
    sums, column_weight = _axis_sums(rows, width, 1)
    weight = np.uint64(row_weight * column_weight)
    return ((2 * sums + weight) // (2 * weight)).astype(np.uint8)


def rgb_sha256(rgb) -> str:
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def parent_frames(parent: Clip, timeline):
    """Yield each parent frame after checking its frozen identity."""
    rows = timeline["frames"]
    count = 0
    with av.open(str(parent.source_path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        rate = stream.average_rate
        for frame in container.decode(stream):
            if count >= len(rows):
                raise ValueError("parent clip has extra frames")
            row = rows[count]
            rgb = frame.to_ndarray(format="rgb24")
            if (
                frame.pts != row["pts"]
                or frame.time_base is None
                or [frame.time_base.numerator, frame.time_base.denominator]
                != row["time_base"]
                or [frame.width, frame.height] != [parent.width, parent.height]
                or rgb_sha256(rgb) != row["decoded_rgb_sha256"]
            ):
                raise ValueError("parent frame pixels or exact timestamp differ")
            count += 1
            yield row, frame.time_base, rgb, rate
    if count != len(rows):
        raise ValueError("parent clip omits frozen frames")


def encode(parent: Clip, timeline, target: Path, size):
    """Write the resized clip; return its rows with the expected RGB hashes."""
    width, height = size
    frames = parent_frames(parent, timeline)
    first = next(frames, None)
    if first is None:
        raise ValueError("parent clip has no frames")
    _, basis, _, rate = first
    rows = []
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
        stream.width, stream.height, stream.pix_fmt = width, height, "rgb24"
        stream.time_base = stream.codec_context.time_base = basis
        stream.codec_context.thread_count = 1
        stream.options = {"crf": "0", "preset": "ultrafast"}
        for row, basis, rgb, _ in itertools.chain([first], frames):
            small = area_resize(rgb, width, height)
            frame = av.VideoFrame.from_ndarray(small, format="rgb24")
            frame.pts, frame.time_base = row["pts"], basis
            destination.mux(stream.encode(frame))
            rows.append(
                {
                    **{
                        key: value for key, value in row.items() if key != "frame_index"
                    },
                    "frame_index": len(rows),
                    "parent_frame_index": row["frame_index"],
                    "parent_decoded_rgb_sha256": row["decoded_rgb_sha256"],
                    "decoded_rgb_sha256": rgb_sha256(small),
                }
            )
        destination.mux(stream.encode())
    return rows


def checked_derived(target: Path, rows, size):
    """Decode the derived clip and require every expected row."""
    count = 0
    with av.open(str(target)) as container:
        container.streams.video[0].codec_context.thread_count = 1
        for frame in container.decode(video=0):
            if count >= len(rows):
                raise ValueError("derived clip has extra frames")
            row = rows[count]
            if (
                frame.pts != row["pts"]
                or frame.time_base is None
                or [frame.time_base.numerator, frame.time_base.denominator]
                != row["time_base"]
                or [frame.width, frame.height] != list(size)
                or rgb_sha256(frame.to_ndarray(format="rgb24"))
                != row["decoded_rgb_sha256"]
            ):
                raise ValueError("derived RGB or integer PTS/time base differs")
            count += 1
    if count != len(rows):
        raise ValueError("derived clip omits parent frames")


def checked_parent(parent: Clip):
    if not parent.eligible:
        raise ValueError(f"parent clip is not eligible: {parent.clip_id}")
    return verify_clip(parent)


def derive_clip(parent: Clip, target: Path, size):
    """Derive one clip into ``target`` and return its timeline without hashes."""
    timeline = checked_parent(parent)
    rows = encode(parent, timeline, target, size)
    checked_derived(target, rows, size)
    return {
        "source_sha256": file_sha256(target),
        "parent_clip_id": parent.clip_id,
        "parent_source_sha256": parent.source_sha256,
        "parent_timeline_sha256": parent.timeline_sha256,
        "parent_image_size": [parent.width, parent.height],
        "original_source_sha256": parent.original_source_sha256,
        "original_source_frame_start": parent.original_source_frame_start,
        "original_source_frame_end_exclusive": (
            parent.original_source_frame_end_exclusive
        ),
        "image_size": list(size),
        "resize_rule": RESIZE_RULE,
        "index_contract": "frame_index and parent_frame_index count frames of the "
        "derived and parent clips; original_frame_index is the parent's index in "
        "the complete downloaded source file. decoded_rgb_sha256 is the resized "
        "image; parent_decoded_rgb_sha256 is the parent image it was made from.",
        "frames": rows,
    }


def build_identity():
    """Exact identities of this tool and of the decoder and encoder binaries."""
    package = distribution("av")
    libraries = {
        Path(str(file)).name: file_sha256(str(package.locate_file(file)))
        for file in package.files or []
        if Path(str(file)).parts[0] == "av.libs"
    }
    if not any(name.startswith("libx264") for name in libraries):
        raise ValueError("bundled PyAV libraries, including x264, were not found")
    return {
        "tool_sha256": TOOL_SHA256,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pyav": av.__version__,
        "pyav_record_sha256": hashlib.sha256(
            (package.read_text("RECORD") or "").encode()
        ).hexdigest(),
        "linked_libraries": av.library_versions,
        "bundled_library_sha256": dict(sorted(libraries.items())),
    }


def checked_pair(clip: Clip, parent: Clip, timeline):
    """Require the derived clip record to agree with its parent and timeline."""
    if (
        clip.source_revision != binding(parent)
        or clip.frame_count != parent.frame_count
        or clip.frame_count != len(timeline["frames"])
        or clip.source_frame_start != parent.source_frame_start
        or [clip.width, clip.height] != timeline["image_size"]
        or clip.source_sha256 != timeline["source_sha256"]
        or any(
            getattr(clip, key) != getattr(parent, key)
            for key in (
                "original_source_sha256",
                "original_source_bytes",
                "original_source_frame_start",
                "original_source_frame_end_exclusive",
                "recording_group",
                "source_url",
                "time_basis",
            )
        )
        or not clip.eligible
    ):
        raise ValueError(
            f"derived clip record disagrees with its parent: {clip.clip_id}"
        )


def binding(parent: Clip) -> str:
    return f"{parent.clip_id}:{parent.source_sha256}:{parent.timeline_sha256}"


def derive(parent_path: Path, output: Path, size, config_name: str, suffix: str):
    parents = read_inventory(parent_path)
    if any(not clip.eligible for clip in parents.clips):
        raise ValueError("every parent clip must be eligible; none may be dropped")
    (output / "prepared").mkdir(parents=True)
    (output / "timelines").mkdir()
    clips, portable, pairs = [], [], []
    for parent in parents.clips:
        started = time.monotonic()
        clip_id = parent.clip_id + suffix
        target = output / "prepared" / f"{clip_id}.mp4"
        timeline_path = output / "timelines" / f"{clip_id}.json"
        with tempfile.TemporaryDirectory(prefix=".derive-", dir=output) as temporary:
            candidate = Path(temporary) / "clip.mp4"
            timeline = derive_clip(parent, candidate, size)
            candidate.rename(target)
        write_json(timeline_path, timeline)
        values = parent.model_dump()
        values.update(
            clip_id=clip_id,
            source_revision=binding(parent),
            source_sha256=timeline["source_sha256"],
            source_bytes=target.stat().st_size,
            source_path=str(target.resolve()),
            width=size[0],
            height=size[1],
            timeline_path=str(timeline_path.resolve()),
            timeline_sha256=file_sha256(timeline_path),
            preparation_note=(
                f"Derived {size[0]}x{size[1]} copy of every frame of parent clip "
                f"{parent.clip_id} (source_revision holds its clip ID, prepared "
                f"SHA-256 and timeline SHA-256). {RESIZE_RULE} Encoding: "
                f"{ENCODING}. Every derived frame keeps the parent integer PTS "
                "and time base."
            ),
            category_evidence={
                key: f"Carried from the parent clip's source review: {note}"
                for key, note in parent.category_evidence.items()
            },
        )
        clip = Clip.model_validate(values)
        verify_clip(clip)
        checked_pair(clip, parent, timeline)
        clips.append(clip.model_dump())
        portable.append(
            {
                **clip.model_dump(),
                "source_path": f"data/{config_name}/prepared/{clip_id}.mp4",
                "timeline_path": f"configs/{config_name}/timelines/{clip_id}.json",
            }
        )
        pairs.append(
            {
                "clip_id": clip_id,
                "parent_clip_id": parent.clip_id,
                "parent_source_sha256": parent.source_sha256,
                "parent_timeline_sha256": parent.timeline_sha256,
                "source_sha256": clip.source_sha256,
                "source_bytes": clip.source_bytes,
                "timeline_sha256": clip.timeline_sha256,
                "frame_count": clip.frame_count,
                "preparation_wall_s": round(time.monotonic() - started, 3),
            }
        )
    common = {
        "frozen_utc": datetime.now(UTC).isoformat(),
        "intended_role": "resolution_stress",
        "selection_rule": (
            f"Every clip of the parent inventory, derived frame for frame at "
            f"{size[0]}x{size[1]}. No clip, frame or range was selected, dropped "
            "or altered, and no model output was read. Parent selection rule: "
            + parents.selection_rule
        ),
        "excluded_development_recording_groups": (
            parents.excluded_development_recording_groups
        ),
        "excluded_development_source_sha256": (
            parents.excluded_development_source_sha256
        ),
        "exclusion_evidence": parents.exclusion_evidence,
        "limitations": [
            (
                f"These are resized copies of the parent clips, not independent "
                f"footage or camera-native {size[0]}x{size[1]} video. They share every "
                "limitation of the parent inventory."
            ),
            (
                "Resized images do not have the parent RGB. Labels or outputs made on "
                "the parent join by frame index and integer PTS, with image "
                "coordinates scaled by the size ratio."
            ),
            *parents.limitations,
        ],
    }
    private = Inventory.model_validate({**common, "clips": clips})
    public = Inventory.model_validate({**common, "clips": portable})
    write_json(output / "inventory.private.json", private.model_dump())
    write_json(output / "inventory.json", public.model_dump())
    write_json(
        output / "pairing.json",
        {
            "schema_version": 1,
            "relation": "paired low-resolution copies, one per parent clip and frame",
            "parent_inventory_sha256": file_sha256(parent_path),
            "parent_clip_identity_sha256": identity(parents),
            "image_size": list(size),
            "resize_rule": RESIZE_RULE,
            "encoding": ENCODING,
            **build_identity(),
            "source_only": True,
            "frames": sum(pair["frame_count"] for pair in pairs),
            "derived_bytes": sum(pair["source_bytes"] for pair in pairs),
            "preparation_wall_s": round(
                sum(pair["preparation_wall_s"] for pair in pairs), 3
            ),
            "timing_scope": "single-process CPU wall time per clip for parent "
            "verification, decode, resize, lossless encode and derived check; "
            "not model or GPU time",
            "clips": pairs,
        },
    )
    return private.summary()


def identity(inventory: Inventory) -> str:
    """Hash of the parent clip identities, independent of paths and wording."""
    rows = [
        [clip.clip_id, clip.source_sha256, clip.timeline_sha256, clip.frame_count]
        for clip in inventory.clips
    ]
    return hashlib.sha256(repr(rows).encode()).hexdigest()


def reproduce(inventory_path: Path, parent_path: Path, output: Path):
    """Rebuild every derived clip from its parent and require the frozen bytes."""
    derived, parents = read_inventory(inventory_path), read_inventory(parent_path)
    by_binding = {binding(parent): parent for parent in parents.clips}
    if len(derived.clips) != len(parents.clips) or {
        clip.source_revision for clip in derived.clips
    } != set(by_binding):
        raise ValueError("derived inventory is not paired with this parent inventory")
    output.mkdir(parents=True)
    records = []
    for clip in derived.clips:
        parent = by_binding[clip.source_revision]
        if file_sha256(clip.timeline_path) != clip.timeline_sha256:
            raise ValueError("frozen derived timeline changed")
        frozen = read_json(clip.timeline_path)
        target = output / f"{clip.clip_id}.mp4"
        checked_pair(clip, parent, frozen)
        timeline = derive_clip(parent, target, (clip.width, clip.height))
        if (
            timeline != frozen
            or file_sha256(target) != clip.source_sha256
            or target.stat().st_size != clip.source_bytes
        ):
            raise ValueError(f"reproduction differs from the freeze: {clip.clip_id}")
        records.append(
            {
                "clip_id": clip.clip_id,
                "parent_clip_id": parent.clip_id,
                "reproduced_sha256": clip.source_sha256,
                "frames": clip.frame_count,
                "every_row_and_byte_matches_freeze": True,
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    step = commands.add_parser("derive")
    step.add_argument("--parent-inventory", type=Path, required=True)
    step.add_argument("--output", type=Path, required=True)
    step.add_argument("--config-name", required=True)
    step.add_argument("--width", type=int, default=256)
    step.add_argument("--height", type=int, default=144)
    step.add_argument("--suffix", default="-lowres")
    step = commands.add_parser("reproduce")
    step.add_argument("--inventory", type=Path, required=True)
    step.add_argument("--parent-inventory", type=Path, required=True)
    step.add_argument("--output", type=Path, required=True)
    step.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output root already exists")
    if args.command == "derive":
        print(
            derive(
                args.parent_inventory,
                args.output,
                (args.width, args.height),
                args.config_name,
                args.suffix,
            )
        )
        return
    if args.receipt.exists():
        parser.error("receipt already exists")
    records = reproduce(args.inventory, args.parent_inventory, args.output)
    write_json(
        args.receipt,
        {
            "inventory_sha256": file_sha256(args.inventory),
            "parent_inventory_sha256": file_sha256(args.parent_inventory),
            **build_identity(),
            "source_only": True,
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
