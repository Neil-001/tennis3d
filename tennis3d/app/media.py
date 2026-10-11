"""Cached browser playback copies with explicit original source-time mapping."""

import os
from fractions import Fraction
from pathlib import Path

from tennis3d.artifacts import file_sha256, read_json, write_json
from tennis3d.inference.progress import report


def playback_copy(source: Path, output: Path, *, progress=None):
    """Encode an H.264 browser copy whose timestamps equal source elapsed time.

    The receipt maps every copied frame to its source PTS. Reconstruction always
    reads the original file; the copy is for display only.
    """
    import av

    receipt = output.with_suffix(".json")
    source_hash = file_sha256(source)
    if output.is_file() and receipt.is_file():
        saved = read_json(receipt)
        if saved["source_sha256"] == source_hash and saved[
            "playback_sha256"
        ] == file_sha256(output):
            return saved
    receipt.unlink(missing_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    mapping = []
    try:
        with (
            av.open(str(source)) as original,
            av.open(str(temporary), mode="w", format="mp4") as encoded,
        ):
            video = original.streams.video[0]
            time_base = video.time_base
            if time_base is None:
                raise ValueError("Source video has no time base")
            stream = encoded.add_stream(
                "libx264", rate=video.average_rate or Fraction(25)
            )
            stream.width, stream.height = video.width // 2 * 2, video.height // 2 * 2
            stream.pix_fmt = "yuv420p"
            # No B-frames: decode order equals presentation order from time zero.
            stream.options = {
                "crf": "20",
                "preset": "veryfast",
                "threads": "2",
                "bf": "0",
            }
            stream.codec_context.time_base = time_base
            origin = None
            for index, frame in enumerate(original.decode(video)):
                report(progress, "playback", index, video.frames or None)
                if frame.pts is None or frame.time_base is None:
                    raise ValueError("Source video has a frame without a timestamp")
                source_time = frame.pts * frame.time_base
                if origin is None:
                    origin = source_time
                elapsed = source_time - origin
                pts = elapsed / time_base
                if pts.denominator != 1:
                    raise ValueError("Playback timebase cannot retain source cadence")
                mapping.append(
                    {
                        "frame_index": index,
                        "source_pts": frame.pts,
                        "source_time_base": [
                            frame.time_base.numerator,
                            frame.time_base.denominator,
                        ],
                        "elapsed_s": float(elapsed),
                        "playback_pts": int(pts),
                    }
                )
                frame = frame.reformat(
                    width=stream.width, height=stream.height, format="yuv420p"
                )
                frame.pts, frame.time_base = int(pts), time_base
                encoded.mux(stream.encode(frame))
            encoded.mux(stream.encode())
        if not mapping:
            raise ValueError("Video contains no decodable frames")
        if file_sha256(source) != source_hash:
            raise ValueError("Source video changed during playback preparation")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    saved = {
        "schema_version": "1.0",
        "kind": "playback_proxy",
        "source_sha256": source_hash,
        "playback_sha256": file_sha256(output),
        "media_offset_s": 0.0,
        "pixel_identity": "Browser copy; original pixels are used for inference",
        "frames": mapping,
    }
    write_json(receipt, saved, overwrite=True)
    return saved
