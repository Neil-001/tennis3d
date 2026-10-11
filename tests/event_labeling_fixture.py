"""Small synthetic VFR gallery. Never use these events as human annotations."""

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

from tennis3d.labeling.dashboard import Gallery, atomic_json, sha


def make_gallery(root):
    import av
    import numpy as np

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    clips = []
    for number in range(2):
        path = root / f"fixture{number}.mp4"
        with av.open(str(path), "w") as output:
            stream = output.add_stream("libx264rgb", rate=30)
            assert isinstance(stream, av.video.stream.VideoStream)
            stream.width, stream.height, stream.pix_fmt = 96, 64, "rgb24"
            stream.codec_context.time_base = Fraction(1, 1000)
            stream.time_base = Fraction(1, 1000)
            stream.options = {"crf": "0", "preset": "ultrafast", "bf": "0"}
            pts = 4000
            for index in range(40):
                rgb = np.zeros((64, 96, 3), dtype=np.uint8)
                rgb[:, :, 0] = index * 5
                rgb[10:20, (index * 2) % 80 : (index * 2) % 80 + 10, 1] = 255
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                frame.pts, frame.time_base = pts, Fraction(1, 1000)
                pts += 40 if index % 2 else 30
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        frames = []
        with av.open(str(path)) as source:
            for i, frame in enumerate(source.decode(video=0)):
                assert frame.time_base is not None
                frames.append(
                    {
                        "frame_index": i,
                        "original_frame_index": i + 71,
                        "pts": frame.pts,
                        "time_base": [
                            frame.time_base.numerator,
                            frame.time_base.denominator,
                        ],
                        "decoded_rgb_sha256": hashlib.sha256(
                            frame.to_ndarray(format="rgb24").tobytes()
                        ).hexdigest(),
                    }
                )
        timeline = root / f"fixture{number}.timeline.json"
        source_hash = sha(path)
        atomic_json(
            timeline,
            {
                "frames": frames,
                "image_size": [96, 64],
                "source_sha256": source_hash,
                "original_source_sha256": "a" * 64,
                "original_source_frame_start": 71,
                "original_source_frame_end_exclusive": 111,
            },
        )
        clips.append(
            {
                "clip_id": f"fixture{number}",
                "recording_group": f"synthetic_match_{number}",
                "source_path": str(path),
                "source_sha256": source_hash,
                "timeline_path": str(timeline),
                "timeline_sha256": sha(timeline),
                "original_source_sha256": "a" * 64,
                "original_source_frame_start": 71,
                "original_source_frame_end_exclusive": 111,
                "frame_count": 40,
                "width": 96,
                "height": 64,
            }
        )
    inventory = root / "inventory.json"
    atomic_json(inventory, {"clips": clips})
    return inventory


if __name__ == "__main__":
    root = Path(sys.argv[1])
    inventory = make_gallery(root)
    gallery = Gallery(inventory, root / "previews", root / "labels")
    gallery.prepare()
    print(json.dumps({"inventory": str(inventory)}))
