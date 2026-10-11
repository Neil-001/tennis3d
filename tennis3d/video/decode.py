"""Stream PyAV presentation order. Reject missing/non-increasing PTS, never invent FPS."""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


@dataclass
class DecodedFrame:
    index: int
    pts: int
    time_base: Fraction
    source_time: Fraction
    t_s: float
    bgr: np.ndarray


def decode(path: Path):
    origin = previous = None
    size = None
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("Input contains no video stream")
        stream = container.streams.video[0]
        stream.thread_count = 2
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"Missing presentation timestamp at frame {index}")
            timestamp = frame.pts * frame.time_base
            if previous is not None and timestamp <= previous:
                raise ValueError(
                    f"Non-increasing presentation timestamp at frame {index}"
                )
            if origin is None:
                origin = timestamp
                size = (frame.width, frame.height)
            if size != (frame.width, frame.height):
                raise ValueError("Video dimensions changed within the shot")
            yield DecodedFrame(
                index,
                frame.pts,
                frame.time_base,
                timestamp,
                float(timestamp - origin),
                frame.to_ndarray(format="bgr24"),
            )
            previous = timestamp
