"""Bounded, lossless source-frame cache, built with one sequential decode per clip."""

import fcntl
import hashlib
import json
import os
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path


class OriginalFrames:
    MEMORY_BYTES = 64 * 1024 * 1024
    DISK_BYTES = 16 * 1024 * 1024 * 1024

    def __init__(self, gallery, root, disk_bytes=DISK_BYTES):
        self.gallery, self.root, self.disk_bytes = gallery, Path(root), disk_bytes
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.records = {}
        self.images = OrderedDict()
        self.memory_bytes = 0
        self.lock = threading.Lock()
        self.readers = threading.BoundedSemaphore(2)

    def directory(self, clip):
        return (
            self.root / f"{clip['source_sha256']}-{clip['timeline_sha256']}-rgbpng-v1"
        )

    @staticmethod
    def write(path, data):
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".frame-")
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def valid(self, record, clip, directory):
        frames = record.get("frames")
        return (
            record.get("source_sha256") == clip["source_sha256"]
            and record.get("timeline_sha256") == clip["timeline_sha256"]
            and isinstance(frames, list)
            and len(frames) == clip["frame_count"]
            and all(
                isinstance(f, dict)
                and f.get("rgb_sha256") == expected["decoded_rgb_sha256"]
                and isinstance(f.get("png_sha256"), str)
                and len(f["png_sha256"]) == 64
                and type(f.get("bytes")) is int
                and f["bytes"] > 0
                and (directory / f"{i}.png").is_file()
                and (directory / f"{i}.png").stat().st_size == f["bytes"]
                for i, (f, expected) in enumerate(zip(frames, clip["frames"]))
            )
        )

    def prepare(self):
        import av

        with (self.root / ".prepare.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
            if used > self.disk_bytes:
                raise ValueError("Original-frame cache exceeds disk budget")
            for cid, clip in self.gallery.clips.items():
                directory = self.directory(clip)
                directory.mkdir(exist_ok=True, mode=0o700)
                receipt = directory / "receipt.json"
                record = json.loads(receipt.read_bytes()) if receipt.exists() else {}
                if self.valid(record, clip, directory):
                    self.records[cid] = record["frames"]
                    continue
                frames = []
                with av.open(clip["source_path"]) as source:
                    source.streams.video[0].codec_context.thread_count = 1
                    codec = av.CodecContext.create("png", "w")
                    codec.width, codec.height = clip["width"], clip["height"]
                    codec.pix_fmt = "rgb24"
                    codec.thread_count = 1
                    codec.options = {"compression_level": "1"}
                    for i, frame in enumerate(source.decode(video=0)):
                        if i >= clip["frame_count"]:
                            raise ValueError("Extra decoded source frame")
                        expected = clip["frames"][i]
                        self.gallery.check_decoded(frame, expected)
                        data = b"".join(
                            bytes(p)
                            for p in codec.encode(frame.reformat(format="rgb24"))
                        )
                        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                            raise ValueError("PNG encoding failed")
                        path = directory / f"{i}.png"
                        prior_bytes = path.stat().st_size if path.exists() else 0
                        if used + len(data) > self.disk_bytes:
                            raise ValueError(
                                "Original-frame cache disk budget exhausted"
                            )
                        self.write(path, data)
                        used += len(data) - prior_bytes
                        frames.append(
                            {
                                "rgb_sha256": expected["decoded_rgb_sha256"],
                                "png_sha256": hashlib.sha256(data).hexdigest(),
                                "bytes": len(data),
                            }
                        )
                    if codec.encode(None) or len(frames) != clip["frame_count"]:
                        raise ValueError("Original-frame cache count mismatch")
                self.gallery.validate_source(clip)
                data = json.dumps(
                    {
                        "source_sha256": clip["source_sha256"],
                        "timeline_sha256": clip["timeline_sha256"],
                        "frames": frames,
                    },
                    sort_keys=True,
                ).encode()
                prior_bytes = receipt.stat().st_size if receipt.exists() else 0
                if used + len(data) > self.disk_bytes:
                    raise ValueError("Original-frame cache disk budget exhausted")
                self.write(receipt, data)
                descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                used += len(data) - prior_bytes
                self.records[cid] = frames
                print(
                    f"Cached {cid}: {len(frames)} lossless original frames", flush=True
                )

    def image(self, cid, index):
        clip = self.gallery.clips[cid]
        if type(index) is not int or not 0 <= index < clip["frame_count"]:
            raise ValueError("Frame out of range")
        key = (cid, index)
        with self.lock:
            if key in self.images:
                self.images.move_to_end(key)
                return self.images[key]
        with self.readers:
            record = self.records[cid][index]
            data = (self.directory(clip) / f"{index}.png").read_bytes()
            if hashlib.sha256(data).hexdigest() != record["png_sha256"]:
                raise ValueError("Original-frame cache hash mismatch")
            with self.lock:
                if key not in self.images and len(data) <= self.MEMORY_BYTES:
                    self.images[key] = data
                    self.memory_bytes += len(data)
                    while self.memory_bytes > self.MEMORY_BYTES:
                        _, evicted = self.images.popitem(last=False)
                        self.memory_bytes -= len(evicted)
            return data
