"""Private localhost event dashboard, with original-frame confirmation.

Run ``python -m tennis3d.labeling.dashboard --help``. PyAV and ffmpeg codecs
are needed for preparation and original frame decoding, never for training.
"""

import argparse
import fcntl
import hashlib
import itertools
import json
import os
import re
import secrets
import signal
import tempfile
import threading
from datetime import UTC, datetime
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from tennis3d.labeling.frames import OriginalFrames

SCHEMA = "tennis3d.manual-events/v1"
ID = re.compile(r"[A-Za-z0-9_-]{1,120}\Z")
# The page, plus the theme and typeface it shares with the Tennis3D app.
HERE = Path(__file__).parent
THEME = HERE.parent / "resources" / "theme"
STATIC = {
    "": (HERE / "dashboard.html", "text/html; charset=utf-8"),
    "dashboard.js": (HERE / "dashboard.js", "text/javascript"),
    "dashboard.css": (HERE / "dashboard.css", "text/css"),
    "theme.css": (THEME / "theme.css", "text/css"),
    "GoogleSansCode-latin.woff2": (THEME / "GoogleSansCode-latin.woff2", "font/woff2"),
    "font-license.txt": (THEME / "OFL.txt", "text/plain; charset=utf-8"),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    """Durable replacement, including the directory entry, on local POSIX disk."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(descriptor, "w") as file:
            json.dump(value, file, sort_keys=True, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        Path(temporary).unlink(missing_ok=True)


class Conflict(ValueError):
    pass


class Gallery:
    def __init__(
        self,
        inventory,
        cache,
        labels,
        frame_cache=None,
        frame_cache_bytes=OriginalFrames.DISK_BYTES,
    ):
        self.inventory_sha256 = sha(inventory)
        self.cache, self.labels = Path(cache), Path(labels)
        self.cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.labels.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.clips = {}
        self.lock = threading.RLock()
        self.token = secrets.token_urlsafe(32)
        for clip in json.loads(Path(inventory).read_bytes())["clips"]:
            cid = clip["clip_id"]
            if not ID.fullmatch(cid) or cid in self.clips:
                raise ValueError("Invalid or duplicate clip ID")
            self.validate_source(clip)
            timeline = json.loads(Path(clip["timeline_path"]).read_bytes())
            frames = timeline["frames"]
            if (
                timeline["source_sha256"] != clip["source_sha256"]
                or timeline["original_source_sha256"] != clip["original_source_sha256"]
                or timeline["image_size"] != [clip["width"], clip["height"]]
                or len(frames) != clip["frame_count"]
                or not frames
                or clip["original_source_frame_end_exclusive"]
                != clip["original_source_frame_start"] + len(frames)
                or timeline["original_source_frame_start"]
                != clip["original_source_frame_start"]
                or timeline["original_source_frame_end_exclusive"]
                != clip["original_source_frame_end_exclusive"]
            ):
                raise ValueError("Timeline/source identity mismatch")
            previous = None
            for i, frame in enumerate(frames):
                tb = frame["time_base"]
                if (
                    frame["frame_index"] != i
                    or frame["original_frame_index"]
                    != clip["original_source_frame_start"] + i
                    or type(frame["pts"]) is not int
                    or len(tb) != 2
                    or any(type(n) is not int or n <= 0 for n in tb)
                    or not re.fullmatch(r"[0-9a-f]{64}", frame["decoded_rgb_sha256"])
                ):
                    raise ValueError("Invalid source frame relationship")
                timestamp = frame["pts"] * Fraction(*tb)
                if previous is not None and timestamp <= previous:
                    raise ValueError("Non-increasing source PTS")
                previous = timestamp
            first = frames[0]["pts"] * Fraction(*frames[0]["time_base"])
            clip = dict(clip, frames=frames)
            clip["times"] = [
                float(f["pts"] * Fraction(*f["time_base"]) - first) for f in frames
            ]
            self.clips[cid] = clip
        if not self.clips:
            raise ValueError("Empty gallery")
        self.originals = OriginalFrames(
            self, frame_cache or self.cache / "original-frames", frame_cache_bytes
        )
        for cid in self.clips:
            self.read(
                cid
            )  # Refuse to silently discard an incompatible/corrupt saved file.

    @staticmethod
    def validate_source(clip):
        if sha(clip["source_path"]) != clip["source_sha256"]:
            raise ValueError("Source SHA256 mismatch")
        if sha(clip["timeline_path"]) != clip["timeline_sha256"]:
            raise ValueError("Timeline SHA256 mismatch")

    def identity(self, cid):
        clip = self.clips[cid]
        return {
            "inventory_sha256": self.inventory_sha256,
            "source_sha256": clip["source_sha256"],
            "timeline_sha256": clip["timeline_sha256"],
        }

    def preview_path(self, cid):
        clip = self.clips[cid]
        return (
            self.cache
            / f"{clip['source_sha256']}-{clip['timeline_sha256']}-vp8-v1.webm"
        )

    def prepare(self):
        """Encode each decoded input once, retain VFR cadence, verify the full join."""
        import av

        for cid, clip in self.clips.items():
            path = self.preview_path(cid)
            receipt = path.with_suffix(".json")
            if path.exists() and receipt.exists():
                record = json.loads(receipt.read_bytes())
                if (
                    record.get("identity") == self.identity(cid)
                    and record.get("preview_sha256") == sha(path)
                    and record.get("frame_count") == clip["frame_count"]
                    and self.valid_preview_times(
                        record.get("normalized_times"), clip["times"]
                    )
                ):
                    clip["playback_times"] = record["normalized_times"]
                    continue
            temporary = path.with_suffix(".partial.webm")
            try:
                with (
                    av.open(clip["source_path"]) as source,
                    av.open(str(temporary), "w") as output,
                ):
                    stream = output.add_stream("libvpx", rate=30)
                    assert isinstance(stream, av.video.stream.VideoStream)
                    stream.width, stream.height = clip["width"], clip["height"]
                    stream.pix_fmt = "yuv420p"
                    stream.bit_rate = 8_000_000
                    stream.codec_context.thread_count = 1
                    stream.codec_context.time_base = Fraction(
                        *clip["frames"][0]["time_base"]
                    )
                    stream.time_base = stream.codec_context.time_base
                    stream.options = {
                        "deadline": "realtime",
                        "cpu-used": "8",
                        "crf": "10",
                        "g": "30",
                    }
                    count = 0
                    first_time = clip["frames"][0]["pts"] * Fraction(
                        *clip["frames"][0]["time_base"]
                    )
                    for i, frame in enumerate(source.decode(video=0)):
                        if i >= clip["frame_count"]:
                            raise ValueError("Extra decoded source frame")
                        expected = clip["frames"][i]
                        self.check_decoded(frame, expected)
                        # The preview uses elapsed PTS only. Labels always use the original timeline.
                        frame = frame.reformat(format="yuv420p")
                        normalized = (
                            expected["pts"] * Fraction(*expected["time_base"])
                            - first_time
                        )
                        ticks = normalized / stream.codec_context.time_base
                        if ticks.denominator != 1:
                            raise ValueError("Unrepresentable preview timestamp")
                        frame.pts = int(ticks)
                        frame.time_base = stream.codec_context.time_base
                        for packet in stream.encode(frame):
                            output.mux(packet)
                        count += 1
                    for packet in stream.encode():
                        output.mux(packet)
                    if count != clip["frame_count"]:
                        raise ValueError("Decoded frame count mismatch")
                decoded_times = []
                with av.open(str(temporary)) as preview:
                    for frame in preview.decode(video=0):
                        if frame.time_base is None:
                            raise ValueError("Preview has no time base")
                        decoded_times.append(float(frame.pts * frame.time_base))
                if len(decoded_times) != len(clip["times"]) or any(
                    abs(a - b) > 0.00051
                    for a, b in zip(decoded_times, clip["times"], strict=True)
                ):
                    raise ValueError("Preview timestamp/frame join mismatch")
                if any(b <= a for a, b in itertools.pairwise(decoded_times)):
                    raise ValueError("Preview timestamps are not distinct")
                clip["playback_times"] = decoded_times
                os.replace(temporary, path)
                atomic_json(
                    receipt,
                    {
                        "identity": self.identity(cid),
                        "preview_sha256": sha(path),
                        "frame_count": count,
                        "normalized_times": decoded_times,
                        "mapping": "VP8/WebM decoded index equals source frame_index; elapsed source PTS rounded to milliseconds, bounded error 0.51 ms; labels use exact original PTS",
                    },
                )
                print(f"Prepared {cid}: {count} verified frames", flush=True)
            finally:
                temporary.unlink(missing_ok=True)

        self.originals.prepare()

    @staticmethod
    def valid_preview_times(times, source_times):
        return (
            isinstance(times, list)
            and len(times) == len(source_times)
            and all(
                type(t) in (int, float) and abs(t - expected) <= 0.00051
                for t, expected in zip(times, source_times, strict=True)
            )
            and all(b > a for a, b in itertools.pairwise(times))
        )

    @staticmethod
    def check_decoded(frame, expected):
        if (
            frame.pts != expected["pts"]
            or Fraction(frame.time_base) != Fraction(*expected["time_base"])
            or hashlib.sha256(frame.to_ndarray(format="rgb24").tobytes()).hexdigest()
            != expected["decoded_rgb_sha256"]
        ):
            raise ValueError("Decoded source RGB/PTS mismatch")

    def image(self, cid, index):
        return self.originals.image(cid, index)

    def read(self, cid):
        path = self.labels / f"{cid}.json"
        if not path.exists():
            return {"revision": 0, "events": [], "submitted": False}
        value = json.loads(path.read_bytes())
        if (
            value.get("schema") != SCHEMA
            or value.get("clip_id") != cid
            or value.get("identity") != self.identity(cid)
            or type(value.get("revision")) is not int
            or value["revision"] < 1
        ):
            raise ValueError("Saved labels conflict with this gallery identity")
        for event in value["events"]:
            if event != self.event(cid, event["frame_index"], event["type"]):
                raise ValueError("Saved event metadata mismatch")
        return value

    def event(self, cid, index, kind):
        clip = self.clips[cid]
        if (
            type(index) is not int
            or not 0 <= index < clip["frame_count"]
            or kind not in ("hit", "bounce")
        ):
            raise ValueError("Invalid event frame/type")
        f = clip["frames"][index]
        timestamp = f["pts"] * Fraction(*f["time_base"])
        before = clip["times"][index] - clip["times"][max(0, index - 1)]
        after = (
            clip["times"][min(index + 1, clip["frame_count"] - 1)]
            - clip["times"][index]
        )
        return {
            **f,
            "source_frame_index": index,
            "type": kind,
            "source": "manual",
            "source_time": {
                "numerator": timestamp.numerator,
                "denominator": timestamp.denominator,
            },
            "source_time_seconds": float(timestamp),
            "elapsed_clip_seconds": clip["times"][index],
            "timing": {
                "precision": "selected_decoded_frame",
                "uncertainty_frames": 1,
                "adjacent_frame_gap_seconds": {"before": before, "after": after},
                "exposure_time_known": False,
            },
        }

    def save(self, cid, payload):
        if not isinstance(payload, dict) or set(payload) != {
            "revision",
            "identity",
            "events",
            "annotator",
        }:
            raise ValueError("Invalid submission fields")
        if (
            payload["identity"] != self.identity(cid)
            or type(payload["revision"]) is not int
        ):
            raise ValueError("Submission identity/revision mismatch")
        annotator = payload["annotator"]
        if not isinstance(annotator, str) or not 1 <= len(annotator.strip()) <= 120:
            raise ValueError("Enter an annotator name")
        events = payload["events"]
        if (
            not isinstance(events, list)
            or len(events) > self.clips[cid]["frame_count"] * 2
        ):
            raise ValueError("Invalid event list")
        canonical = []
        seen = set()
        for event in events:
            if not isinstance(event, dict) or set(event) != {"frame_index", "type"}:
                raise ValueError("Invalid event fields")
            value = self.event(cid, event["frame_index"], event["type"])
            key = (value["frame_index"], value["type"])
            if key in seen:
                raise ValueError("Duplicate identical event")
            seen.add(key)
            canonical.append(value)
        with self.lock, (self.labels / ".save.lock").open("a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            current = self.read(cid)
            if current["revision"] != payload["revision"]:
                raise Conflict(
                    "A newer submission exists. Your draft is retained. Reload saved labels explicitly before resubmitting."
                )
            self.validate_source(self.clips[cid])
            clip = self.clips[cid]
            value = {
                "schema": SCHEMA,
                "clip_id": cid,
                "recording_id": clip["recording_group"],
                "identity": self.identity(cid),
                "original_source_sha256": clip["original_source_sha256"],
                "frame_count": clip["frame_count"],
                "original_source_frame_start": clip["original_source_frame_start"],
                "original_source_frame_end_exclusive": clip[
                    "original_source_frame_end_exclusive"
                ],
                "source": "manual",
                "annotator": annotator.strip(),
                "submitted": True,
                "submitted_at": datetime.now(UTC).isoformat(),
                "revision": current["revision"] + 1,
                "previous_revision": current["revision"],
                "reviewed_event_inventory": {
                    "types": ["hit", "bounce"],
                    "frame_start": 0,
                    "frame_end_exclusive": clip["frame_count"],
                    "unmarked_frames": "unannotated_for_ball_location_visibility_and_xyz",
                },
                "events": sorted(
                    canonical, key=lambda x: (x["frame_index"], x["type"])
                ),
            }
            # Keep earlier submissions as an append-only correction history.
            history = self.labels / "revisions"
            history.mkdir(exist_ok=True, mode=0o700)
            if current["revision"]:
                prior = history / f"{cid}-r{current['revision']}.json"
                if not prior.exists():
                    atomic_json(prior, current)
            atomic_json(self.labels / f"{cid}.json", value)
            return value

    def catalog(self):
        return {
            "inventory_sha256": self.inventory_sha256,
            "token": self.token,
            "clips": [
                {
                    "clip_id": cid,
                    "recording_id": c["recording_group"],
                    "frame_count": c["frame_count"],
                    "submitted": self.read(cid).get("submitted", False),
                    "revision": self.read(cid)["revision"],
                }
                for cid, c in self.clips.items()
            ],
        }

    def detail(self, cid):
        clip = self.clips[cid]
        return {
            "clip_id": cid,
            "identity": self.identity(cid),
            "frames": clip["frames"],
            "times": clip["times"],
            "playback_times": clip["playback_times"],
            "labels": self.read(cid),
        }


def handler(gallery):
    class Handler(BaseHTTPRequestHandler):
        server: ThreadingHTTPServer

        def valid_host(self):
            return self.headers.get("Host") in {
                f"localhost:{self.server.server_port}",
                f"127.0.0.1:{self.server.server_port}",
            }

        def send(self, status, data, content_type="application/json", immutable=False):
            if content_type == "application/json":
                data = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header(
                "Cache-Control",
                "private, max-age=31536000, immutable" if immutable else "no-store",
            )
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; media-src 'self'; connect-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if not self.valid_host():
                self.send(403, {"error": "Local host required"})
                return
            parts = urlparse(self.path).path.strip("/").split("/")
            try:
                if parts == ["api", "gallery"]:
                    self.send(200, gallery.catalog())
                elif (
                    len(parts) >= 3
                    and parts[:2] == ["api", "clips"]
                    and parts[2] in gallery.clips
                ):
                    cid = parts[2]
                    if len(parts) == 3:
                        self.send(200, gallery.detail(cid))
                    elif parts[3:] == ["video"]:
                        self.video(gallery.preview_path(cid))
                    elif (
                        len(parts) == 5
                        and parts[3] == "frames"
                        and parts[4].isascii()
                        and parts[4].isdigit()
                    ):
                        identity = parse_qs(urlparse(self.path).query).get("identity")
                        if identity is not None and identity != [
                            gallery.inventory_sha256
                        ]:
                            raise ValueError("Frame inventory identity mismatch")
                        self.send(
                            200,
                            gallery.image(cid, int(parts[4])),
                            "image/png",
                            immutable=identity is not None,
                        )
                    else:
                        self.send(404, {"error": "Unknown gallery resource"})
                elif len(parts) == 1 and parts[0] in STATIC:
                    path, content_type = STATIC[parts[0]]
                    self.send(200, path.read_bytes(), content_type)
                else:
                    self.send(404, {"error": "Unknown gallery resource"})
            except (ValueError, KeyError, OSError):
                self.send(
                    422,
                    {
                        "error": "Gallery data unavailable or inconsistent; no labels were changed"
                    },
                )

        def video(self, path):
            size = path.stat().st_size
            start, end = 0, size - 1
            requested = self.headers.get("Range")
            if requested:
                match = re.fullmatch(r"bytes=(\d+)-(\d*)", requested)
                if not match:
                    self.send(416, {"error": "Invalid range"})
                    return
                start = int(match[1])
                end = min(int(match[2]) if match[2] else end, end)
                if start > end:
                    self.send(416, {"error": "Invalid range"})
                    return
            self.send_response(206 if requested else 200)
            self.send_header("Content-Type", "video/webm")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if requested:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with path.open("rb") as file:
                    file.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        block = file.read(min(remaining, 256 * 1024))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            origin = self.headers.get("Origin")
            expected_origins = {
                f"http://localhost:{self.server.server_port}",
                f"http://127.0.0.1:{self.server.server_port}",
            }
            if (
                not self.valid_host()
                or origin not in expected_origins
                or self.headers.get("X-Label-Token") != gallery.token
                or self.headers.get("Content-Type") != "application/json"
            ):
                self.send(403, {"error": "Same-origin dashboard submission required"})
                return
            parts = urlparse(self.path).path.strip("/").split("/")
            if (
                len(parts) != 4
                or parts[:2] != ["api", "clips"]
                or parts[2] not in gallery.clips
                or parts[3] != "labels"
            ):
                self.send(404, {"error": "Unknown clip"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 * 1024:
                    raise ValueError("Invalid submission size")
                payload = json.loads(self.rfile.read(length))
                result = gallery.save(parts[2], payload)
                self.send(200, {"saved": True, "labels": result})
            except Conflict as error:
                self.send(409, {"error": str(error)})
            except (ValueError, KeyError, TypeError):
                self.send(
                    422,
                    {"error": "Invalid submission or source identity; draft not saved"},
                )
            except OSError:
                self.send(
                    500,
                    {
                        "error": "Disk persistence failed. Your draft is retained; retry after checking disk access."
                    },
                )

    return Handler


class LabelServer(ThreadingHTTPServer):
    # server_close joins successful in-flight submissions before process exit.
    daemon_threads = False

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(30)
        return connection, address


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--frame-cache", type=Path)
    parser.add_argument(
        "--frame-cache-bytes", type=int, default=OriginalFrames.DISK_BYTES
    )
    args = parser.parse_args()
    gallery = Gallery(
        args.inventory,
        args.cache,
        args.labels,
        args.frame_cache,
        args.frame_cache_bytes,
    )
    gallery.prepare()
    if not args.prepare_only:
        server = LabelServer(("127.0.0.1", args.port), handler(gallery))
        stopping = threading.Event()

        def stop(signum, frame):
            if not stopping.is_set():
                stopping.set()
                threading.Thread(target=server.shutdown).start()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        print(f"Labeling dashboard: http://localhost:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
