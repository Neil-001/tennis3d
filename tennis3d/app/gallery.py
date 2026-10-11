"""Precomputed examples: existing results and videos, shown without rerunning.

A gallery manifest lists clips with their source and result hashes. Results are
verified against the manifest before they are served. Videos that browsers
cannot decode get a cached playback copy keyed by the source hash; the manifest
and its original files are never modified.
"""

import hashlib
import re
import threading
from pathlib import Path

from tennis3d.artifacts import atomic_file, file_sha256, read_json

CLIP_ID = re.compile(r"[A-Za-z0-9_.-]{1,100}")


class Gallery:
    def __init__(self, manifest: Path | None, cache: Path):
        self.cache = cache
        self.clips = {}
        self.errors = {}
        self.digests = {}
        self.lock = threading.Lock()
        if manifest is None:
            return
        manifest = Path(manifest).resolve()
        document = read_json(manifest)
        if document.get("schema_version") != 1:
            raise ValueError("Unsupported gallery manifest version")
        for clip in document["clips"]:
            if (
                not CLIP_ID.fullmatch(clip["id"])
                or clip["id"] in (".", "..")
                or clip["id"] in self.clips
            ):
                raise ValueError(f"Invalid or repeated gallery clip id: {clip['id']}")
            for key in ("source_path", "result_path", "playback_path", "timeline_path"):
                if clip.get(key):
                    clip[key] = str((manifest.parent / clip[key]).resolve())
            self.clips[clip["id"]] = clip

    def describe(self) -> list[dict]:
        return [
            {
                "id": clip["id"],
                "title": clip.get("title") or clip["id"],
                "has_result": bool(clip.get("result_path")),
                "has_timeline": bool(clip.get("timeline_path")),
                "video": self.video_state(clip["id"]),
                "media_offset_s": self.offset(clip["id"]),
                # The manifest's coverage note stays in the manifest; the result shows it.
                "note": self.errors.get(clip["id"]),
                "rights": clip.get("rights", {"redistribution": "unknown"}),
            }
            for clip in self.clips.values()
        ]

    def restore_line_endings(self) -> list[Path]:
        """Undo a Git checkout that wrote results or timelines with CRLF line endings.

        A file is replaced only when its bytes with every CRLF turned back to LF
        have the recorded SHA-256. Any other difference is left for the hash
        checks to refuse.
        """
        restored = []
        for clip in self.clips.values():
            for kind in ("result", "timeline"):
                expected = clip.get(f"{kind}_sha256")
                if not clip.get(f"{kind}_path") or not expected:
                    continue
                path = Path(clip[f"{kind}_path"])
                try:
                    changed = path.read_bytes()
                    original = changed.replace(b"\r\n", b"\n")
                    if (
                        original == changed
                        or hashlib.sha256(original).hexdigest() != expected
                    ):
                        continue
                    mode = path.stat().st_mode & 0o7777
                    with atomic_file(path, overwrite=True) as stream:
                        stream.write(original)
                    path.chmod(mode)
                except OSError:
                    continue  # Unreadable or read-only: the hash check reports it.
                restored.append(path)
        return restored

    def result(self, clip_id: str) -> Path:
        clip = self.clips[clip_id]
        if not clip.get("result_path"):
            raise KeyError(clip_id)
        path = Path(clip["result_path"])
        if file_sha256(path) != clip["result_sha256"]:
            raise ValueError("Example result does not match its recorded SHA-256")
        return path

    def timeline(self, clip_id: str) -> Path:
        clip = self.clips[clip_id]
        if not clip.get("timeline_path"):
            raise KeyError(clip_id)
        path = Path(clip["timeline_path"])
        if file_sha256(path) != clip.get("timeline_sha256"):
            raise ValueError("Example timeline does not match its recorded SHA-256")
        if read_json(path).get("source_sha256") != clip.get("source_sha256"):
            raise ValueError("Example timeline belongs to a different source")
        return path

    def _copy(self, clip) -> Path | None:
        if clip.get("playback_path"):
            return Path(clip["playback_path"])
        if not clip.get("source_sha256"):
            return None
        return self.cache / f"{clip['source_sha256']}.mp4"

    def _verified(self, clip, copy: Path) -> bool:
        """The copy's bytes match its record, and the record names this source."""
        if clip.get("playback_path"):
            expected = clip.get("playback_sha256")
        else:
            receipt = read_json(copy.with_suffix(".json"))
            expected = (
                receipt["playback_sha256"]
                if receipt["source_sha256"] == clip["source_sha256"]
                else None
            )
        # Hash each file once per version; every clip compares its own expectation.
        stat = copy.stat()
        key = (str(copy), stat.st_size, stat.st_mtime_ns)
        if key not in self.digests:
            self.digests[key] = file_sha256(copy)
        return expected is not None and self.digests[key] == expected

    def video_state(self, clip_id: str) -> str:
        clip = self.clips[clip_id]
        copy = self._copy(clip)
        if clip_id in self.errors:
            return "unavailable"
        if copy is None:
            return "unavailable"  # A result-only example.
        try:
            if copy.is_file() and self._verified(clip, copy):
                return "ready"
        except (OSError, ValueError, KeyError):
            pass
        source = clip.get("source_path")
        if clip.get("playback_path") or not source:
            return "unavailable"
        return "preparing" if Path(source).is_file() else "unavailable"

    def video(self, clip_id: str) -> Path:
        copy = self._copy(self.clips[clip_id])
        if copy is None or self.video_state(clip_id) != "ready":
            raise KeyError(clip_id)
        return copy

    def offset(self, clip_id: str) -> float:
        return float(self.clips[clip_id].get("media_offset_s") or 0.0)

    def prepare(self, progress=None) -> dict:
        """Create missing playback copies. Returns the state of every clip."""
        from tennis3d.app.media import playback_copy

        with self.lock:
            for clip_id, clip in self.clips.items():
                if self.video_state(clip_id) != "preparing":
                    continue
                try:
                    source = Path(clip["source_path"])
                    if file_sha256(source) != clip["source_sha256"]:
                        raise ValueError("source video differs from the manifest")
                    playback_copy(
                        source,
                        self.cache / f"{clip['source_sha256']}.mp4",
                        progress=progress,
                    )
                except Exception as error:  # noqa: BLE001 - one bad clip must not hide the rest
                    self.errors[clip_id] = f"Video unavailable: {error}"
        return {clip_id: self.video_state(clip_id) for clip_id in self.clips}
