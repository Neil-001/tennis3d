"""Durable JSONL events. Readers never repair or rewrite the history."""

import fcntl
import hashlib
import math
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from tennis3d.artifacts import _loads, canonical_json, read_json, write_json

OUTCOMES = {"kept", "discarded", "crashed", "timed-out", "interrupted", "checks-failed"}
KINDS = {"segment", "started", "progress", "finished", "control", "tail-recovery"}


@contextmanager
def lock(path: Path, *, blocking: bool = True) -> Iterator[int]:
    """Linux/macOS advisory lock; an inherited descriptor also protects a child."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise ValueError("another experiment is still active in this run") from exc
        yield stream.fileno()


def _validate(event: dict) -> None:
    if not isinstance(event, dict) or event.get("event") not in KINDS:
        raise ValueError("unknown experiment event")
    if event.get("version") != 1 or not isinstance(event.get("time"), (int, float)):
        raise ValueError("invalid experiment event envelope")
    if not math.isfinite(event["time"]):
        raise ValueError("non-finite event time")
    kind = event["event"]
    if kind in {"segment", "started", "progress", "finished"} and (
        not isinstance(event.get("segment"), str) or not event["segment"]
    ):
        raise ValueError("missing segment identity")
    if kind in {"started", "progress", "finished"} and (
        not isinstance(event.get("trial"), str) or not event["trial"]
    ):
        raise ValueError("missing trial identity")
    if kind == "control" and event.get("action") not in {"stop", "resume"}:
        raise ValueError("unknown search control")
    if kind == "finished":
        status = event.get("status")
        if status not in OUTCOMES:
            raise ValueError("unknown trial outcome")
        score = event.get("score")
        if status in {"kept", "discarded"}:
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or score < 0
            ):
                raise ValueError("a scored trial needs a finite nonnegative score")
            if event.get("checks_passed") is not True:
                raise ValueError("a scored trial must pass checks")
        elif score is not None:
            raise ValueError("failed trials cannot publish comparable scores")


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _append_bytes(path: Path, payload: bytes) -> None:
    existed = path.exists()
    with path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if not existed:
        _sync_directory(path.parent)


def _repair_path(path: Path) -> Path:
    return path.with_name(path.name + ".recovery.json")


def _pending_repair(path: Path, raw: bytes) -> tuple[int, bytes] | None:
    try:
        journal = read_json(_repair_path(path))
    except FileNotFoundError:
        return None
    size, payload = journal["original_size"], journal["append"].encode("utf-8")
    if (
        not isinstance(size, int)
        or size < 0
        or size > len(raw)
        or hashlib.sha256(raw[:size]).hexdigest() != journal["original_sha256"]
        or raw[size : size + len(payload)] != payload[: len(raw) - size]
    ):
        raise ValueError("log differs from its pending tail-recovery journal")
    return size, payload


def _finish_repair(path: Path) -> None:
    raw = path.read_bytes()
    pending = _pending_repair(path, raw)
    if pending is None:
        return
    size, payload = pending
    _append_bytes(path, payload[len(raw) - size :])
    _repair_path(path).unlink()
    _sync_directory(path.parent)


def read_events(path: Path) -> tuple[list[dict], list[str]]:
    """Preserve prior events on a partial tail; reject unacknowledged corruption."""
    if not path.exists() and not _repair_path(path).exists():
        return [], []
    raw = path.read_bytes()
    events, warnings = [], []
    pending = _pending_repair(path, raw)
    if pending is not None and len(raw) < pending[0] + len(pending[1]):
        # Project the original complete records while a writer's repair is torn.
        # Only an exact journal-identified prefix is ignored; readers never write.
        raw = raw[: pending[0]]
        warnings.append("Interrupted tail recovery; original records preserved")
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            event = _loads(line)
            _validate(event)
        except (ValueError, UnicodeError, TypeError) as exc:
            digest = hashlib.sha256(line.rstrip(b"\n")).hexdigest()
            recovered = False
            if index + 1 < len(lines):
                try:
                    marker = _loads(lines[index + 1])
                    recovered = (
                        marker.get("event") == "tail-recovery"
                        and marker.get("line") == index + 1
                        and marker.get("sha256") == digest
                    )
                except (ValueError, UnicodeError, AttributeError):
                    pass
            if recovered:
                warnings.append(f"Recovered incomplete log line {index + 1}")
                continue
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                warnings.append(f"Incomplete log tail at line {index + 1}")
                continue
            raise ValueError(
                f"corrupt experiment log at line {index + 1}: {exc}"
            ) from exc
        events.append(event)
    return events, warnings


class EventTail:
    """Read the prefix once; rebuild only after replacement or damaged tails."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.signature = None
        self.identity = None
        self.rebuild = False
        self.warnings = []

    def poll(self, *, locked=False) -> tuple[bool, list[dict]]:
        """Return (rebuilt, events); callers holding .log.lock pass locked=True."""
        if locked:
            return self._read()
        try:
            guard = (self.path.parent / ".log.lock").open("rb")
        except FileNotFoundError:
            # A copied history has no writer. Never create files from a reader.
            return self._read()
        with guard:
            fcntl.flock(guard, fcntl.LOCK_SH)
            return self._read()

    def _read(self) -> tuple[bool, list[dict]]:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            if _repair_path(self.path).exists():
                raise ValueError("recovery journal exists without its log") from None
            reset = self.signature is not None
            self.__init__(self.path)
            return reset, []
        journal = _repair_path(self.path)
        repair = journal.stat() if journal.exists() else None
        signature = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            (repair.st_size, repair.st_mtime_ns, repair.st_ctime_ns)
            if repair
            else None,
        )
        if signature == self.signature:
            return False, []
        identity = (stat.st_dev, stat.st_ino)
        reset = (
            self.rebuild
            or repair is not None
            or self.identity not in (None, identity)
            or stat.st_size < self.offset
            or (self.signature is not None and stat.st_size == self.offset)
        )
        events = []
        if not reset:
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                suffix = stream.read(stat.st_size - self.offset)
            if suffix and not suffix.endswith(b"\n"):
                reset = True
            else:
                try:
                    for line in suffix.splitlines():
                        event = _loads(line)
                        _validate(event)
                        events.append(event)
                except (ValueError, UnicodeError, TypeError):
                    reset = True
        if reset:
            # Recovery authenticates the entire original prefix. This rare path
            # keeps exactly the same corruption policy as an uncached reader.
            events, self.warnings = read_events(self.path)
            with self.path.open("rb") as stream:
                stream.seek(max(0, stat.st_size - 1))
                terminated = stream.read(1) == b"\n"
            self.rebuild = repair is not None or not terminated
        else:
            self.rebuild = False
        self.offset = stat.st_size
        self.identity = identity
        self.signature = signature
        return reset, events


_WRITER_TAILS: dict[Path, EventTail] = {}


class StopReader:
    """Keep only the last durable control action, without replaying old trials."""

    def __init__(self, run: Path):
        self.tail = EventTail(run / "experiments.jsonl")
        self.stopped = False
        self.poll()

    def poll(self) -> bool:
        reset, events = self.tail.poll()
        if reset:
            self.stopped = False
        for event in events:
            if event["event"] == "control":
                self.stopped = event["action"] == "stop"
        return self.stopped


def append(run: Path, kind: str, **fields) -> dict:
    """Serialize writers, fsync every event, and acknowledge a torn final append."""
    run.mkdir(parents=True, exist_ok=True)
    path = run / "experiments.jsonl"
    event = {**fields, "event": kind, "version": 1, "time": time.time()}
    _validate(event)
    with lock(run / ".log.lock"):
        if _repair_path(path).exists():
            _finish_repair(path)
        reader = _WRITER_TAILS.setdefault(path, EventTail(path))
        reader.poll(locked=True)
        warnings = reader.warnings
        if path.exists():
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 1))
                tail = stream.read(1)
        else:
            tail = b""
        if tail and tail != b"\n":
            raw = path.read_bytes()
            if any(w.startswith("Incomplete log tail") for w in warnings):
                marker = {
                    "event": "tail-recovery",
                    "version": 1,
                    "time": time.time(),
                    "line": len(raw.splitlines()),
                    "sha256": hashlib.sha256(raw.splitlines()[-1]).hexdigest(),
                }
                payload = b"\n" + canonical_json(marker) + b"\n"
                # Persist intent before the newline can turn the damaged tail
                # into an apparent interior record. A retry appends only the
                # missing suffix, even if this repair itself tears repeatedly.
                write_json(
                    _repair_path(path),
                    {
                        "original_size": len(raw),
                        "original_sha256": hashlib.sha256(raw).hexdigest(),
                        "append": payload.decode("utf-8"),
                    },
                )
                _sync_directory(path.parent)
                _finish_repair(path)
            else:
                _append_bytes(path, b"\n")
        _append_bytes(path, canonical_json(event) + b"\n")
    return event


class Projection:
    """Current trial records and retained scores from newly read events."""

    def __init__(self):
        self.trials, self.segments, self.best = {}, {}, {}
        self.stopped = False
        self.last_update = None

    def apply(self, events):
        for event in events:
            kind = event["event"]
            if kind == "segment":
                if event["segment"] in self.segments:
                    raise ValueError("duplicate segment")
                self.segments[event["segment"]] = event
            elif kind == "control":
                self.stopped = event["action"] == "stop"
            elif kind in {"started", "progress", "finished"}:
                key, segment = event["trial"], event["segment"]
                if segment not in self.segments:
                    raise ValueError("trial references an unknown segment")
                if kind == "started":
                    if key in self.trials:
                        raise ValueError("duplicate trial start")
                    self.trials[key] = {
                        **event,
                        "status": "running",
                        "started": event["time"],
                    }
                else:
                    if (
                        key not in self.trials
                        or self.trials[key]["status"] != "running"
                    ):
                        raise ValueError("trial event without an active attempt")
                    if self.trials[key]["segment"] != segment:
                        raise ValueError("trial changed segment")
                    self.trials[key].update(event)
                trial = self.trials[key]
                if kind == "finished" and trial["status"] == "kept":
                    if trial.get("diagnostic") or trial.get("baseline") == "small":
                        raise ValueError("a diagnostic baseline cannot be kept")
                    if (
                        segment in self.best
                        and trial["score"] >= self.best[segment]["score"]
                    ):
                        raise ValueError("kept trial does not improve its segment")
                    self.best[segment] = {
                        k: trial.get(k)
                        for k in ("trial", "score", "revision", "candidate_sha256")
                    }
            self.last_update = event["time"]

    def view(self, warnings):
        return {
            "segments": list(self.segments.values()),
            "trials": list(self.trials.values()),
            "best": self.best,
            "stopped": self.stopped,
            "warnings": warnings,
            "last_update": self.last_update,
            "server_time": time.time(),
        }


def snapshot(run: Path) -> dict:
    """Project the event history without inventing results for unfinished attempts."""
    events, warnings = read_events(run / "experiments.jsonl")
    projection = Projection()
    projection.apply(events)
    return projection.view(warnings)


def reconcile(run: Path) -> list[str]:
    """Caller holds .trial.lock, including any surviving training child's lock."""
    interrupted = []
    for trial in snapshot(run)["trials"]:
        if trial["status"] == "running":
            append(
                run,
                "finished",
                trial=trial["trial"],
                segment=trial["segment"],
                status="interrupted",
                score=None,
                reason="Previous supervisor exited before a durable result; retry with fresh weights",
            )
            interrupted.append(trial["trial"])
    return interrupted
