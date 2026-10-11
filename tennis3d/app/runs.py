"""Owned run storage and the single-worker reconstruction queue.

Each run is a directory below `<home>/runs/<id>`: the uploaded source, the
ordinary inference run directory, a browser playback copy and small JSON state
files. One reconstruction executes at a time in a separate process, so a cancel
or crash never takes down the server and completed stages remain reusable.
"""

import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from tennis3d.app.capabilities import DEVICES, IntelCapability
from tennis3d.app.home import installed_config, problems
from tennis3d.artifacts import (
    atomic_file,
    file_sha256,
    json_sha256,
    read_json,
    write_json,
)

RUN_ID = re.compile(r"[0-9a-f]{16}")
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".ts", ".mts"}
ACTIVE = ("queued", "running")
STAGES = ("checking", "detect", "calibrate", "lift", "export", "playback", "complete")
# Bound Intel discovery, model compilation and native teardown as well as
# inference. Long videos may keep running while they report real progress.
INTEL_IDLE_TIMEOUT_S = 600
MESSAGES = {
    2: "This video could not be processed. Use one continuous single-camera clip.",
    3: "The reconstruction runtime or model weights are unavailable.",
    4: "No usable 3D trajectory was found in this video.",
}


class AlreadyRunning(RuntimeError):
    """Another Tennis3D server already owns this app directory."""


def lock_home(root: Path):
    """Hold an exclusive OS lock for the life of the server process."""
    stream = (root / "server.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        raise AlreadyRunning("Tennis3D is already running") from error
    return stream


class RunError(ValueError):
    """A request that cannot be honoured; the message is shown to the user."""


def now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def display_name(name: str) -> str:
    """Keep only a printable base name; storage never uses the supplied name."""
    name = re.split(r"[\\/]", name or "")[-1]
    name = "".join(c for c in name if c.isprintable()).strip()[:120]
    return name or "video"


def declared_frames(path: Path):
    """Container-declared frame count, used only to scale the progress bar."""
    try:
        import av

        with av.open(str(path)) as container:
            return container.streams.video[0].frames or None
    except Exception:  # noqa: BLE001 - any decoder failure means unknown
        return None


class RunStore:
    def __init__(self, root: Path):
        self.root = root
        self.directory = root / "runs"
        self.directory.mkdir(parents=True, exist_ok=True)
        # Two servers on one directory would each start a worker for the same run.
        self.owner = lock_home(root)
        self.lock = threading.RLock()
        self.process = None
        self.current = None
        self.closed = False
        self.intel = IntelCapability()
        for run in self.list():
            # A previous server ended while this run was active.
            if run["status"] in ACTIVE:
                self._update(
                    run["id"],
                    status="interrupted",
                    error="Processing stopped before it finished. Start it again to resume.",
                )
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def worker_command(self, path: Path) -> list[str]:
        return [sys.executable, "-m", "tennis3d.app.worker", str(path)]

    def path(self, run_id: str) -> Path:
        if not RUN_ID.fullmatch(run_id or ""):
            raise KeyError(run_id)
        path = self.directory / run_id
        if not (path / "run.json").is_file():
            raise KeyError(run_id)
        return path

    def _update(self, run_id, **changes):
        with self.lock:
            path = self.path(run_id) / "run.json"
            record = read_json(path) | changes
            write_json(path, record, overwrite=True)
            return record

    def get(self, run_id: str) -> dict:
        path = self.path(run_id)
        with self.lock:
            record = read_json(path / "run.json")
        if record["status"] == "running":
            try:
                record |= read_json(path / "progress.json")
            except (OSError, ValueError):
                pass
        record["has_result"] = (path / "inference" / "result.json").is_file()
        record["has_video"] = (path / "playback.mp4").is_file() and (
            path / "playback.json"
        ).is_file()
        return record

    def list(self) -> list[dict]:
        runs = []
        for path in self.directory.iterdir():
            try:
                runs.append(self.get(path.name))
            except (KeyError, OSError, ValueError):
                continue
        return sorted(runs, key=lambda run: run["created"], reverse=True)

    def create(self, name: str, stream, length: int, device: str) -> dict:
        """Store an uploaded video under a generated identifier and queue it."""
        name = display_name(name)
        suffix = Path(name).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise RunError(
                "Unsupported file type. Choose a video such as MP4, MOV or MKV."
            )
        self.validate_device(device)
        run_id = secrets.token_hex(8)
        path = self.directory / run_id
        path.mkdir()
        source = path / f"source{suffix}"
        try:
            with atomic_file(source) as target:
                remaining = length
                while remaining:
                    block = stream.read(min(1 << 20, remaining))
                    if not block:
                        raise RunError("The video transfer ended early. Try again.")
                    target.write(block)
                    remaining -= len(block)
            write_json(
                path / "run.json",
                {
                    "id": run_id,
                    "name": name,
                    "created": now(),
                    "status": "created",
                    "device": device,
                    "source_name": source.name,
                    "source_bytes": length,
                    "source_sha256": file_sha256(source),
                    "declared_frames": declared_frames(source),
                },
            )
        except BaseException:
            shutil.rmtree(path, ignore_errors=True)
            raise
        return self.start(run_id, device)

    def validate_device(self, device):
        if device not in DEVICES:
            raise RunError("Unknown processing device.")
        if device == "intel-gpu" and self.intel.snapshot()["status"] != "available":
            raise RunError(
                "Intel GPU detection is unavailable. Choose CPU or Automatic, "
                "or restart Tennis3D after installing a supported host runtime."
            )

    def start(self, run_id: str, device=None) -> dict:
        with self.lock:
            record = self.get(run_id)
            device = record["device"] if device is None else device
            self.validate_device(device)
            if record["status"] in ACTIVE:
                return record
            if missing := problems(self.root):
                raise RunError(" ".join(missing))
            write_json(
                self.path(run_id) / "request.json",
                {
                    "source_name": record["source_name"],
                    "config": installed_config(self.root, device),
                },
                overwrite=True,
            )
            for name in ("progress.json", "outcome.json"):
                (self.path(run_id) / name).unlink(missing_ok=True)
            return self._update(
                run_id,
                status="queued",
                device=device,
                queued=time.time(),
                error=None,
                summary=None,
            )

    def cancel(self, run_id: str) -> dict:
        with self.lock:
            record = self.get(run_id)
            if record["status"] not in ACTIVE:
                return record
            record = self._update(
                run_id,
                status="cancelled",
                error="Cancelled. Finished stages are kept; start again to resume.",
            )
            if self.current == run_id and self.process:
                self.process.terminate()
            return record

    def delete(self, run_id: str):
        self.cancel(run_id)
        path = self.path(run_id)
        with self.lock:
            if self.current == run_id and self.process:
                self.process.wait(timeout=30)
            shutil.rmtree(path)

    def close(self):
        with self.lock:
            self.closed = True
            if self.process and self.process.poll() is None:
                self.process.terminate()
        self.thread.join(timeout=30)
        self.owner.close()

    def _next(self):
        queued = [run for run in self.list() if run["status"] == "queued"]
        return min(queued, key=lambda run: run["queued"])["id"] if queued else None

    def _loop(self):
        while not self.closed:
            with self.lock:
                run_id = None if self.closed else self._next()
                if run_id:
                    path = self.path(run_id)
                    self._update(run_id, status="running", started=now())
                    self.current = run_id
                    device = self.get(run_id)["device"]
                    with (path / "worker.log").open("ab") as log:
                        self.process = subprocess.Popen(
                            self.worker_command(path),
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL,
                            # Never import a `tennis3d` folder from the caller's directory.
                            cwd=path,
                        )
            if not run_id:
                time.sleep(0.2)
                continue
            process = self.process
            assert process is not None
            code = self._wait(process, path, device)
            with self.lock:
                self.current = self.process = None
                self._finish(run_id, code)

    def _wait(self, process, path, device):
        if device != "intel-gpu":
            return process.wait()
        last_progress, version = time.monotonic(), None
        while True:
            try:
                return process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
            try:
                current = (path / "progress.json").stat().st_mtime_ns
            except OSError:
                current = None
            if current is not None and current != version:
                last_progress, version = time.monotonic(), current
            if time.monotonic() - last_progress < INTEL_IDLE_TIMEOUT_S:
                continue
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            # Cancel/close messages take precedence over the watchdog failure.
            with self.lock:
                try:
                    record = self.get(path.name)
                except KeyError:
                    return 3  # Deleted after cancellation.
                if not self.closed and record["status"] == "running":
                    write_json(
                        path / "outcome.json",
                        {
                            "error": f"Intel GPU processing stopped after {INTEL_IDLE_TIMEOUT_S:g} seconds without progress. The host runtime may have stalled.",
                            "error_type": "TimeoutError",
                            "produced_result": False,
                        },
                        overwrite=True,
                    )
            return 3

    def _finish(self, run_id, code):
        try:
            record = self.get(run_id)
        except KeyError:
            return
        if record["status"] != "running":
            return  # Cancelled; its message is already recorded.
        if self.closed:
            self._update(
                run_id,
                status="interrupted",
                error="Processing stopped before it finished. Start it again to resume.",
            )
            return
        try:
            outcome = read_json(self.path(run_id) / "outcome.json")
        except (OSError, ValueError):
            outcome = {}
        if (
            code in (1, 3)
            and record["device"] == "intel-gpu"
            and not outcome.get("detector_stalled")
            and (failure := self._published_detection_failure(run_id, record))
        ):
            # The hard deadline may fall between publishing a prefix
            # outcome and tagging it in process_main. Keep its bytes, but
            # never call the newly recorded detector failure a success.
            outcome = (
                outcome
                | failure
                | {
                    "produced_result": bool(outcome.get("produced_result")),
                    "detector_failed": True,
                }
            )
        summary = outcome.get("summary")
        summary = summary if isinstance(summary, dict) else {}
        # Only this attempt's own summary and result make the run complete; a
        # result kept from an earlier attempt stays viewable beside the failure.
        detector_failed = bool(
            outcome.get("detector_stalled") or outcome.get("detector_failed")
        )
        usable = (
            bool(outcome.get("produced_result"))
            and record["has_result"]
            and not detector_failed
            and code not in (1, 3)
        )
        error = None
        if code:
            error = MESSAGES.get(code, "Reconstruction failed.")
            if not outcome:
                error = "The reconstruction process stopped unexpectedly."
            elif detail := outcome.get("error") or summary.get("error"):
                error = f"{error} ({detail})"
            if not usable and record["has_result"]:
                if outcome.get("produced_result"):
                    error += (
                        " The result from this attempt's decoded prefix is shown for inspection."
                        if detector_failed
                        else " The result from this failed attempt is shown for inspection."
                    )
                else:
                    error += " The earlier result is still shown."
        self._update(
            run_id,
            status="complete" if usable else "failed",
            finished=now(),
            exit_code=code,
            error=error,
            summary=summary or None,
        )

    def _published_detection_failure(self, run_id, record) -> dict:
        """Recover a newly published detector failure after worker exit.

        A retained cache from another attempt must never explain this worker's
        exit. Check publication time, source and resolved request settings.
        """
        from tennis3d.config import load_config
        from tennis3d.inference.settings import ReconstructionConfig

        path = self.path(run_id)
        detection = path / "inference" / "detect.json"
        try:
            if detection.stat().st_mtime < record["queued"]:
                return {}
            summary = read_json(detection)
            request = read_json(path / "request.json")
            config = load_config(ReconstructionConfig, overrides=request["config"])
            if (
                summary.get("status") != "failed"
                or summary.get("failure_kind") != "runtime"
                or summary.get("exit_code") != 3
                or summary.get("device") != "intel-gpu"
                or summary.get("source_sha256") != record["source_sha256"]
                or summary.get("config_sha256")
                != json_sha256(config.detection.model_dump(mode="json"))
            ):
                return {}
        except (OSError, ValueError, KeyError, TypeError):
            return {}
        return {
            "summary": summary,
            "produced_result": False,
            "error": "Intel GPU detection failed. The worker exited before committing its outcome. "
            + str(summary.get("failure") or "The native runtime may have stalled."),
        }
