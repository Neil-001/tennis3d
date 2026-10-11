#!/usr/bin/env python3
"""Read-only Stage 4 metrics exporter. Requires a separate monitoring environment."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import stat
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

SDK_VERSION = "0.30.0"
JOURNALS = ("loss.jsonl", "validation-history.jsonl")
AXES = {
    "step": "axes/training_step",
    "windows": "axes/accepted_windows",
    "time": "axes/train_seconds",
    "job_time": "axes/job_seconds",
}
UNITS = {
    "loss": "trainer loss_units, dimensionless scaled objective; not metres",
    "validation/*_m": "metres",
    "validation/*_px": "pixels",
    "*_seconds": "seconds",
    "throughput/windows_per_second": "accepted windows/second, single update",
    "gradient_norm": "L2 norm before clipping",
    "learning_rate": "optimizer group rate",
    "weight_rms": "parameter RMS",
    "first_moment_rms": "optimizer first moment RMS",
    "budget/*_fraction": "fraction of configured cap, may exceed 1",
    "windows": "accepted crop windows, not unique data coverage",
    "sampled_views": "view draws",
    "rejected_crops": "rejected crop draws",
}


class Blocked(ValueError):
    """Source or tracking identity requires operator recovery."""


class Waiting(RuntimeError):
    """A synced source snapshot is not ready yet."""


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    try:
        return json.loads(path.read_bytes())
    except FileNotFoundError as error:
        raise Waiting(f"waiting for {path.name}") from error
    except (ValueError, UnicodeError) as error:
        raise Blocked(f"invalid JSON in {path.name}") from error


def atomic_json(path, value):
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def numbers(value, prefix=""):
    """Only aggregate numeric diagnostics; exclude per-view/rally payloads."""
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if key not in {"views", "rallies", "source_identity", "manifest_sha256"}:
                result.update(numbers(item, f"{prefix}/{key}" if prefix else key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise Blocked("nonfinite source metric")
        result[prefix] = value
    return result


def metrics(kind, row, config, index=0, replay=False):
    result = {
        "axes/training_step": row["step"],
        "axes/train_seconds": row["train_seconds"],
        "axes/job_seconds": row["job_seconds"],
    }
    if "windows" in row:
        result["axes/accepted_windows"] = row["windows"]
    values = {}
    if kind == "loss.jsonl":
        for key in (
            "loss",
            "gradient_norm",
            "step_seconds",
            "epoch",
            "sampled_views",
            "rejected_crops",
        ):
            if key in row:
                values[f"train/{key}"] = row[key]
        if row.get("step_seconds", 0) > 0:
            values["throughput/windows_per_second"] = (
                config["batch_size"] / row["step_seconds"]
            )
        for i, rate in enumerate(row.get("learning_rates", [])):
            values[f"stability/group_{i}/learning_rate"] = rate
        for i, group in enumerate(row.get("optimizer_stability", [])):
            values.update(numbers(group, f"stability/group_{i}"))
        result["export/replayed_step"] = int(replay)
        result["export/loss_units"] = row.get("loss_units", "unspecified")
    elif kind == "validation-history.jsonl":
        values = numbers(row["modes"], "validation")
        result.update(
            {
                "export/validation_attempt": index + 1,
                "export/invocation_id": row["invocation_id"],
                "export/resumed_from_step": row.get("resumed_from_step"),
            }
        )
    else:
        for field, limit in (
            ("train_seconds", "max_train_seconds"),
            ("job_seconds", "max_job_seconds"),
            ("step", "max_steps"),
        ):
            values[f"budget/{field}"] = row[field]
            if config.get(limit):
                values[f"budget/{field}_fraction"] = row[field] / config[limit]
                values[f"budget/{field}_remaining"] = max(0, config[limit] - row[field])
    values = numbers(values)
    # Scalar panels in each section have an explicit default X axis. The same
    # event supplies all axes, even when a checkpoint causes steps to replay.
    for axis, axis_metric in AXES.items():
        if axis_metric in result:
            result.update({f"by_{axis}/{key}": value for key, value in values.items()})
    return result


class Reader:
    def __init__(self, source, state_dir, project, name, entity=None, read_only=False):
        self.source, self.state_dir = source.resolve(), state_dir.resolve()
        if self.state_dir == self.source or self.source in self.state_dir.parents:
            raise Blocked("tracking state must be outside training output")
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.read_only = read_only
        self.path = self.state_dir / "state.json"
        self.state: dict[str, Any] = (
            read_json(self.path)
            if self.path.exists()
            else {
                "schema_version": 1,
                "project": project,
                "name": name,
                "entity": entity,
                "journals": {},
                "acknowledged": [],
            }
        )
        if self.state["project"] != project or (
            entity and self.state["entity"] not in (None, entity)
        ):
            raise Blocked("external tracking project/entity differs")
        if entity and not self.state["entity"]:
            self.state["entity"] = entity

    def snapshot(self):
        run, config = (
            read_json(self.source / "run.json"),
            read_json(self.source / "config.resolved.json"),
        )
        run_id = run["run_id"]
        if self.state.get("trainer_run_id", run_id) != run_id:
            raise Blocked("external tracking state belongs to another trainer run_id")
        if not isinstance(run_id, str) or not run_id.isalnum():
            raise Blocked("invalid trainer run_id")
        if run.get("config_sha256") != digest(config):
            raise Waiting("run.json/config.resolved.json sync not yet coherent")
        self.state.update(trainer_run_id=run_id, wandb_run_id=run_id)
        events, pending = [], []
        observed = {}
        for filename in JOURNALS:
            try:
                raw = (self.source / filename).read_bytes()
            except FileNotFoundError:
                raw = b""
            lines = raw.splitlines(keepends=True)
            if lines and not lines[-1].endswith(b"\n"):
                pending.append(f"{filename}: incomplete trailing line")
                lines.pop()
            hashes = []
            previous_step = -1
            for index, line in enumerate(lines):
                try:
                    row = json.loads(line)
                    if filename == JOURNALS[1] and row["run_id"] != run_id:
                        raise Blocked("validation journal belongs to another run")
                    hashed = digest(row)
                    payload = metrics(
                        filename, row, config, index, row["step"] <= previous_step
                    )
                    previous_step = row["step"]
                except (KeyError, TypeError, ValueError, UnicodeError) as error:
                    raise Blocked(f"invalid {filename} record {index + 1}") from error
                hashes.append(hashed)
                event_id = f"{filename}:{index}:{hashed}"
                events.append(
                    {"export/event_id": event_id, "export/source": filename, **payload}
                )
            old = self.state["journals"].get(filename, [])
            common = min(len(old), len(hashes))
            if old[:common] != hashes[:common]:
                raise Blocked(
                    f"divergent {filename}; restore complete original history; source preserved"
                )
            if len(hashes) < len(old):
                raise Waiting(f"{filename} temporarily shorter than observed history")
            observed[filename] = hashes
        budget = read_json(self.source / "budget.json")
        if budget["run_id"] != run_id:
            raise Blocked("budget belongs to another trainer run_id")
        events.append(
            {
                "export/event_id": f"budget:{digest(budget)}",
                "export/source": "budget.json",
                **metrics("budget", budget, config),
            }
        )
        summary = self.source / "summary.json"
        if summary.exists():
            allocation = read_json(summary)
            if allocation["run_id"] != run_id:
                raise Blocked("summary belongs to another trainer run_id")
            events.append(
                {
                    "export/event_id": f"summary:{digest(allocation)}",
                    "export/source": "summary.json",
                    "allocation/stop_reason": allocation["stop_reason"],
                    "allocation/summary_step": allocation["step"],
                    **metrics("budget", allocation, config),
                }
            )
        self.state["journals"] = observed
        self.save()
        return run, config, events, pending

    def save(self):
        if not self.read_only:
            atomic_json(self.path, self.state)

    def status(self, status, **fields):
        atomic_json(
            self.state_dir / ("verification.json" if self.read_only else "status.json"),
            {
                "status": status,
                "updated_at_unix": time.time(),
                "trainer_run_id": self.state.get("trainer_run_id"),
                "run_id": self.state.get("trainer_run_id"),
                "wandb_run_id": self.state.get("wandb_run_id"),
                "run_url": self.state.get("run_url"),
                **fields,
            },
        )


class Backend:
    def __init__(self, reader, timeout):
        import wandb  # ty: ignore[unresolved-import] - optional isolated SDK

        if wandb.__version__ != SDK_VERSION:
            raise Blocked(f"monitoring requires wandb=={SDK_VERSION}")
        self.wandb, self.reader, self.timeout = wandb, reader, timeout
        self.run: Any = None
        self.sent = set()

    def connect(self, provenance, config, publish=True):
        state = self.reader.state
        self.api = self.wandb.Api(timeout=30)
        if not state["entity"]:
            state["entity"] = self.api.default_entity
        if not state["entity"]:
            raise Blocked("credential has no usable default entity")
        self.remote_path = (
            f"{state['entity']}/{state['project']}/{state['wandb_run_id']}"
        )
        state["run_url"] = f"https://wandb.ai/{self.remote_path}"
        self.reader.save()
        if publish:
            settings = self.wandb.Settings(
                console="off",
                disable_code=True,
                disable_git=True,
                save_code=False,
                x_disable_meta=True,
                x_disable_stats=True,
                x_disable_machine_info=True,
                silent=True,
                init_timeout=60,
                finish_timeout=self.timeout,
                finish_timeout_raises=True,
            )
            metadata = {
                key: provenance[key]
                for key in (
                    "run_id",
                    "code_revision",
                    "source_files",
                    "source_sha256",
                    "config_sha256",
                    "manifest_sha256",
                    "dataset_identity",
                    "parameter_count",
                    "training_objective",
                    "execution_limits",
                )
                if key in provenance
            }
            # Paths and environment metadata never leave the login host.
            safe_config = {
                k: v
                for k, v in config.items()
                if k not in ("train_manifest", "val_manifest", "artifact_label")
            }
            self.run = self.wandb.init(
                entity=state["entity"],
                project=state["project"],
                id=state["wandb_run_id"],
                name=state["name"],
                resume="allow",
                mode="online",
                dir=str(self.reader.state_dir),
                config={
                    "trainer": safe_config,
                    "provenance": metadata,
                    "metric_units": UNITS,
                    "monitoring": "read-only login-host exporter; no training GPU system metrics",
                },
                settings=settings,
            )
            for axis in (*AXES.values(), "axes/job_seconds"):
                self.run.define_metric(axis, hidden=True)
            for key, axis in AXES.items():
                self.run.define_metric(f"by_{key}/*", step_metric=axis, step_sync=False)
            self.run.define_metric("export/*", hidden=True)
            self.run.define_metric("allocation/*", hidden=True)
        return self.acknowledge()

    def acknowledge(self):
        self.api.flush()
        remote = self.api.run(self.remote_path)
        if (
            remote.config.get("provenance", {}).get("run_id")
            != self.reader.state["trainer_run_id"]
        ):
            raise Blocked("remote W&B run identity differs")
        acknowledged = set()
        for row in remote.scan_history(keys=["export/event_id"], use_cache=False):
            acknowledged.add(row["export/event_id"])
        self.reader.state["acknowledged"] = sorted(acknowledged)
        self.reader.state["last_server_verification_unix"] = time.time()
        self.reader.save()
        return acknowledged

    def publish(self, events, acknowledged):
        for row in events:
            event_id = row["export/event_id"]
            if event_id not in acknowledged and event_id not in self.sent:
                self.run.log(
                    row
                )  # W&B assigns its internal step. Never use training step here.
                self.sent.add(event_id)

    def finish(self):
        if self.run is not None:
            run, self.run = self.run, None
            run.finish()


def authenticate(path):
    if path is None:
        return  # Standard WANDB_API_KEY or private SDK authentication.
    mode = path.stat().st_mode
    if not stat.S_ISREG(mode) or mode & 0o077 or path.parent.stat().st_mode & 0o077:
        raise Blocked("API key file and parent directory must be private")
    os.environ["WANDB_API_KEY"] = path.read_text().strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--entity")
    parser.add_argument("--project", default="tennis3d")
    parser.add_argument("--name", default="stage4-seed42")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--minimum-step",
        type=int,
        help="verify first server-visible loss at or above this update",
    )
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--drain-timeout", type=float, default=180)
    args = parser.parse_args(argv)
    if (
        args.poll_seconds <= 0
        or args.drain_timeout <= 0
        or (args.watch and args.verify_only)
    ):
        parser.error("positive timeouts required; --verify-only cannot watch")
    if args.minimum_step is not None and (
        not args.verify_only or args.minimum_step < 1
    ):
        parser.error("--minimum-step requires --verify-only and a positive update")
    os.umask(0o077)
    reader = Reader(
        args.source,
        args.state_dir,
        args.project,
        args.name,
        args.entity,
        read_only=args.verify_only,
    )
    with (reader.state_dir / "exporter.lock").open("a") as lock:
        try:
            if not args.verify_only:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            reader.status("blocked", detail="another exporter holds the lock")
            return 2
        stopped = threading.Event()
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: stopped.set())
        backend = None
        try:
            authenticate(args.api_key_file)
            while True:
                try:
                    run, config, events, pending = reader.snapshot()
                    if backend is None:
                        backend = Backend(reader, args.drain_timeout)
                        acknowledged = backend.connect(
                            run, config, not args.verify_only
                        )
                    else:
                        acknowledged = backend.acknowledge()
                    if args.minimum_step is not None:
                        events = [
                            row
                            for row in events
                            if row["export/source"] == "loss.jsonl"
                            and row["axes/training_step"] >= args.minimum_step
                        ][:1]
                        if not events:
                            raise Waiting(
                                "no complete loss record at minimum update yet"
                            )
                        pending = []  # Later incomplete rows do not invalidate this first-live proof.
                    if not args.verify_only:
                        backend.publish(events, acknowledged)
                    missing = {row["export/event_id"] for row in events} - acknowledged
                    reader.status(
                        "verified"
                        if args.verify_only and not missing and not pending
                        else "live"
                        if not args.verify_only
                        and acknowledged.intersection(
                            row["export/event_id"] for row in events
                        )
                        and not pending
                        else "awaiting_server",
                        pending_events=len(missing),
                        source_event_count=len(events),
                        acknowledged_event_count=len(events) - len(missing),
                        latest_step=max(
                            (
                                row["axes/training_step"]
                                for row in events
                                if row["export/source"] == "loss.jsonl"
                            ),
                            default=0,
                        ),
                        acknowledged_latest_step=max(
                            (
                                row["axes/training_step"]
                                for row in events
                                if row["export/source"] == "loss.jsonl"
                                and row["export/event_id"] in acknowledged
                            ),
                            default=0,
                        ),
                        verified_metrics=[
                            row
                            for row in events
                            if row["export/source"] == "loss.jsonl"
                            and row["export/event_id"] in acknowledged
                        ][:1]
                        if args.verify_only
                        else [],
                        source_wait=pending,
                        last_server_verification_unix=reader.state.get(
                            "last_server_verification_unix"
                        ),
                    )
                    if args.verify_only:
                        return 0 if not missing and not pending else 2
                    if not args.watch or stopped.is_set():
                        # A signal can arrive during upload, after the polling snapshot.
                        # Read current durable artifacts again before declaring a drain.
                        _, _, events, pending = reader.snapshot()
                        backend.publish(events, acknowledged)
                        backend.finish()
                        deadline = time.monotonic() + args.drain_timeout
                        while True:
                            acknowledged = backend.acknowledge()
                            missing = {
                                row["export/event_id"] for row in events
                            } - acknowledged
                            if not missing or time.monotonic() >= deadline:
                                break
                            stopped.wait(2) if not stopped.is_set() else time.sleep(2)
                        success = not missing and not pending
                        reader.status(
                            "drained" if success else "retry_required",
                            pending_events=len(missing),
                            source_wait=pending,
                        )
                        return 0 if success else 2
                except (Blocked, Waiting) as error:
                    reader.status(
                        "blocked"
                        if isinstance(error, Blocked)
                        else "waiting_for_source",
                        detail=str(error),
                    )
                    if not args.watch or stopped.is_set():
                        return 2
                except Exception as error:  # noqa: BLE001 - SDK failure boundary, no secret exception text
                    # Do not echo SDK exception text: network errors may include credentials.
                    reader.status(
                        "disconnected",
                        error_type=type(error).__name__,
                        retry_seconds=args.poll_seconds,
                    )
                    if backend is not None and backend.run is None:
                        backend = None
                    if not args.watch or stopped.is_set():
                        return 2
                stopped.wait(args.poll_seconds)
        finally:
            if backend is not None:
                try:
                    backend.finish()
                except Exception as error:  # noqa: BLE001 - SDK failure boundary, no secret exception text
                    reader.status("retry_required", error_type=type(error).__name__)


if __name__ == "__main__":
    raise SystemExit(main())
