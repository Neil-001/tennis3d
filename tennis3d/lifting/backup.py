"""Immutable run snapshots and a separate private-Hub uploader.

Run with ``python -m tennis3d.lifting.backup --help``. Only the fixed run
artifact allowlist is copied. Network operations never run in the trainer.
"""

import argparse
import fcntl
import math
import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from tennis3d.artifacts import _loads, file_sha256, json_sha256, read_json, write_json
from tennis3d.lifting.checkpoint import _same_dataset_identity, read_checkpoint
from tennis3d.lifting.model import ModelConfig

SCHEMA = "tennis3d-backup-1"
REQUIRED = (
    "checkpoints/last.pt",
    "budget.json",
    "config.resolved.json",
    "run.json",
)
OPTIONAL = (
    "checkpoints/best.pt",
    "loss.jsonl",
    "validation-history.jsonl",
    "validation.json",
    "summary.json",
)
ALLOWED = frozenset(REQUIRED + OPTIONAL)
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")


class BackupError(ValueError):
    """A safe diagnostic produced by this module, without SDK request details."""


MAX_PENDING = 3
KEEP_UPLOADED = 2


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _copy(source, destination):
    shutil.copyfile(source, destination)
    with destination.open("rb") as stream:
        os.fsync(stream.fileno())
    _fsync_directory(destination.parent)


def _state(store):
    path = store / "state.json"
    return read_json(path) if path.exists() else {"receipts": {}}


def _save_state(store, state):
    write_json(store / "state.json", state, overwrite=True)


def _files(source):
    files = {}
    for name in REQUIRED + OPTIONAL:
        path = source / name
        if path.is_symlink() or any(
            (source / p).is_symlink() for p in Path(name).parents if p != Path(".")
        ):
            raise BackupError("symlink in run artifacts")
        if not path.exists() and name in OPTIONAL:
            continue
        if not path.is_file():
            raise BackupError(f"missing run artifact: {name}")
        files[name] = {"sha256": file_sha256(path), "size": path.stat().st_size}
    return files


def _number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _journal(path, run_id=None):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = _loads(line)
            if (
                not line.endswith("\n")
                or not isinstance(row, dict)
                or not _number(row.get("step"))
                or (run_id is not None and row.get("run_id") != run_id)
            ):
                raise BackupError("invalid or torn journal; preserve source evidence")
            rows.append(row)
    return rows


def _validate_run(source):
    last = read_checkpoint(source / "checkpoints/last.pt")
    run = read_json(source / "run.json")
    config = read_json(source / "config.resolved.json")
    ledger = read_json(source / "budget.json")
    if (
        not isinstance(last["run_id"], str)
        or not last["run_id"]
        or run != last["provenance"]
        or config != last["config"]
        or run["run_id"] != last["run_id"]
        or ledger["run_id"] != last["run_id"]
        or run["config_sha256"] != json_sha256(config)
        or run["manifest_sha256"] != last["manifest_sha256"]
        or run.get("dataset_identity") != last.get("dataset_identity")
        or not _number(last["step"])
        or not _number(ledger["step"])
        or ledger["step"] < last["step"]
    ):
        raise BackupError("checkpoint, config, provenance or ledger identity mismatch")
    for key in ("train_seconds", "job_seconds"):
        if not _number(ledger[key]) or not _number(last["budget"][key]):
            raise BackupError("invalid budget")
    if ledger["train_seconds"] < last["budget"]["train_seconds"]:
        raise BackupError("ledger loses charged training budget")
    best_path = source / "checkpoints/best.pt"
    if best_path.exists():
        best = read_checkpoint(best_path)
        if (
            best["run_id"] != last["run_id"]
            or best["manifest_sha256"] != last["manifest_sha256"]
            or not _same_dataset_identity(best, last)
            or ModelConfig.model_validate(best["config"]["model"])
            != ModelConfig.model_validate(last["config"]["model"])
            or not _number(best["best_score"])
            or best["step"] != last["best_step"]
            or best["best_score"] != last["best_score"]
            or best["step"] > last["step"]
            or best.get("best_step", best["step"]) != best["step"]
        ):
            raise BackupError("best checkpoint does not match last publication")
    elif last["best_score"] is not None:
        raise BackupError("selected original best checkpoint is missing")
    for name in ("loss.jsonl", "validation-history.jsonl"):
        rows = _journal(
            source / name,
            last["run_id"] if name == "validation-history.jsonl" else None,
        )
        if any(row["step"] > ledger["step"] for row in rows):
            raise BackupError("journal is ahead of budget ledger")
    validation = source / "validation.json"
    if validation.exists() and read_json(validation)["step"] > ledger["step"]:
        raise BackupError("validation is ahead of budget ledger")
    summary = source / "summary.json"
    if summary.exists():
        value = read_json(summary)
        if (
            value.get("run_id") != last["run_id"]
            or value.get("step") != last["step"]
            or any(
                value.get(key) != expected
                for key, expected in last["provenance"].items()
            )
            or value.get("best_score") != last["best_score"]
            or value.get("last_validation_step") != last.get("last_validation_step", -1)
            or not _number(value.get("job_seconds"))
            or value.get("last_checkpoint_sha256")
            != file_sha256(source / "checkpoints/last.pt")
        ):
            raise BackupError("summary does not match checkpoint")
    return last


def _manifest_body(manifest):
    return {key: value for key, value in manifest.items() if key != "snapshot_id"}


def _check_manifest(manifest, snapshot_id):
    if (
        not HEX64.fullmatch(snapshot_id)
        or manifest.get("schema") != SCHEMA
        or manifest.get("snapshot_id") != snapshot_id
        or json_sha256(_manifest_body(manifest)) != snapshot_id
        or not isinstance(manifest.get("files"), dict)
        or not set(REQUIRED).issubset(manifest["files"])
        or not set(manifest["files"]).issubset(ALLOWED)
    ):
        raise BackupError("invalid snapshot manifest or identity")
    for metadata in manifest["files"].values():
        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("sha256"), str)
            or not HEX64.fullmatch(metadata["sha256"])
            or not isinstance(metadata.get("size"), int)
            or metadata["size"] < 0
        ):
            raise BackupError("invalid snapshot file metadata")


def verify_snapshot(path, snapshot_id, run_id=None):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    _check_manifest(manifest, snapshot_id)
    if run_id is not None and manifest["run_id"] != run_id:
        raise BackupError("snapshot belongs to another run")
    if _files(path) != manifest["files"]:
        raise BackupError("snapshot file hash or size mismatch")
    last = _validate_run(path)
    if last["run_id"] != manifest["run_id"] or last["step"] != manifest["step"]:
        raise BackupError("snapshot checkpoint identity mismatch")
    return manifest


def _pending(store, state):
    root = store / "snapshots"
    return (
        sorted(
            (p for p in root.iterdir() if p.name not in state["receipts"]),
            key=lambda p: p.stat().st_mtime_ns,
        )
        if root.exists()
        else []
    )


def publish(source, store, *, interval=0, started=None):
    """Copy a stable, semantically checked publication, without changing source."""
    capture_started = time.monotonic()
    if started is not None and (
        not math.isfinite(started) or started > capture_started
    ):
        raise BackupError("invalid final-publication start time")
    source, store = Path(source).absolute(), Path(store).absolute()
    if source.is_symlink() or store.is_symlink():
        raise BackupError("run/store directory may not be a symlink")
    resolved_source, resolved_store = source.resolve(), store.resolve()
    if (
        resolved_source == resolved_store
        or resolved_source in resolved_store.parents
        or resolved_store in resolved_source.parents
    ):
        raise BackupError("backup store and run directory must be separate")
    with locked(str(source) + ".sync.lock"), locked(store / ".lock"):
        state = _state(store)
        if interval and time.time() - state.get("published_at", 0) < interval:
            return state.get("latest_snapshot")
        recorded_final = started is not None and f"{started:.9f}" in state.get(
            "final_publications", {}
        )
        if not recorded_final and len(_pending(store, state)) >= MAX_PENDING:
            raise BackupError("backup queue full; drain uploads and retry publication")
        snapshots = store / "snapshots"
        snapshots.mkdir(exist_ok=True)
        for abandoned in store.glob(".capture-*"):
            shutil.rmtree(abandoned)
        with tempfile.TemporaryDirectory(prefix=".capture-", dir=store) as temporary:
            target = Path(temporary)
            before = _files(source)
            for name in before:
                destination = target / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                _copy(source / name, destination)
            if _files(source) != before or _files(target) != before:
                raise BackupError(
                    "run changed during snapshot; retry without rewriting it"
                )
            # A previous allocation's summary may remain while its run continues.
            # Include it only if its hash/identity agrees with the copied checkpoint.
            summary = target / "summary.json"
            if summary.exists():
                last = read_checkpoint(target / "checkpoints/last.pt")
                value = read_json(summary)
                if (
                    value.get("last_checkpoint_sha256")
                    != before["checkpoints/last.pt"]["sha256"]
                    or value.get("run_id") != last["run_id"]
                    or value.get("step") != last["step"]
                ):
                    summary.unlink()
            last = _validate_run(target)
            if _files(source) != before:
                raise BackupError("run changed during validation; retry publication")
            source_files = _files(target)
            source_id = json_sha256(source_files)
            final_event = f"{started:.9f}" if started is not None else None
            final_publications = state.setdefault("final_publications", {})
            previous_final = final_publications.get(final_event)
            if previous_final is not None:
                if (
                    previous_final["run_id"] != last["run_id"]
                    or previous_final["checkpoint_sha256"]
                    != source_files["checkpoints/last.pt"]["sha256"]
                ):
                    raise BackupError(
                        "final publication event belongs to a different checkpoint"
                    )
                return previous_final["snapshot_id"]
            if final_event is None and source_id == state.get("latest_source"):
                state["published_at"] = time.time()
                _save_state(store, state)
                return state["latest_snapshot"]
            # The optimizer has already stopped for final allocation publication.
            # Charge measured local copy/verification wall time to the snapshot's
            # ledger without altering the durable source or checkpoint bytes.
            ledger = read_json(target / "budget.json")
            previous_job = max(ledger["job_seconds"], last["budget"]["job_seconds"])
            if summary.exists():
                previous_job = max(previous_job, read_json(summary)["job_seconds"])
            now = time.monotonic()
            charge = now - capture_started
            wrapper_charge = now - started if started is not None else charge
            ledger["job_seconds"] = previous_job + wrapper_charge
            write_json(target / "budget.json", ledger, overwrite=True)
            body = {
                "schema": SCHEMA,
                "run_id": last["run_id"],
                "step": last["step"],
                "source_files": source_files,
                "capture_job_seconds": charge,
                "wrapper_job_seconds": wrapper_charge,
                "final_event": final_event,
                "files": _files(target),
            }
            snapshot_id = json_sha256(body)
            destination = snapshots / snapshot_id
            if snapshot_id != state.get("latest_snapshot") and not destination.exists():
                if len(_pending(store, state)) >= MAX_PENDING:
                    raise BackupError(
                        "backup queue full; drain uploads and retry publication"
                    )
                write_json(
                    target / "manifest.json", {**body, "snapshot_id": snapshot_id}
                )
                target.rename(destination)
                _fsync_directory(snapshots)
            elif destination.exists():
                verify_snapshot(destination, snapshot_id)
            if final_event is not None:
                final_publications[final_event] = {
                    "run_id": last["run_id"],
                    "checkpoint_sha256": source_files["checkpoints/last.pt"]["sha256"],
                    "snapshot_id": snapshot_id,
                    "recorded_at": time.time(),
                }
                state["final_publications"] = dict(
                    sorted(
                        final_publications.items(),
                        key=lambda item: item[1]["recorded_at"],
                    )[-64:]
                )
            state.update(
                latest_snapshot=snapshot_id,
                latest_source=source_id,
                published_at=time.time(),
            )
            _save_state(store, state)
            return snapshot_id


def account_final(source, store, *, event=None, started=None):
    """Charge final wrapper work once, and recover an interrupted ledger write.

    The write-ahead record binds the exact old/new ledger bytes and checkpoint.
    It never overwrites a ledger that has diverged. Run reconciliation before
    staging scratch for an ordinary resume.
    """
    source, store = Path(source).absolute(), Path(store).absolute()
    with locked(str(source) + ".sync.lock"), locked(store / ".lock"):
        state = _state(store)
        charges = state.setdefault("final_charges", {})
        if event is not None and event not in charges:
            if (
                started is None
                or not math.isfinite(started)
                or started > time.monotonic()
            ):
                raise BackupError("invalid final-accounting start time")
            last = read_checkpoint(source / "checkpoints/last.pt")
            ledger = read_json(source / "budget.json")
            if (
                ledger["run_id"] != last["run_id"]
                or ledger["step"] < last["step"]
                or not all(_number(ledger[k]) for k in ("train_seconds", "job_seconds"))
                or ledger["train_seconds"] < last["budget"]["train_seconds"]
            ):
                raise BackupError("final accounting ledger identity mismatch")
            baseline = max(ledger["job_seconds"], last["budget"]["job_seconds"])
            summary = source / "summary.json"
            checkpoint_hash = file_sha256(source / "checkpoints/last.pt")
            if summary.exists():
                value = read_json(summary)
                if value.get("last_checkpoint_sha256") == checkpoint_hash:
                    if value.get("run_id") != last["run_id"] or not _number(
                        value.get("job_seconds")
                    ):
                        raise BackupError("final accounting summary identity mismatch")
                    baseline = max(baseline, value["job_seconds"])
            elapsed = time.monotonic() - started
            charges[event] = {
                "before_sha256": file_sha256(source / "budget.json"),
                "checkpoint_sha256": checkpoint_hash,
                "ledger": {**ledger, "job_seconds": baseline + elapsed},
                "wrapper_job_seconds": elapsed,
                "applied": False,
                "recorded_at": time.time(),
            }
            _save_state(store, state)
        for charge in charges.values():
            if charge["applied"]:
                continue
            ledger_path = source / "budget.json"
            current = read_json(ledger_path)
            if current == charge["ledger"]:
                # The ledger committed, but its acknowledgment did not.
                charge["applied"] = True
            elif (
                file_sha256(ledger_path) == charge["before_sha256"]
                and file_sha256(source / "checkpoints/last.pt")
                == charge["checkpoint_sha256"]
            ):
                write_json(ledger_path, charge["ledger"], overwrite=True)
                charge["applied"] = True
            else:
                raise BackupError(
                    "pending final charge conflicts with source; preserve both"
                )
            _save_state(store, state)
        # Keep pending records and a bounded recent idempotency window.
        completed = sorted(
            ((key, value) for key, value in charges.items() if value["applied"]),
            key=lambda item: item[1]["recorded_at"],
        )
        for key, _ in completed[:-64]:
            del charges[key]
        _save_state(store, state)
        return charges.get(event)


class Hub:
    """Official SDK only. Authentication comes from the SDK's saved token/env."""

    def __init__(self):
        from huggingface_hub import HfApi

        self.api = HfApi()

    def info(self, repo):
        if repo == "N-il/tennis-3d-synthetic":
            raise BackupError("dataset repository is not a checkpoint destination")
        info = self.api.repo_info(repo, repo_type="model")
        if not info.private:
            raise BackupError("backup requires an existing private model repository")
        return info.sha

    def exists_prefix(self, repo, commit, prefix):
        from huggingface_hub.errors import RemoteEntryNotFoundError

        try:
            entries = self.api.list_repo_tree(
                repo, path_in_repo=prefix, repo_type="model", revision=commit
            )
            return next(iter(entries), None) is not None
        except RemoteEntryNotFoundError:
            return False

    def commits(self, repo):
        return self.api.list_repo_commits(repo, repo_type="model")

    def download(self, repo, commit, name, cache):
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo, name, repo_type="model", revision=commit, cache_dir=cache
            )
        )

    def commit(self, repo, parent, snapshot_id, path, manifest):
        from huggingface_hub import CommitOperationAdd

        prefix = f"snapshots/{snapshot_id}"
        return self.api.create_commit(
            repo_id=repo,
            repo_type="model",
            parent_commit=parent,
            commit_message=f"tennis3d backup {snapshot_id}",
            operations=[
                CommitOperationAdd(
                    path_in_repo=f"{prefix}/{name}", path_or_fileobj=str(path / name)
                )
                for name in [*manifest["files"], "manifest.json"]
            ],
        ).oid


def _download(hub, repo, commit, snapshot_id, target, cache):
    if not HEX40.fullmatch(commit) or not HEX64.fullmatch(snapshot_id):
        raise BackupError("download requires exact commit and snapshot hashes")
    prefix = f"snapshots/{snapshot_id}"
    manifest_path = hub.download(repo, commit, f"{prefix}/manifest.json", cache)
    manifest = read_json(manifest_path)
    _check_manifest(manifest, snapshot_id)
    shutil.copyfile(manifest_path, target / "manifest.json")
    for name in manifest["files"]:
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(hub.download(repo, commit, f"{prefix}/{name}", cache), path)
    return verify_snapshot(target, snapshot_id)


def upload_pending(store, repo, *, hub=None):
    """One batched commit per snapshot. Reconcile lost replies before retrying."""
    store = Path(store).absolute()
    hub = hub or Hub()
    # Keep publication independent of network waits. Only uploaders share this lock.
    with locked(store / ".upload.lock"):
        parent = hub.info(repo)
        for abandoned in store.glob(".verify-*"):
            shutil.rmtree(abandoned)
        with locked(store / ".lock"):
            state = _state(store)
            if state.get("repo") not in (None, repo):
                raise BackupError("backup store is already bound to another repository")
            state["repo"] = repo
            _save_state(store, state)
            pending = _pending(store, state)
        for path in pending:
            snapshot_id = path.name
            manifest = verify_snapshot(path, snapshot_id)
            title = f"tennis3d backup {snapshot_id}"
            # The deterministic commit title and immutable content path survive an
            # uncertain create_commit outcome, process death, and lost local receipts.
            matches = [c for c in hub.commits(repo) if c.title == title]
            if matches:
                commit = matches[-1].commit_id
            elif hub.exists_prefix(repo, parent, f"snapshots/{snapshot_id}"):
                # A snapshot from another uploader may have a different commit
                # title. Verify it rather than overwriting its immutable path.
                commit = parent
            else:
                commit = hub.commit(repo, parent, snapshot_id, path, manifest)
            with tempfile.TemporaryDirectory(prefix=".verify-", dir=store) as temporary:
                root = Path(temporary)
                downloaded = root / "download"
                downloaded.mkdir()
                remote = _download(
                    hub, repo, commit, snapshot_id, downloaded, root / "cache"
                )
                if remote != manifest:
                    raise BackupError("remote snapshot differs from local publication")
            with locked(store / ".lock"):
                state = _state(store)
                state["receipts"][snapshot_id] = {
                    "repo": repo,
                    "commit": commit,
                    "manifest": manifest,
                    "uploaded_at": time.time(),
                }
                # Receipts for old snapshots remain in their immutable Hub commits.
                state["receipts"] = dict(
                    sorted(
                        state["receipts"].items(),
                        key=lambda item: item[1]["uploaded_at"],
                    )[-64:]
                )
                _save_state(store, state)
                completed = sorted(
                    (
                        p
                        for p in (store / "snapshots").iterdir()
                        if p.name in state["receipts"]
                    ),
                    key=lambda p: p.stat().st_mtime_ns,
                )
                for obsolete in completed[:-KEEP_UPLOADED]:
                    shutil.rmtree(obsolete)
            parent = hub.info(repo)
        return _state(store)["receipts"]


def restore(repo, commit, snapshot_id, output, *, run_id=None, hub=None):
    """Verify pinned bytes and resume identity before reserving a new output."""
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("restore requires a new output directory")
    hub = hub or Hub()
    hub.info(repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".restore-", dir=output.parent
    ) as temporary:
        root = Path(temporary)
        downloaded = root / "download"
        downloaded.mkdir()
        manifest = _download(hub, repo, commit, snapshot_id, downloaded, root / "cache")
        if run_id is not None and manifest["run_id"] != run_id:
            raise BackupError("snapshot belongs to another run")
        # mkdir refuses a competing output rather than merging/overwriting it.
        output.mkdir()
        try:
            for name in manifest["files"]:
                target = output / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(downloaded / name, target)
        except BaseException:
            shutil.rmtree(output)
            raise
    return {
        "repo": repo,
        "commit": commit,
        "snapshot_id": snapshot_id,
        "run_id": manifest["run_id"],
    }


def _status(store, command, *, error=None, retry_seconds=None):
    """Persist each operation's independent status without HTTP details."""
    with locked(Path(store) / ".lock"):
        state = _state(Path(store))
        statuses = state.setdefault("status", {})
        statuses[command] = {
            "ok": error is None,
            "updated_at": time.time(),
            "error": error,
            "retry_seconds": retry_seconds,
        }
        _save_state(Path(store), state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publication = commands.add_parser("publish")
    publication.add_argument("--source", required=True, type=Path)
    publication.add_argument("--store", required=True, type=Path)
    publication.add_argument("--interval", type=float, default=0)
    publication.add_argument(
        "--started", type=float, help="final wrapper monotonic start"
    )
    upload = commands.add_parser("upload")
    upload.add_argument("--store", required=True, type=Path)
    upload.add_argument("--repo", required=True)
    upload.add_argument("--watch", action="store_true")
    upload.add_argument("--poll-seconds", type=float, default=60)
    accounting = commands.add_parser("account")
    accounting.add_argument("--source", required=True, type=Path)
    accounting.add_argument("--store", required=True, type=Path)
    accounting.add_argument("--event")
    accounting.add_argument("--started", type=float)
    download = commands.add_parser("restore")
    download.add_argument("--repo", required=True)
    download.add_argument("--commit", required=True)
    download.add_argument("--snapshot", required=True)
    download.add_argument("--output", required=True, type=Path)
    download.add_argument("--run-id")
    args = parser.parse_args(argv)
    if args.command == "publish" and (
        not math.isfinite(args.interval) or args.interval < 0
    ):
        parser.error("--interval must be finite and nonnegative")
    if args.command == "upload" and (
        not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0
    ):
        parser.error("--poll-seconds must be finite and positive")
    delay = 5
    while True:
        try:
            if args.command == "publish":
                print(
                    publish(
                        args.source,
                        args.store,
                        interval=args.interval,
                        started=args.started,
                    )
                )
            elif args.command == "account":
                account_final(
                    args.source, args.store, event=args.event, started=args.started
                )
            elif args.command == "restore":
                print(
                    restore(
                        args.repo,
                        args.commit,
                        args.snapshot,
                        args.output,
                        run_id=args.run_id,
                    )
                )
            else:
                upload_pending(args.store, args.repo)
            if args.command != "restore":
                _status(args.store, args.command)
            delay = 5
        except Exception as error:  # noqa: BLE001
            # SDK exception text may contain request headers or signed URLs.
            # Emit a useful category, never arbitrary HTTP exception contents.
            diagnostic = (
                str(error) if isinstance(error, BackupError) else type(error).__name__
            )
            print(
                f"backup {args.command} failed ({diagnostic}); source preserved; retry required",
                file=sys.stderr,
                flush=True,
            )
            if args.command != "restore":
                try:
                    _status(
                        args.store,
                        args.command,
                        error=diagnostic,
                        retry_seconds=delay
                        if args.command == "upload" and args.watch
                        else None,
                    )
                except OSError as status_error:
                    print(
                        f"backup status recording failed ({type(status_error).__name__})",
                        file=sys.stderr,
                        flush=True,
                    )
            if args.command != "upload" or not args.watch:
                return 1
            print(f"backup upload retry in {delay}s", file=sys.stderr, flush=True)
            time.sleep(delay)
            delay = min(300, delay * 2)
            continue
        if args.command != "upload" or not args.watch:
            return 0
        if args.poll_seconds <= 0:
            parser.error("--poll-seconds must be positive")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
