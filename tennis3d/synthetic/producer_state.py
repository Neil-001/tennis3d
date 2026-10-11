"""Small durable producer checkpoints and explicit transaction ownership."""

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path

from tennis3d.artifacts import file_sha256, json_sha256, read_json, write_json
from tennis3d.datasets.schema import Artifact


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(path: Path, value, *, overwrite: bool = False) -> None:
    write_json(
        path, {"sha256": json_sha256(value), "value": value}, overwrite=overwrite
    )
    fsync_dir(path.parent)


def load(path: Path):
    envelope = read_json(path)
    if set(envelope) != {"sha256", "value"} or envelope["sha256"] != json_sha256(
        envelope["value"]
    ):
        raise ValueError(f"producer checkpoint checksum mismatch: {path.name}")
    return envelope["value"]


def save_once(path: Path, value) -> None:
    """Write a new checkpoint, or require the existing one to be identical."""
    if path.exists():
        if load(path) != value:
            raise ValueError(f"incompatible or conflicting checkpoint: {path.name}")
    else:
        save(path, value)


def artifact(root: Path, relative: str) -> Artifact:
    path = root / relative
    return Artifact(
        path=relative, sha256=file_sha256(path), size_bytes=path.stat().st_size
    )


def verify(root: Path, spec: Artifact) -> Path:
    path = root / spec.path
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != spec.size_bytes
        or file_sha256(path) != spec.sha256
    ):
        raise ValueError(f"owned artifact differs: {spec.path}")
    return path


def usage(root: Path) -> int:
    size = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("producer output cannot contain symlinks")
        if path.is_file():
            size += path.stat().st_size
    return size


@contextmanager
def coordinator_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".producer.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another coordinator owns this output") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
