"""Atomic, checksummed complete-view predictions for interrupted final evaluation."""

import fcntl
import hashlib
import os
import signal
from pathlib import Path

import numpy as np

from tennis3d.artifacts import (
    _loads,
    atomic_file,
    canonical_json,
    json_sha256,
    read_json,
    write_json,
)


def arrays_sha256(arrays):
    """Bind dtype, shape and exact contiguous bytes, including masked NaNs."""
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        digest.update(canonical_json([key, array.dtype.str, list(array.shape)]))
        digest.update(array.tobytes())
    return digest.hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_directory(path):
    """Persist every new directory's link in its parent before publishing files."""
    if not path.exists():
        durable_directory(path.parent)
        path.mkdir()
    sync_directory(path.parent)
    sync_directory(path)


class PredictionJournal:
    """One writer; reject changed identities or damaged completed artifacts.

    A prediction, its input/output checksums and original inference duration
    commit together. No sidecar completion marker can be lost after inference.
    Replay still traverses and scores every declared view and both modes.
    """

    def __init__(self, root, identity, *, resume=False):
        self.root = Path(root).resolve()
        self.identity = identity
        self.counts = {
            "predictions_created": 0,
            "predictions_reused": 0,
            "physical_created": 0,
            "physical_reused": 0,
        }
        if resume:
            if not self.root.is_dir():
                raise ValueError("resume requires an existing evaluation directory")
            durable_directory(self.root)
        else:
            durable_directory(self.root.parent)
            self.root.mkdir(exist_ok=False)
            sync_directory(self.root.parent)
        self.lock = None
        self.stop_requested = False
        self.previous_signal = None

    def __enter__(self):
        self.lock = (self.root / "writer.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self.root / "identity.json"
            if path.exists():
                if read_json(path) != self.identity:
                    raise ValueError("evaluation resume identity differs")
            elif any(p.name != "writer.lock" for p in self.root.iterdir()):
                raise ValueError("incomplete evaluation identity; reuse refused")
            else:
                write_json(path, self.identity)
                sync_directory(self.root)
        except BaseException:
            self.lock.close()
            raise
        self.previous_signal = signal.signal(signal.SIGUSR1, self.request_stop)
        return self

    def request_stop(self, signum, frame):
        self.stop_requested = True

    def __exit__(self, *args):
        signal.signal(signal.SIGUSR1, self.previous_signal)
        assert self.lock is not None
        self.lock.close()

    def prediction(self, index, record, arrays, mode, compute):
        if self.stop_requested:
            raise InterruptedError(
                "evaluation warning received; resume saved predictions"
            )
        if mode not in ("clean", "corrupted") or index < 0:
            raise ValueError("invalid prediction journal view or mode")
        path = self.root / "predictions" / f"{index:05d}-{mode}.npz"
        expected = {
            "identity": self.identity,
            "index": index,
            "record": record,
            "mode": mode,
            "input_sha256": arrays_sha256(arrays),
        }
        durable_directory(path.parent)
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                metadata = _loads(archive["metadata"].tobytes())
                values = {k: archive[k] for k in ("xyz", "covered", "disagreement")}
            if (
                metadata["metadata_sha256"]
                != json_sha256(
                    {k: v for k, v in metadata.items() if k != "metadata_sha256"}
                )
                or metadata["key"] != expected
                or metadata["arrays_sha256"] != arrays_sha256(values)
            ):
                raise ValueError("prediction journal checksum or identity differs")
            self.counts["predictions_reused"] += 1
        else:
            xyz, covered, disagreement, wall_s = compute()
            values = {"xyz": xyz, "covered": covered, "disagreement": disagreement}
            metadata = {
                "key": expected,
                "arrays_sha256": arrays_sha256(values),
                "prediction_wall_s": wall_s,
            }
            metadata["metadata_sha256"] = json_sha256(metadata)
            with atomic_file(path) as stream:
                np.savez_compressed(
                    stream,
                    **values,
                    metadata=np.frombuffer(canonical_json(metadata), dtype=np.uint8),
                )
            sync_directory(path.parent)
            self.counts["predictions_created"] += 1
        return (
            values["xyz"],
            values["covered"],
            values["disagreement"],
            metadata["prediction_wall_s"],
        )

    def physical_report(self, index, compute):
        path = self.root / "physical-journal" / f"{index:05d}.json"
        durable_directory(path.parent)
        if path.exists():
            saved = read_json(path)
            if (
                saved["identity"] != self.identity
                or saved["index"] != index
                or saved["sha256"] != json_sha256(saved["report"])
            ):
                raise ValueError("physical journal checksum or identity differs")
            self.counts["physical_reused"] += 1
            return saved["report"]
        report = compute()
        write_json(
            path,
            {
                "identity": self.identity,
                "index": index,
                "report": report,
                "sha256": json_sha256(report),
            },
        )
        sync_directory(path.parent)
        self.counts["physical_created"] += 1
        return report
