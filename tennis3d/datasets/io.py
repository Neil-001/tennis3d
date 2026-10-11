"""Atomic Parquet shards and verified reader handles with bounded row-group IO."""

import hashlib
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tennis3d.artifacts import (
    atomic_file,
    canonical_json,
    json_sha256,
    read_json,
    write_json,
)
from tennis3d.datasets.arrays import RallyRecord, arrow_schema, decode_row, encode
from tennis3d.datasets.schema import Artifact, Manifest, Provenance, Shard

TARGET_BYTES = 384 * 1024**2
DEFAULT_MAX_GROUP_BYTES = 256 << 20


def _shard_schema(provenance: Provenance) -> pa.Schema:
    return arrow_schema(
        {
            b"tennis3d.schema_version": b"2.0",
            b"tennis3d.provenance": canonical_json(provenance.model_dump()),
        }
    )


class ShardWriter:
    """One writer per unique path. The size target is advisory, not a disk quota.

    At most row_group_rallies records are buffered. A single large record can
    exceed that bound in bytes; the producer must enforce its staging admission.
    Context exit closes the shard, or discards a failed partial file.
    """

    def __init__(
        self,
        path: str | Path,
        provenance: Provenance,
        *,
        target_bytes: int = TARGET_BYTES,
        row_group_rallies: int = 16,
        relative_path: str | None = None,
    ):
        if target_bytes <= 0 or row_group_rallies <= 0:
            raise ValueError("shard target and row group size must be positive")
        self.path = Path(path)
        self.relative_path = Artifact.relative_path(relative_path or self.path.name)
        self.provenance = Provenance.model_validate(provenance.model_dump())
        self.target_bytes, self.row_group_rallies = target_bytes, row_group_rallies
        self.records, self.pending = [], []
        self.pending_bytes = 0
        self.artifact = None
        self._atomic = atomic_file(self.path, readable=True)
        self._stream = self._atomic.__enter__()
        self.schema = _shard_schema(self.provenance)
        self._writer = pq.ParquetWriter(
            self._stream,
            self.schema,
            compression="zstd",
            use_dictionary=False,
            write_page_checksum=True,
        )
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if kind is None:
            self.close()
        else:
            self.abort()

    @property
    def target_reached(self) -> bool:
        return self._stream.tell() + self.pending_bytes >= self.target_bytes

    def append(self, record: RallyRecord) -> None:
        if self._closed:
            raise ValueError("shard writer is closed")
        from tennis3d.datasets.audit import audit_rally

        audit_rally(record)
        index = record.index()
        if any(r.rally_id == index.rally_id for r in self.records):
            raise ValueError("duplicate physical rally in shard")
        self.provenance.check_rally(index)
        self.pending.append(encode(record))
        self.pending_bytes += sum(a.nbytes for a in record.physical.values()) + sum(
            a.nbytes for v in record.views for a in v.arrays.values()
        )
        self.records.append(index)
        if len(self.pending) >= self.row_group_rallies:
            self._flush()

    def _flush(self):
        if self.pending:
            table = pa.Table.from_pylist(self.pending, schema=self.schema)
            self._writer.write_table(table, row_group_size=self.row_group_rallies)
            self.pending.clear()
            self.pending_bytes = 0

    def abort(self):
        if not self._closed:
            self._closed = True
            try:
                self._writer.close()
            finally:
                self._atomic.__exit__(ValueError, ValueError("aborted shard"), None)

    def close(self) -> Shard:
        if self.artifact is not None:
            return self.artifact
        if self._closed:
            raise ValueError("shard writer is closed")
        if not self.records:
            self.abort()
            raise ValueError("empty shards are not published")
        try:
            self._flush()
            self._writer.close()
            self._stream.flush()
            size = os.fstat(self._stream.fileno()).st_size
            self._stream.seek(0)
            digest = hashlib.sha256()
            while chunk := self._stream.read(1024 * 1024):
                digest.update(chunk)
            artifact = Shard(
                path=self.relative_path,
                sha256=digest.hexdigest(),
                size_bytes=size,
                row_group_rallies=self.row_group_rallies,
                records=self.records,
            )
            # No finalized path is visible until its complete descriptor is ready.
            self._atomic.__exit__(None, None, None)
        except BaseException:
            self.abort()
            raise
        self._closed = True
        self.artifact = artifact
        return self.artifact


def _signature(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class ShardReader:
    """Hash once per open handle, reject replacement/mutation on every read.

    Owns one file descriptor and at most one decoded row group. No global cache
    or unchecked public constructor exists. Consumers may retain this handle
    while a bounded cache pins the file. Mutation/replacement invalidates it.
    """

    def __init__(
        self,
        path: str | Path,
        shard: Shard,
        *,
        provenance: Provenance,
        max_group_bytes: int | None = None,
    ):
        self.path, self.shard = Path(path), shard
        self.max_group_bytes = max_group_bytes
        if max_group_bytes is not None and max_group_bytes <= 0:
            raise ValueError("decoded group byte bound must be positive")
        self._stream = self.path.open("rb")
        try:
            self._signature = _signature(os.fstat(self._stream.fileno()))
            if (
                self._signature[2] != shard.size_bytes
                or hashlib.file_digest(self._stream, "sha256").hexdigest()
                != shard.sha256
            ):
                raise ValueError("shard size/checksum mismatch")
            self._stream.seek(0)
            self._parquet = pq.ParquetFile(
                self._stream, page_checksum_verification=True
            )
            if not self._parquet.schema_arrow.equals(
                _shard_schema(provenance), check_metadata=True
            ):
                raise ValueError("shard schema/provenance mismatch")
            if self._parquet.metadata.num_rows != len(shard.records):
                raise ValueError("shard membership count mismatch")
            sizes = [
                self._parquet.metadata.row_group(i).num_rows
                for i in range(self._parquet.num_row_groups)
            ]
            expected = [
                min(shard.row_group_rallies, len(shard.records) - i)
                for i in range(0, len(shard.records), shard.row_group_rallies)
            ]
            if sizes != expected:
                raise ValueError("unexpected shard row-group layout")
            self._group, self._rows = None, None
            self._unchanged()
        except BaseException:
            self._stream.close()
            raise

    def _unchanged(self):
        if (
            self._stream.closed
            or _signature(os.fstat(self._stream.fileno())) != self._signature
            or _signature(self.path.stat()) != self._signature
        ):
            raise ValueError("verified shard changed; reopen and verify it")

    def _admit(self, nbytes: int):
        # Charge Arrow buffers plus conservative Python scalar/list overhead.
        # Only one row is decoded into Python rather than the whole group.
        if self.max_group_bytes is not None and (
            16 * nbytes + (1 << 20) > self.max_group_bytes
        ):
            raise ValueError("row group exceeds decoded memory admission bound")

    def read(self, row: int) -> RallyRecord:
        if not 0 <= row < len(self.shard.records):
            raise IndexError(row)
        self._unchanged()
        group, offset = divmod(row, self.shard.row_group_rallies)
        if group != self._group:
            self._admit(self._parquet.metadata.row_group(group).total_byte_size)
            self._group, self._rows = None, None
            table = self._parquet.read_row_group(group, use_threads=False)
            self._admit(table.nbytes)
            self._rows = table
            self._group = group
        assert self._rows is not None
        record = decode_row(self._rows.slice(offset, 1))
        if record.index() != self.shard.records[row]:
            raise ValueError("shard row membership differs from manifest")
        self._unchanged()
        return record

    def close(self):
        self._rows = None
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def read_rally(
    path: str | Path, shard: Shard, row: int, *, provenance: Provenance
) -> RallyRecord:
    with ShardReader(path, shard, provenance=provenance) as reader:
        return reader.read(row)


def iter_rallies(path: str | Path, shard: Shard, *, provenance: Provenance):
    with ShardReader(path, shard, provenance=provenance) as reader:
        for row in range(len(shard.records)):
            yield reader.read(row)


def write_manifest(
    path: str | Path, manifest: Manifest, *, overwrite: bool = False
) -> None:
    checked = Manifest.model_validate(manifest.model_dump())
    payload = checked.model_dump()
    write_json(
        path, {"manifest": payload, "sha256": json_sha256(payload)}, overwrite=overwrite
    )


def read_manifest(path: str | Path, *, require_complete: bool = True) -> Manifest:
    raw = read_json(path)
    if (
        set(raw) != {"manifest", "sha256"}
        or json_sha256(raw["manifest"]) != raw["sha256"]
    ):
        raise ValueError("manifest checksum/envelope mismatch")
    result = Manifest.model_validate(raw["manifest"])
    if require_complete and result.state != "complete":
        raise ValueError("dataset is incomplete")
    return result
