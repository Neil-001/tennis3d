"""Pinned split sources and a synchronous, byte-bounded verified download cache."""

import fcntl
import hashlib
import os
import re
import time
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from tennis3d.artifacts import file_sha256, json_sha256
from tennis3d.datasets.io import (
    DEFAULT_MAX_GROUP_BYTES,
    ShardReader,
    _signature,
    read_manifest,
)
from tennis3d.datasets.schema import Artifact, Split, local_path
from tennis3d.schemas import SHA256, StrictModel

ORDER_VERSION = "shard-group-rally-view-v1"
VIEW_CACHE_BYTES = 4 << 30
RALLY_CACHE_BYTES = 16 << 20


class SourceSpec(StrictModel):
    source_version: Literal[1] = 1
    manifest: Artifact
    split: Split
    root: str | None = None
    repo_id: str | None = None
    revision: str | None = None
    subset_sha256: SHA256 | None = None
    exclude_rally_ids: list[str] = Field(default_factory=list)
    cache_dir: str | None = None
    cache_bytes: int = Field(default=1 << 30, gt=0)
    max_group_bytes: int = Field(default=DEFAULT_MAX_GROUP_BYTES, gt=0)
    retries: int = Field(default=2, ge=0, le=5)
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    transfer_seconds: float = Field(default=600, gt=0, le=1800)

    @model_validator(mode="after")
    def pinned(self):
        if any(not value for value in self.exclude_rally_ids) or len(
            set(self.exclude_rally_ids)
        ) != len(self.exclude_rally_ids):
            raise ValueError("excluded physical rally IDs must be nonempty and unique")
        if (self.repo_id is None) != (self.revision is None):
            raise ValueError("repo_id and immutable revision must be supplied together")
        if self.revision is not None and not re.fullmatch(
            "[0-9a-f]{40}", self.revision
        ):
            raise ValueError("source revision must be a full immutable commit SHA")
        if self.repo_id is not None and not re.fullmatch(
            r"[^/\s]+/[^/\s]+", self.repo_id
        ):
            raise ValueError("source requires namespace/repository")
        if self.root is None and (self.repo_id is None or self.cache_dir is None):
            raise ValueError("remote source requires pinned repo and owned cache_dir")
        return self


def hub_chunks(spec: SourceSpec, artifact: Artifact):
    """Read public Hub artifacts anonymously, without SDK/Xet disk caches."""
    import httpx
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers, get_session

    assert spec.repo_id is not None
    url = hf_hub_url(
        spec.repo_id, artifact.path, repo_type="dataset", revision=spec.revision
    )
    try:
        with get_session().stream(
            "GET",
            url,
            headers=build_hf_headers(token=False),
            follow_redirects=True,
            timeout=spec.timeout_seconds,
        ) as response:
            response.raise_for_status()
            deadline = time.monotonic() + spec.transfer_seconds
            stream = response.stream
            assert isinstance(stream, httpx.SyncByteStream)

            # Check each transport fragment before HTTPX buffers or decodes it.
            class DeadlineStream(httpx.SyncByteStream):
                def __iter__(self):
                    for chunk in stream:
                        if time.monotonic() > deadline:
                            raise OSError("download exceeded transfer deadline")
                        yield chunk

                def close(self):
                    stream.close()

            response.stream = DeadlineStream()
            for chunk in response.iter_bytes(chunk_size=64 << 10):
                if time.monotonic() > deadline:
                    raise OSError("download exceeded transfer deadline")
                yield chunk
    except (httpx.HTTPError, OSError):
        # Signed URLs and authentication details must not enter training logs.
        raise OSError("pinned Hub download failed") from None


class VerifiedCache:
    """One owner and one active reader. No prefetch or SDK on-disk state.

    The bound charges all regular-file contents including interrupted partials.
    A nonblocking lock prevents two processes from exceeding the same budget.
    Each source/split must use its own cache directory while open concurrently.
    """

    def __init__(self, root: Path, spec: SourceSpec):
        self.root, self.spec = root, spec
        root.mkdir(parents=True, exist_ok=True)
        self._check_paths()
        self.lock = (root / "lock").open("a+b")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise ValueError(
                "cache already in use; use a separate owned cache"
            ) from None
        self.pinned: Path | None = None
        self._reserve(0)
        self.peak_bytes = self.usage()

    def _check_paths(self):
        for p in (self.root, *self.root.rglob("*")):
            if p.is_symlink() or (p != self.root and not p.is_file()):
                raise ValueError("cache must contain only owned regular files")
            if (
                p != self.root
                and p.name != "lock"
                and not re.fullmatch(r"[0-9a-f]{64}(\.partial)?", p.name)
            ):
                raise ValueError("cache contains unrelated files")

    def usage(self):
        self._check_paths()
        return sum(p.stat().st_size for p in self.root.iterdir() if p.is_file())

    def path(self, artifact):
        key = json_sha256(
            {
                "repo": self.spec.repo_id,
                "commit": self.spec.revision,
                **artifact.model_dump(include={"path", "sha256", "size_bytes"}),
            }
        )
        return self.root / key

    def _reserve(self, size, keep=None):
        for entry in sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime_ns):
            if self.usage() + size <= self.spec.cache_bytes:
                return
            if entry.name != "lock" and entry not in (self.pinned, keep):
                entry.unlink()
        if self.usage() + size > self.spec.cache_bytes:
            raise ValueError("cache bound cannot accommodate pinned entry and fetch")

    def get(self, artifact):
        if artifact.size_bytes > self.spec.cache_bytes:
            raise ValueError("artifact exceeds cache byte bound before fetch")
        path = self.path(artifact)
        if path.exists():
            if (
                path.stat().st_size == artifact.size_bytes
                and file_sha256(path) == artifact.sha256
            ):
                self._reserve(0, keep=path)
                os.utime(path, None)
                return path
            if path == self.pinned:
                raise ValueError("pinned cache entry changed")
            path.unlink()
        partial = path.with_suffix(".partial")
        partial.unlink(missing_ok=True)
        self._reserve(artifact.size_bytes)
        for attempt in range(self.spec.retries + 1):
            try:
                digest, size = hashlib.sha256(), 0
                with partial.open("xb") as stream:
                    for chunk in hub_chunks(self.spec, artifact):
                        size += len(chunk)
                        if size > artifact.size_bytes:
                            raise ValueError("download exceeds declared artifact size")
                        stream.write(chunk)
                        digest.update(chunk)
                    if (
                        size != artifact.size_bytes
                        or digest.hexdigest() != artifact.sha256
                    ):
                        raise ValueError("download size/checksum mismatch")
                    stream.flush()
                    os.fsync(stream.fileno())
                self.peak_bytes = max(self.peak_bytes, self.usage())
                partial.replace(path)
                return path
            except OSError:
                if attempt == self.spec.retries:
                    raise
                time.sleep(min(0.1 * 2**attempt, 1))
            finally:
                partial.unlink(missing_ok=True)
        raise AssertionError("unreachable")

    def close(self):
        self.pinned = None
        self.lock.close()


class V2Source:
    def __init__(self, descriptor: Path, raw: dict):
        self.spec = SourceSpec.model_validate(raw)
        self.cache = None
        self.reader = None
        self.active = None
        self._rally_row = None
        self._rally_views = None
        self.views: dict[tuple[int, int], dict[int, dict]] = {}
        self.view_bytes = 0
        self.view_files: dict[int, tuple[Path, tuple]] = {}
        self.root = None
        self.descriptor = descriptor
        try:
            if self.spec.root is None:
                assert self.spec.cache_dir is not None
                cache_path = Path(self.spec.cache_dir)
                if not cache_path.is_absolute():
                    cache_path = descriptor.parent / cache_path
                self.cache = VerifiedCache(cache_path, self.spec)
                self.manifest_path = self.cache.get(self.spec.manifest)
            else:
                self.root = (descriptor.parent / self.spec.root).resolve()
                self.manifest_path = local_path(self.root, self.spec.manifest.path)
            if (
                self.manifest_path.stat().st_size != self.spec.manifest.size_bytes
                or file_sha256(self.manifest_path) != self.spec.manifest.sha256
            ):
                raise ValueError("source manifest artifact mismatch")
            self.manifest = read_manifest(self.manifest_path)
            self.subset_path = None
            if self.spec.subset_sha256 is not None:
                if self.root is None:
                    raise ValueError(
                        "compact subsets require verified local materialization"
                    )
                from tennis3d.datasets.subset import validate_source_subset

                validate_source_subset(
                    self.root,
                    self.spec.subset_sha256,
                    self.spec.manifest,
                    self.spec.repo_id,
                    self.spec.revision,
                )
                self.subset_path = self.root / "subset.json"
                self.subset_file_sha256 = file_sha256(self.subset_path)
            self.records, self.locations = [], []
            self.groups = {}
            excluded = set(self.spec.exclude_rally_ids)
            available = {
                rally.rally_id
                for shard in self.manifest.shards
                for rally in shard.records
                if rally.split == self.spec.split
            }
            if not excluded <= available:
                raise ValueError("excluded physical rally is absent from source split")
            for si, shard in enumerate(self.manifest.shards):
                for ri, rally in enumerate(shard.records):
                    if rally.split != self.spec.split or rally.rally_id in excluded:
                        continue
                    views = []
                    for vi, view in enumerate(rally.views):
                        views.append(len(self.records))
                        self.records.append(
                            {
                                **rally.model_dump(exclude={"views"}),
                                **view.model_dump(),
                                "shard_path": shard.path,
                                "shard_sha256": shard.sha256,
                                "row_index": ri,
                            }
                        )
                        self.locations.append((si, ri, vi))
                    self.groups.setdefault(si, {}).setdefault(
                        ri // shard.row_group_rallies, []
                    ).append(views)
            self.identity = {
                "format_version": 2,
                "repo_id": self.spec.repo_id,
                "revision": self.spec.revision,
                "manifest_sha256": self.spec.manifest.sha256,
                "split": self.spec.split,
                "membership_sha256": json_sha256(self.records),
                "order": ORDER_VERSION,
                "subset_sha256": self.spec.subset_sha256,
            }
        except BaseException:
            self.close()
            raise

    def order(self, rng):
        order = []
        for si in rng.permutation(list(self.groups)):
            groups = self.groups[int(si)]
            for gi in rng.permutation(list(groups)):
                rallies = groups[int(gi)]
                for ri in rng.permutation(len(rallies)):
                    order.extend(rng.permutation(rallies[int(ri)]).tolist())
        return order

    def read(self, index):
        si, ri, vi = self.locations[index]
        # Verified local compact subsets are small and fixed: decode each rally once.
        cached = self.views.get((si, ri))
        if cached is not None:
            path, signature = self.view_files[si]
            if _signature(path.stat()) != signature:
                raise ValueError("verified shard changed; reopen and verify it")
            return dict(cached[vi])
        if self.active != si:
            self._rally_row = self._rally_views = None
            if self.reader is not None:
                self.reader.close()
                self.reader = None
                self.active = None
            shard = self.manifest.shards[si]
            if self.cache:
                self.cache.pinned = None
                path = self.cache.get(shard)
                self.cache.pinned = path
            else:
                assert self.root is not None
                path = local_path(self.root, shard.path)
            self.reader = ShardReader(
                path,
                shard,
                provenance=self.manifest.provenance,
                max_group_bytes=self.spec.max_group_bytes,
            )
            self.active = si
        assert self.reader is not None
        if self.spec.subset_sha256 is None and self._rally_row == ri:
            # Order keeps a rally's camera views together. Reuse its validated
            # decode, but still reject file changes on every access. Copies keep
            # caller mutations from affecting later reads of the same rally.
            self.reader._unchanged()
            assert self._rally_views is not None
            return {key: array.copy() for key, array in self._rally_views[vi].items()}
        self._rally_row = self._rally_views = None
        record = self.reader.read(ri)
        if self.spec.subset_sha256 is None:
            size = sum(a.nbytes for view in record.views for a in view.arrays.values())
            if size <= min(RALLY_CACHE_BYTES, self.spec.max_group_bytes):
                self._rally_views = [view.arrays for view in record.views]
                self._rally_row = ri
                return {
                    key: array.copy() for key, array in record.views[vi].arrays.items()
                }
            return record.views[vi].arrays
        arrays = {k: dict(v.arrays) for k, v in enumerate(record.views)}
        size = sum(a.nbytes for views in arrays.values() for a in views.values())
        if self.view_bytes + size <= VIEW_CACHE_BYTES:
            for views in arrays.values():
                for a in views.values():
                    a.setflags(write=False)
            self.views[(si, ri)] = arrays
            self.view_files[si] = (self.reader.path, self.reader._signature)
            self.view_bytes += size
        return dict(arrays[vi])

    def close(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None
        if self.cache is not None:
            self.cache.close()
            self.cache = None
        self.views, self.view_bytes, self.view_files = {}, 0, {}
        self._rally_row = self._rally_views = None
        self.active = None
