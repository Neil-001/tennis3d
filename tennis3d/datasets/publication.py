"""Bounded synchronous staging and verified, restartable Hub publication.

The staging lock serializes cooperating producers and publishers. A full staging
area raises StagingFull before advancing a producer iterator. Callers must pause
production and retry after publication frees space. Arbitrary external writes to
the staging directory are outside this contract.

Hub transfers run in a fresh Linux subprocess with Xet disabled before SDK import.
RLIMIT_FSIZE=0 forbids disk writes, including SDK caches and temporary downloads.
Upload uses the supported HTTP/LFS path; readback hashes a streamed HTTP response.
The adapter uses explicit or standard cached SDK authentication and requires an
existing dataset repository.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol

from tennis3d.artifacts import canonical_json, json_sha256
from tennis3d.datasets.schema import Artifact


class ArtifactLike(Protocol):
    path: str
    sha256: str
    size_bytes: int


class StagingFull(RuntimeError):
    """Pause the producer until verified publication frees local space."""


class VerificationError(RuntimeError):
    pass


class RemoteMissing(FileNotFoundError):
    pass


class PublicationBackend(Protocol):
    def head(self, repo_id: str, revision: str) -> str: ...

    def upload(
        self, repo_id: str, revision: str, parent: str, path: str, local: Path
    ) -> str:
        """Add only path to parent atomically; fail if revision no longer equals parent."""
        ...

    def fingerprint(self, repo_id: str, commit: str, path: str) -> tuple[str, int]:
        """Hash and count streamed file contents at the exact commit."""
        ...


def _artifact(artifact: ArtifactLike) -> dict[str, Any]:
    return Artifact(
        path=artifact.path, sha256=artifact.sha256, size_bytes=artifact.size_bytes
    ).model_dump()


def _commit(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise VerificationError("Hub did not return a full immutable commit SHA")
    return value


def _fingerprint(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class BoundedStaging:
    """Disk budget includes payload, partial files and two ledger generations.

    ``max_bytes`` accounts for regular-file contents, not filesystem allocation
    blocks or directory metadata. Reserve at least twice the ledger size limit.
    Files from interrupted producers remain charged to the budget for inspection.
    """

    def __init__(
        self, root: Path, max_bytes: int, *, metadata_reserve_bytes: int = 8 << 20
    ):
        if metadata_reserve_bytes < 1024 or max_bytes <= metadata_reserve_bytes:
            raise ValueError(
                "Budget must exceed the metadata reserve of at least 1024 bytes"
            )
        self.root = Path(root).absolute()
        self.max_bytes = max_bytes
        self.metadata_reserve_bytes = metadata_reserve_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        self.files = self.root / "files"
        self.control = self.root / ".publication"
        self.files.mkdir(exist_ok=True)
        self.control.mkdir(exist_ok=True)
        self._usage()  # Reject symlinks before any payload operations.

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        with (self.control / "lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _usage(self) -> int:
        total = 0
        for path in (self.root, *self.root.rglob("*")):
            if path.is_symlink():
                raise ValueError("Staging must not contain symlinks")
            if path.is_file():
                total += path.stat().st_size
        return total

    def local_path(self, artifact: ArtifactLike) -> Path:
        spec = _artifact(artifact)
        path = self.files / spec["path"]
        for parent in (path, *path.parents):
            if parent.is_symlink():
                raise ValueError("Staged paths must not contain symlinks")
            if parent == self.root:
                break
        return path

    def stage(self, artifact: ArtifactLike, chunks: Iterable[bytes]) -> Path:
        """Consume at most the declared size, checking admission first."""
        spec = _artifact(artifact)
        with self.locked():
            path = self.local_path(artifact)
            if path.exists():
                if _fingerprint(path) != (spec["sha256"], spec["size_bytes"]):
                    raise VerificationError(
                        "Existing staged file differs from artifact"
                    )
                return path
            # Reserve both the old and new ledger, even when neither exists yet.
            control_size = sum(
                p.stat().st_size for p in self.control.rglob("*") if p.is_file()
            )
            charged = (
                self._usage()
                - control_size
                + max(control_size, self.metadata_reserve_bytes)
            )
            if charged + spec["size_bytes"] > self.max_bytes:
                raise StagingFull(
                    "Staging budget exhausted; pause producer and publish queued files"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
            partial = Path(name)
            try:
                digest, size = hashlib.sha256(), 0
                with os.fdopen(fd, "wb") as stream:
                    for chunk in chunks:
                        if not isinstance(chunk, bytes):
                            raise TypeError("Staging chunks must be bytes")
                        size += len(chunk)
                        if size > spec["size_bytes"]:
                            raise VerificationError(
                                "Producer exceeded declared artifact size"
                            )
                        stream.write(chunk)
                        digest.update(chunk)
                    if (digest.hexdigest(), size) != (
                        spec["sha256"],
                        spec["size_bytes"],
                    ):
                        raise VerificationError("Produced bytes do not match artifact")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(partial, path)
                _fsync_dir(path.parent)
            finally:
                partial.unlink(missing_ok=True)
            return path


class Publisher:
    """Receipts become durable before local eviction; failures retain local data."""

    def __init__(
        self,
        staging: BoundedStaging,
        backend: PublicationBackend,
        repo_id: str,
        revision: str = "main",
    ):
        if not re.fullmatch(r"[^/\s]+/[^/\s]+", repo_id) or not revision:
            raise ValueError(
                "An explicit namespace/repository and revision are required"
            )
        self.staging, self.backend = staging, backend
        self.repo_id, self.revision = repo_id, revision
        self.ledger_path = staging.control / "receipts.json"

    def _load(self) -> dict[str, Any]:
        if not self.ledger_path.exists():
            return {
                "version": 1,
                "repo_id": self.repo_id,
                "revision": self.revision,
                "entries": {},
            }
        limit = self.staging.metadata_reserve_bytes // 2
        try:
            with self.ledger_path.open("rb") as stream:
                raw = stream.read(limit + 1)
            if len(raw) > limit:
                raise ValueError("Oversized receipt ledger")
            envelope = json.loads(raw)
            ledger = envelope["ledger"]
            if json_sha256(ledger) != envelope["sha256"]:
                raise ValueError("Receipt checksum mismatch")
            if (ledger["version"], ledger["repo_id"], ledger["revision"]) != (
                1,
                self.repo_id,
                self.revision,
            ):
                raise ValueError("Receipt ledger belongs to another publication")
            return ledger
        except (ValueError, KeyError, TypeError) as exc:
            raise VerificationError(
                "Invalid receipt ledger; retaining local artifacts"
            ) from exc

    def _save(self, ledger: dict[str, Any]) -> None:
        raw = canonical_json({"ledger": ledger, "sha256": json_sha256(ledger)})
        if len(raw) > self.staging.metadata_reserve_bytes // 2:
            raise StagingFull("Receipt ledger reserve exhausted; pause publication")
        temporary = self.staging.control / "receipts.tmp"
        with temporary.open("wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.ledger_path)
        _fsync_dir(self.staging.control)

    def _verify(self, spec: dict[str, Any], commit: str) -> None:
        actual = self.backend.fingerprint(self.repo_id, _commit(commit), spec["path"])
        if actual != (spec["sha256"], spec["size_bytes"]):
            raise VerificationError(
                f"Remote content mismatch at {commit}: {spec['path']}"
            )

    def _commit_file(
        self, ledger: dict[str, Any], spec: dict[str, Any], local: Path, head: str
    ) -> str:
        """Record intent, then reuse head if a lost response hid a prior commit."""
        ledger["entries"][spec["path"]] = {**spec, "state": "pending"}
        self._save(ledger)
        try:
            self._verify(spec, head)
            commit = head
        except (RemoteMissing, VerificationError):
            commit = _commit(
                self.backend.upload(
                    self.repo_id, self.revision, head, spec["path"], local
                )
            )
        ledger["entries"][spec["path"]] = {
            **spec,
            "state": "uploaded",
            "commit": commit,
        }
        self._save(ledger)
        return commit

    def publish(self, artifact: ArtifactLike, *, evict: bool = False) -> str:
        spec = _artifact(artifact)
        with self.staging.locked():
            ledger = self._load()
            entry = ledger["entries"].get(spec["path"])
            if entry and any(entry[key] != value for key, value in spec.items()):
                raise VerificationError(
                    "An artifact path cannot change within a release"
                )
            local = self.staging.local_path(artifact)
            if entry and entry.get("commit"):
                commit = _commit(entry["commit"])
                self._verify(spec, commit)  # Recheck even a receipt before eviction.
            else:
                if not local.is_file() or _fingerprint(local) != (
                    spec["sha256"],
                    spec["size_bytes"],
                ):
                    raise VerificationError(
                        "A verified local artifact is required for upload"
                    )
                if "complete" in ledger:
                    raise VerificationError(
                        "Cannot add artifacts to a completed release"
                    )
                head = _commit(self.backend.head(self.repo_id, self.revision))
                commit = self._commit_file(ledger, spec, local, head)
                self._verify(spec, commit)
            ledger["entries"][spec["path"]] = {
                **spec,
                "state": "verified",
                "commit": commit,
            }
            self._save(ledger)
            if evict and local.exists():
                # Refuse to remove a replacement or externally modified file.
                if _fingerprint(local) != (spec["sha256"], spec["size_bytes"]):
                    raise VerificationError("Local file changed; refusing eviction")
                local.unlink()
                _fsync_dir(local.parent)
            return commit

    def publish_release(
        self,
        manifest: ArtifactLike,
        expected: Sequence[ArtifactLike],
        audit: ArtifactLike,
        validate_release: Callable[[Path, Path], None],
    ) -> str:
        """Commit the finalized manifest last, after the audit and content pass.

        ``expected`` excludes the final manifest and includes the audit, data
        card, license, indexes and data shards. The trusted validator must check
        the manifest produced by ``tennis3d.datasets.audit.finalize_manifest``, its
        expected artifact set, and its full audit. It must raise if any of those
        disagree. A parent-commit constraint
        prevents another writer changing the validated tree before publication.
        """
        spec = _artifact(manifest)
        specs = [_artifact(item) for item in expected]
        audit_spec = _artifact(audit)
        paths = {item["path"] for item in specs}
        if (
            not specs
            or len(paths) != len(specs)
            or spec["path"] in paths
            or audit_spec not in specs
        ):
            raise VerificationError(
                "Release artifacts must be unique and include the audit"
            )
        with self.staging.locked():
            ledger = self._load()
            if set(ledger["entries"]) - {spec["path"]} != paths:
                raise VerificationError("Published and expected artifact sets differ")
            for item in specs:
                entry = ledger["entries"][item["path"]]
                if entry["state"] != "verified" or any(
                    entry[k] != v for k, v in item.items()
                ):
                    raise VerificationError(
                        "Release needs matching verified artifact receipts"
                    )
            previous = ledger["entries"].get(spec["path"])
            if previous and any(previous[k] != v for k, v in spec.items()):
                raise VerificationError("Final manifest cannot change within a release")
            local, audit_path = (
                self.staging.local_path(manifest),
                self.staging.local_path(audit),
            )
            for path, item in ((local, spec), (audit_path, audit_spec)):
                if _fingerprint(path) != (item["sha256"], item["size_bytes"]):
                    raise VerificationError(
                        "Release metadata differs from expected bytes"
                    )
            validate_release(local, audit_path)
            head = _commit(self.backend.head(self.repo_id, self.revision))
            for item in specs:
                self._verify(item, head)
            commit = self._commit_file(ledger, spec, local, head)
            # Upload adds only the manifest to the verified immutable parent tree.
            # Reusing head also preserves that tree, so only the manifest needs reading.
            self._verify(spec, commit)
            ledger["entries"][spec["path"]]["state"] = "verified"
            ledger["complete"] = {
                "commit": commit,
                "audit_sha256": audit.sha256,
                "manifest_sha256": manifest.sha256,
            }
            self._save(ledger)
            return commit


# This code runs in a clean interpreter: SDK environment settings are read at import.
_HUB_WORKER = r"""
import hashlib, json, resource, sys
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_url
from huggingface_hub.utils import build_hf_headers, get_session
request = json.load(sys.stdin)
api = HfApi(token=request["token"])
repo, action = request["repo_id"], request["action"]
if action == "head":
    result = api.repo_info(repo_id=repo, repo_type="dataset", revision=request["revision"]).sha
elif action == "upload":
    result = api.create_commit(
        repo_id=repo, repo_type="dataset", revision=request["revision"],
        parent_commit=request["parent"],
        operations=[CommitOperationAdd(path_in_repo=request["path"], path_or_fileobj=request["local"])],
        commit_message="Publish verified dataset artifact", num_threads=1,
    ).oid
elif action == "fingerprint":
    url = hf_hub_url(repo_id=repo, repo_type="dataset", revision=request["commit"], filename=request["path"])
    digest, size = hashlib.sha256(), 0
    with get_session().stream("GET", url, headers=build_hf_headers(token=request["token"]), follow_redirects=True) as response:
        if response.status_code == 404:
            print(json.dumps({"missing": True}))
            sys.exit(0)
        response.raise_for_status()
        for chunk in response.iter_bytes(chunk_size=1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    result = [digest.hexdigest(), size]
else:
    raise ValueError("Unknown Hub operation")
print(json.dumps({"result": result}))
"""


class HuggingFaceBackend:
    """Official huggingface_hub 1.x adapter; constructor performs no network I/O.

    Credentials pass over stdin, never argv. Transfers are synchronous. An upload
    exception is ambiguous and safe to retry through Publisher.publish. Tokens
    are resolved with the SDK before isolating HF_HOME for the worker.
    No SDK download cache is used, and the worker cannot create nonempty files.
    """

    def __init__(
        self,
        staging: BoundedStaging,
        *,
        token: str | None = None,
        timeout: float = 3600,
    ):
        if token == "":
            raise ValueError("An explicit Hub token must not be empty")
        self.staging, self.token, self.timeout = staging, token, timeout

    def _run(self, action: str, repo_id: str, **values: str) -> Any:
        # One fixed directory, with no growing collection of per-call caches.
        cache = self.staging.control / "sdk"
        cache.mkdir(exist_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "HF_HOME": str(cache),
                "HF_HUB_CACHE": str(cache / "hub"),
                "HF_XET_CACHE": str(cache / "xet"),
                "HF_ASSETS_CACHE": str(cache / "assets"),
                "HF_HUB_DISABLE_XET": "1",
                "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
                "HF_XET_SHARD_CACHE_SIZE_LIMIT": "0",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "TMPDIR": str(cache),
            }
        )
        token = self.token
        if token is None:
            from huggingface_hub import get_token

            token = get_token()
        if token is None:
            raise RuntimeError(
                "Hub authentication is required; use the standard SDK login"
            )
        request = {"action": action, "repo_id": repo_id, "token": token, **values}
        try:
            result = subprocess.run(
                [sys.executable, "-c", _HUB_WORKER],
                input=canonical_json(request),
                capture_output=True,
                env=env,
                timeout=self.timeout,
                check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            # Do not expose tokens, signed URLs or credential-bearing exceptions.
            raise RuntimeError(
                f"Hub {action} failed; retain local data and retry"
            ) from None
        response = json.loads(result.stdout)
        if response.get("missing"):
            raise RemoteMissing(values.get("path", ""))
        return response["result"]

    def head(self, repo_id: str, revision: str) -> str:
        return _commit(self._run("head", repo_id, revision=revision))

    def upload(
        self, repo_id: str, revision: str, parent: str, path: str, local: Path
    ) -> str:
        return _commit(
            self._run(
                "upload",
                repo_id,
                revision=revision,
                parent=_commit(parent),
                path=path,
                local=str(local.absolute()),
            )
        )

    def fingerprint(self, repo_id: str, commit: str, path: str) -> tuple[str, int]:
        digest, size = self._run(
            "fingerprint", repo_id, commit=_commit(commit), path=path
        )
        return digest, size
