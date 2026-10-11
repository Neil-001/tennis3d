"""Portable, bounded recovery archives for interrupted synthetic generation."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import BinaryIO, cast

_FORMAT = "tennis3d-producer-recovery"
_VERSION = 1
_MANIFEST_PATH = ".recovery/manifest.json"
_MANIFEST_HASH_PATH = ".recovery/manifest.sha256"
_RESERVED_PREFIX = ".recovery"
_COPY_CHUNK_BYTES = 1024 * 1024

_PRIVATE_DIRECTORIES = {
    ".aws",
    ".azure",
    ".gnupg",
    ".ssh",
    "auth",
    "authentication",
    "credentials",
    "tokens",
    "stored_tokens",
}
_CACHE_DIRECTORIES = {".cache", "huggingface", "huggingface_hub", "xet"}
_PRIVATE_FILES = {
    ".env",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "auth.json",
    "credentials",
    "credentials.json",
    "token",
    "token.json",
    "tokens.json",
}
_PRIVATE_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}


class RecoveryBundleError(ValueError):
    """The requested archive or restoration violates the recovery contract."""


class BundleSizeError(RecoveryBundleError):
    """The archive exceeds its configured byte limit."""


@dataclass(frozen=True)
class BundledFile:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class BundleArtifact:
    path: Path
    size_bytes: int
    sha256: str
    manifest_sha256: str
    files: tuple[BundledFile, ...]


@dataclass(frozen=True)
class RestoreReport:
    destination: Path
    bundle_size_bytes: int
    bundle_sha256: str
    manifest_sha256: str
    files: tuple[BundledFile, ...]


class _BoundedWriter:
    """Limit the largest byte offset written by ZipFile."""

    def __init__(self, file: BinaryIO, max_bytes: int):
        self._file = file
        self._max_bytes = max_bytes

    def write(self, data: bytes) -> int:
        end = self._file.tell() + len(data)
        if end > self._max_bytes:
            raise BundleSizeError(
                f"recovery archive exceeds max_bytes={self._max_bytes}"
            )
        return self._file.write(data)

    def tell(self) -> int:
        return self._file.tell()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._file.seek(offset, whence)

    def flush(self) -> None:
        self._file.flush()

    def __getattr__(self, name: str):
        return getattr(self._file, name)


def _validate_max_bytes(max_bytes: int) -> None:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise RecoveryBundleError("max_bytes must be a positive integer")


def _validate_relative_path(value: str, *, allow_metadata: bool = False) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise RecoveryBundleError(f"invalid recovery path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RecoveryBundleError(
            f"recovery paths must be canonical relative paths: {value!r}"
        )
    if ":" in value:
        raise RecoveryBundleError(
            f"drive-qualified recovery path is forbidden: {value!r}"
        )
    if path.parts[0] == _RESERVED_PREFIX:
        if allow_metadata and value in {_MANIFEST_PATH, _MANIFEST_HASH_PATH}:
            return value
        raise RecoveryBundleError(f"recovery path is reserved: {value!r}")
    _reject_private_path(path)
    return value


def _reject_private_path(path: PurePosixPath) -> None:
    lowered = tuple(part.casefold() for part in path.parts)
    if any(
        part in _PRIVATE_DIRECTORIES or part in _CACHE_DIRECTORIES for part in lowered
    ) or any(pair == (".publication", "sdk") for pair in pairwise(lowered)):
        raise RecoveryBundleError(
            f"credentials or SDK cache path is forbidden: {str(path)!r}"
        )
    name = lowered[-1]
    if (
        name in _PRIVATE_FILES
        or name.startswith((".env.", "id_rsa", "id_ed25519"))
        or PurePosixPath(name).suffix in _PRIVATE_SUFFIXES
    ):
        raise RecoveryBundleError(f"credential path is forbidden: {str(path)!r}")


def _source_file(root: Path, relative: str) -> tuple[BinaryIO, os.stat_result]:
    current = root
    if current.is_symlink() or not current.is_dir():
        raise RecoveryBundleError(f"recovery root must be a real directory: {root}")
    parts = PurePosixPath(relative).parts
    for part in parts[:-1]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError as error:
            raise RecoveryBundleError(
                f"recovery source does not exist: {relative!r}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RecoveryBundleError(
                f"recovery source parent is not a real directory: {relative!r}"
            )
    source = current / parts[-1]
    try:
        metadata = source.lstat()
    except FileNotFoundError as error:
        raise RecoveryBundleError(
            f"recovery source does not exist: {relative!r}"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RecoveryBundleError(
            f"recovery source must be a regular file: {relative!r}"
        )
    if metadata.st_nlink != 1:
        raise RecoveryBundleError(
            f"hard-linked recovery source is forbidden: {relative!r}"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise RecoveryBundleError(
            f"cannot open recovery source safely: {relative!r}"
        ) from error
    file = os.fdopen(descriptor, "rb")
    opened = os.fstat(file.fileno())
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
    ):
        file.close()
        raise RecoveryBundleError(
            f"recovery source changed while opening: {relative!r}"
        )
    return file, opened


def _zip_info(path: str, size: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.file_size = size
    return info


def _manifest_bytes(files: Sequence[BundledFile]) -> bytes:
    payload = {
        "files": [
            {"path": item.path, "sha256": item.sha256, "size_bytes": item.size_bytes}
            for item in files
        ],
        "format": _FORMAT,
        "version": _VERSION,
    }
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(_COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_bundle(
    root: Path,
    files: Sequence[str],
    destination: Path,
    max_bytes: int,
) -> BundleArtifact:
    """Archive exactly the allowlisted files without exceeding ``max_bytes``."""

    _validate_max_bytes(max_bytes)
    root = Path(root)
    destination = Path(destination)
    relative_files = tuple(_validate_relative_path(path) for path in files)
    if len(relative_files) != len(set(relative_files)):
        raise RecoveryBundleError("recovery allowlist contains duplicate paths")
    relative_files = tuple(sorted(relative_files))
    if destination.exists() or destination.is_symlink():
        raise RecoveryBundleError(f"bundle destination already exists: {destination}")
    if not destination.parent.is_dir():
        raise RecoveryBundleError(
            f"bundle destination parent does not exist: {destination.parent}"
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    records: list[BundledFile] = []
    try:
        with os.fdopen(descriptor, "w+b") as raw:
            bounded = _BoundedWriter(raw, max_bytes)
            with zipfile.ZipFile(
                cast(BinaryIO, bounded), "w", compression=zipfile.ZIP_STORED
            ) as archive:
                for relative in relative_files:
                    source, initial = _source_file(root, relative)
                    digest = hashlib.sha256()
                    copied = 0
                    with source:
                        with archive.open(
                            _zip_info(relative, initial.st_size), "w"
                        ) as target:
                            for chunk in iter(
                                lambda source=source: source.read(_COPY_CHUNK_BYTES),
                                b"",
                            ):
                                target.write(chunk)
                                digest.update(chunk)
                                copied += len(chunk)
                        final = os.fstat(source.fileno())
                    if copied != initial.st_size or (
                        final.st_size,
                        final.st_mtime_ns,
                        final.st_dev,
                        final.st_ino,
                    ) != (
                        initial.st_size,
                        initial.st_mtime_ns,
                        initial.st_dev,
                        initial.st_ino,
                    ):
                        raise RecoveryBundleError(
                            f"recovery source changed while archiving: {relative!r}"
                        )
                    records.append(BundledFile(relative, copied, digest.hexdigest()))
                manifest = _manifest_bytes(records)
                manifest_sha256 = hashlib.sha256(manifest).hexdigest()
                archive.writestr(_zip_info(_MANIFEST_PATH, len(manifest)), manifest)
                encoded_hash = (manifest_sha256 + "\n").encode("ascii")
                archive.writestr(
                    _zip_info(_MANIFEST_HASH_PATH, len(encoded_hash)), encoded_hash
                )
            raw.flush()
            os.fsync(raw.fileno())
        size = temporary.stat().st_size
        if size > max_bytes:
            raise BundleSizeError(
                f"recovery archive is {size} bytes, above max_bytes={max_bytes}"
            )
        archive_sha256 = _sha256_file(temporary)
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise RecoveryBundleError(
                f"bundle destination already exists: {destination}"
            ) from error
        temporary.unlink()
        directory = None
        try:
            directory = os.open(destination.parent, os.O_RDONLY)
            os.fsync(directory)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        finally:
            if directory is not None:
                os.close(directory)
        return BundleArtifact(
            destination,
            size,
            archive_sha256,
            manifest_sha256,
            tuple(records),
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RecoveryBundleError(f"duplicate manifest key: {key!r}")
        result[key] = value
    return result


def _read_manifest(
    archive: zipfile.ZipFile, max_bytes: int
) -> tuple[tuple[BundledFile, ...], str]:
    by_name = {info.filename: info for info in archive.infolist()}
    manifest_info = by_name.get(_MANIFEST_PATH)
    digest_info = by_name.get(_MANIFEST_HASH_PATH)
    if manifest_info is None or digest_info is None:
        raise RecoveryBundleError(
            "recovery archive lacks its manifest or manifest hash"
        )
    if manifest_info.file_size > max_bytes or digest_info.file_size != 65:
        raise RecoveryBundleError("invalid recovery manifest size")
    manifest = archive.read(manifest_info)
    recorded_hash = archive.read(digest_info)
    try:
        recorded_hash_text = recorded_hash.decode("ascii").removesuffix("\n")
    except UnicodeDecodeError as error:
        raise RecoveryBundleError("manifest hash is not ASCII") from error
    actual_hash = hashlib.sha256(manifest).hexdigest()
    if not hmac.compare_digest(recorded_hash_text, actual_hash):
        raise RecoveryBundleError("recovery manifest hash mismatch")
    try:
        payload = json.loads(manifest, object_pairs_hook=_reject_duplicate_json_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RecoveryBundleError("invalid recovery manifest JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"files", "format", "version"}:
        raise RecoveryBundleError("invalid recovery manifest fields")
    if (
        payload["format"] != _FORMAT
        or type(payload["version"]) is not int
        or payload["version"] != _VERSION
    ):
        raise RecoveryBundleError("unsupported recovery manifest format")
    raw_files = payload["files"]
    if not isinstance(raw_files, list):
        raise RecoveryBundleError("recovery manifest files must be a list")
    files: list[BundledFile] = []
    for raw in raw_files:
        if not isinstance(raw, dict) or set(raw) != {"path", "sha256", "size_bytes"}:
            raise RecoveryBundleError("invalid recovery manifest file record")
        path = _validate_relative_path(raw["path"])
        size = raw["size_bytes"]
        digest = raw["sha256"]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 <= size <= max_bytes
        ):
            raise RecoveryBundleError(f"invalid recovery file size for {path!r}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RecoveryBundleError(f"invalid recovery file hash for {path!r}")
        files.append(BundledFile(path, size, digest))
    paths = [item.path for item in files]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise RecoveryBundleError("recovery manifest paths must be unique and sorted")
    expected_names = paths + [_MANIFEST_PATH, _MANIFEST_HASH_PATH]
    if [info.filename for info in archive.infolist()] != expected_names:
        raise RecoveryBundleError("archive members do not exactly match the manifest")
    if sum(item.size_bytes for item in files) > max_bytes:
        raise RecoveryBundleError("recovery contents exceed max_bytes")
    return tuple(files), actual_hash


def _validate_members(archive: zipfile.ZipFile, max_bytes: int) -> None:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise RecoveryBundleError("recovery archive contains duplicate entries")
    for info in infos:
        _validate_relative_path(info.filename, allow_metadata=True)
        mode = info.external_attr >> 16
        if (
            info.create_system != 3
            or not stat.S_ISREG(mode)
            or info.is_dir()
            or info.flag_bits & 1
            or info.compress_type != zipfile.ZIP_STORED
            or info.file_size != info.compress_size
            or info.file_size > max_bytes
        ):
            raise RecoveryBundleError(
                f"recovery member is not a plain stored file: {info.filename!r}"
            )


def _prepare_destination(destination: Path) -> None:
    if destination.is_symlink():
        raise RecoveryBundleError(
            f"restore destination cannot be a symlink: {destination}"
        )
    if destination.exists():
        if not destination.is_dir():
            raise RecoveryBundleError(
                f"restore destination is not a directory: {destination}"
            )
        try:
            next(destination.iterdir())
        except StopIteration:
            return
        raise RecoveryBundleError(f"restore destination is not empty: {destination}")
    if not destination.parent.is_dir():
        raise RecoveryBundleError(
            f"restore destination parent does not exist: {destination.parent}"
        )


def restore_bundle(bundle: Path, destination: Path, max_bytes: int) -> RestoreReport:
    """Verify a recovery archive and atomically expose it at ``destination``."""

    _validate_max_bytes(max_bytes)
    bundle = Path(bundle)
    destination = Path(destination)
    try:
        bundle_metadata = bundle.lstat()
    except FileNotFoundError as error:
        raise RecoveryBundleError(
            f"recovery archive does not exist: {bundle}"
        ) from error
    if stat.S_ISLNK(bundle_metadata.st_mode) or not stat.S_ISREG(
        bundle_metadata.st_mode
    ):
        raise RecoveryBundleError(f"recovery archive must be a regular file: {bundle}")
    if bundle_metadata.st_size > max_bytes:
        raise BundleSizeError(
            f"recovery archive is {bundle_metadata.st_size} bytes, above max_bytes={max_bytes}"
        )
    _prepare_destination(destination)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.restore-", dir=destination.parent)
    )
    try:
        try:
            with zipfile.ZipFile(bundle, "r") as archive:
                _validate_members(archive, max_bytes)
                files, manifest_sha256 = _read_manifest(archive, max_bytes)
                for record in files:
                    target = staging.joinpath(*PurePosixPath(record.path).parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    copied = 0
                    with (
                        archive.open(record.path, "r") as source,
                        target.open("xb") as output,
                    ):
                        for chunk in iter(
                            lambda source=source: source.read(_COPY_CHUNK_BYTES), b""
                        ):
                            copied += len(chunk)
                            if copied > record.size_bytes:
                                raise RecoveryBundleError(
                                    f"recovery file exceeds declared size: {record.path!r}"
                                )
                            output.write(chunk)
                            digest.update(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                    if copied != record.size_bytes or not hmac.compare_digest(
                        digest.hexdigest(), record.sha256
                    ):
                        raise RecoveryBundleError(
                            f"recovery file hash or size mismatch: {record.path!r}"
                        )
        except (zipfile.BadZipFile, EOFError) as error:
            raise RecoveryBundleError("invalid or damaged recovery archive") from error
        bundle_sha256 = _sha256_file(bundle)
        final_metadata = bundle.lstat()
        if (
            final_metadata.st_dev,
            final_metadata.st_ino,
            final_metadata.st_size,
            final_metadata.st_mtime_ns,
        ) != (
            bundle_metadata.st_dev,
            bundle_metadata.st_ino,
            bundle_metadata.st_size,
            bundle_metadata.st_mtime_ns,
        ):
            raise RecoveryBundleError("recovery archive changed while restoring")
        _prepare_destination(destination)
        os.replace(staging, destination)
        return RestoreReport(
            destination,
            bundle_metadata.st_size,
            bundle_sha256,
            manifest_sha256,
            files,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
