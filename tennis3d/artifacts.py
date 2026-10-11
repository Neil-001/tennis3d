"""Atomic local files and numeric NPZ artifacts. No pickle or implicit cache reuse."""

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np
from pydantic import Field

from tennis3d.schemas import SHA256, StrictModel


@contextmanager
def atomic_file(
    path: str | Path, *, overwrite: bool = False, readable: bool = False
) -> Iterator[BinaryIO]:
    """Publish a fully written file atomically; existing files are protected by default."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w+b" if readable else "wb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Atomic no-clobber publication on the same filesystem.
            os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def json_sha256(value) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: str | Path, value, *, overwrite: bool = False) -> None:
    payload = canonical_json(value) + b"\n"
    with atomic_file(path, overwrite=overwrite) as stream:
        stream.write(payload)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number: {token}")
    return value


def _loads(text: str | bytes):
    return json.loads(
        text,
        object_pairs_hook=_unique_pairs,
        parse_constant=_invalid_constant,
        parse_float=_finite_float,
    )


def read_json(path: str | Path):
    return _loads(Path(path).read_text(encoding="utf-8"))


class ArtifactProvenance(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    source_sha256: SHA256
    config_sha256: SHA256
    upstream_sha256: dict[str, SHA256] = Field(default_factory=dict)
    code_revision: str


def _numeric_arrays(arrays):
    checked = {}
    for key, value in arrays.items():
        if (
            not key
            or not key.replace("_", "").isalnum()
            or key.startswith("__")
            or key in {"file", "allow_pickle"}
        ):
            raise ValueError(f"invalid or reserved array name: {key}")
        array = np.asarray(value)
        if array.dtype.kind not in "biufc":
            raise ValueError(
                f"{key}: only boolean/integer/float/complex numeric arrays are allowed"
            )
        checked[key] = array
    return checked


def write_npz(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    provenance: ArtifactProvenance,
    *,
    overwrite: bool = False,
) -> None:
    """Write numeric arrays plus JSON provenance encoded as reserved uint8 bytes.

    NaN is allowed in numeric arrays for masked observations. Stage consumers
    validate shapes and masks. Metadata and arrays commit together in one file.
    """
    checked = _numeric_arrays(arrays)
    metadata = ArtifactProvenance.model_validate(provenance.model_dump())
    checked["__provenance__"] = np.frombuffer(
        canonical_json(metadata.model_dump()), dtype=np.uint8
    )
    with atomic_file(path, overwrite=overwrite) as stream:
        np.savez_compressed(stream, **checked)


def read_npz(
    path: str | Path, *, expected: ArtifactProvenance | None = None
) -> tuple[dict[str, np.ndarray], ArtifactProvenance]:
    """Read with allow_pickle=False; optionally require exact provenance for reuse."""
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("duplicate NPZ entries")
        raw = archive["__provenance__"]
        if raw.dtype != np.uint8 or raw.ndim != 1:
            raise ValueError("provenance must be a uint8 JSON byte vector")
        metadata = ArtifactProvenance.model_validate(_loads(raw.tobytes()))
        arrays = _numeric_arrays(
            {k: archive[k] for k in archive.files if k != "__provenance__"}
        )
    if expected is not None and metadata != expected:
        raise ValueError("artifact provenance differs; cache reuse refused")
    return arrays, metadata
