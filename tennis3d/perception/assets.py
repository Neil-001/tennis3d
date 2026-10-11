"""Explicit, hash-checked asset retrieval. Importing this module performs no I/O."""

import hashlib
import urllib.request
from pathlib import Path

from pydantic import Field

from tennis3d.artifacts import atomic_file, file_sha256
from tennis3d.config import load_config
from tennis3d.schemas import SHA256, StrictModel


class MissingAssetError(FileNotFoundError):
    """A required inference checkpoint is unavailable locally."""


class AssetIntegrityError(ValueError):
    """A checkpoint does not match its declared size or SHA-256."""


class Asset(StrictModel):
    url: str
    path: Path
    sha256: SHA256
    bytes: int = Field(gt=0)
    source: str
    revision: str
    license: str
    purpose: str


class AssetManifest(StrictModel):
    version: int = 1
    assets: dict[str, Asset]


def manifest(path):
    return load_config(AssetManifest, path)


def verify(asset: Asset):
    if not asset.path.is_file():
        raise MissingAssetError(
            f"Missing asset {asset.path}; run tennis3d assets fetch"
        )
    if (
        asset.path.stat().st_size != asset.bytes
        or file_sha256(asset.path) != asset.sha256
    ):
        raise AssetIntegrityError(f"Asset integrity mismatch: {asset.path}")


def fetch(asset: Asset, *, overwrite=False):
    if asset.path.exists() and not overwrite:
        verify(asset)
        return
    if not asset.url.startswith("https://"):
        raise ValueError("Asset downloads require an HTTPS URL")
    digest = hashlib.sha256()
    count = 0
    with atomic_file(asset.path, overwrite=overwrite) as target:
        with urllib.request.urlopen(asset.url, timeout=60) as response:
            while block := response.read(1024 * 1024):
                count += len(block)
                if count > asset.bytes:
                    raise AssetIntegrityError("Asset exceeds its declared size")
                digest.update(block)
                target.write(block)
        if count != asset.bytes or digest.hexdigest() != asset.sha256:
            raise AssetIntegrityError("Downloaded asset does not match manifest")
