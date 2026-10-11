"""Installed application directory: verified weights, settings and owned runs.

The directory is `TENNIS3D_HOME` when set, otherwise `~/.tennis3d`. Packaged
defaults are materialized there with absolute paths, so the command's working
directory never selects weights or configuration.
"""

import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
from importlib import resources
from pathlib import Path

import yaml

from tennis3d.artifacts import atomic_file, file_sha256
from tennis3d.inference.settings import lifting_device
from tennis3d.perception.assets import (
    Asset,
    AssetIntegrityError,
    MissingAssetError,
    manifest,
    verify,
)

SETUP_HINT = "run `tennis3d setup`"


def home(path=None) -> Path:
    return Path(
        path or os.environ.get("TENNIS3D_HOME") or Path.home() / ".tennis3d"
    ).resolve()


def resource(name: str) -> bytes:
    return resources.files("tennis3d.resources").joinpath(name).read_bytes()


def lifter_manifest() -> dict:
    return json.loads(resource("lifter.json"))


def lifter_path(root: Path) -> Path:
    return root / "weights" / lifter_manifest()["file"]


def detector_manifest(root: Path) -> Path:
    """Write the packaged detector manifest with weights below this directory."""
    values = yaml.safe_load(resource("assets.yaml"))
    for asset in values["assets"].values():
        asset["path"] = str(root / "weights" / Path(asset["path"]).name)
    payload = yaml.safe_dump(values, sort_keys=True).encode()
    target = root / "assets.yaml"
    if not target.is_file() or target.read_bytes() != payload:
        with atomic_file(target, overwrite=True) as stream:
            stream.write(payload)
    return target


def installed_config(root: Path, device="auto") -> dict:
    """Reconstruction settings for the packaged model and installed weights."""
    values = yaml.safe_load(resource("reconstruction.yaml"))
    # Detector convolutions dominate CPU time and scale with physical cores.
    threads = min(8, max(2, (os.cpu_count() or 2) // 2))
    values["detection"] |= {
        "threads": threads,
        "assets_manifest": str(detector_manifest(root)),
        "device": device,
    }
    values["lifting"] |= {
        "threads": threads,
        "checkpoint": str(lifter_path(root)),
        "device": lifting_device(device),
        "model_label": lifter_manifest()["model_label"],
    }
    return values


def _matches(path: Path, sha256: str, size: int) -> bool:
    return (
        path.is_file() and path.stat().st_size == size and file_sha256(path) == sha256
    )


def _download(url: str, target: Path, sha256: str, size: int):
    digest, count = hashlib.sha256(), 0
    with (
        atomic_file(target, overwrite=True) as stream,
        urllib.request.urlopen(url, timeout=60) as response,
    ):
        while block := response.read(1024 * 1024):
            count += len(block)
            if count > size:
                raise AssetIntegrityError(f"{target.name} exceeds its pinned size")
            digest.update(block)
            stream.write(block)
        if count != size or digest.hexdigest() != sha256:
            raise AssetIntegrityError(
                f"Downloaded {target.name} does not match its pinned SHA-256"
            )


def _install(name, target: Path, sha256, size, url, supplied, offline):
    """Reuse, import or download one pinned file. Returns how it was obtained."""
    if _matches(target, sha256, size):
        return "present"
    for candidate in supplied:
        if _matches(candidate, sha256, size):
            with (
                atomic_file(target, overwrite=True) as stream,
                candidate.open("rb") as source,
            ):
                shutil.copyfileobj(source, stream)
            return "imported"
    if offline:
        raise MissingAssetError(
            f"{name} weights are not installed and no matching local file was supplied"
        )
    try:
        _download(url, target, sha256, size)
    except urllib.error.URLError as error:
        raise MissingAssetError(
            f"Could not download {name} weights ({error}). Check the network, "
            "or supply the file with `tennis3d setup --weights DIR`."
        ) from error
    return "downloaded"


def setup(root: Path, *, weights=(), offline=False) -> dict:
    """Install and verify the detector and lifter weights. Safe to repeat."""
    supplied = []
    for item in map(Path, weights):
        supplied += sorted(item.iterdir()) if item.is_dir() else [item]
    supplied = [path for path in supplied if path.is_file()]
    report = {"home": str(root), "weights": {}}
    for name, asset in manifest(detector_manifest(root)).assets.items():
        report["weights"][name] = _install(
            name, asset.path, asset.sha256, asset.bytes, asset.url, supplied, offline
        )
    lifter = lifter_manifest()
    report["weights"]["lifter"] = _install(
        "lifter",
        lifter_path(root),
        lifter["sha256"],
        lifter["bytes"],
        lifter["url"],
        supplied,
        offline,
    )
    (root / "runs").mkdir(parents=True, exist_ok=True)
    return report


def problems(root: Path) -> list[str]:
    """User-facing reasons reconstruction cannot start; empty when ready."""
    found = []
    for module, label in (("torch", "PyTorch"), ("av", "PyAV"), ("cv2", "OpenCV")):
        try:
            __import__(module)
        except ImportError:
            found.append(f"{label} is not installed; reinstall with the launcher.")
    try:
        for asset in manifest(detector_manifest(root)).assets.values():
            verify(Asset.model_validate(asset))
        lifter = lifter_manifest()
        if not _matches(lifter_path(root), lifter["sha256"], lifter["bytes"]):
            raise MissingAssetError("lifter")
    except (MissingAssetError, AssetIntegrityError):
        found.append(f"Model weights are missing or damaged; {SETUP_HINT}.")
    return found


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False
