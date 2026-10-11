"""Fingerprint the installed Python implementation, including uncommitted edits."""

import hashlib
import importlib.metadata
from pathlib import Path


def code_identity():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "python-source-sha256:" + digest.hexdigest()


DETECTION_RUNTIME = ("torch", "numpy", "av", "opencv-python-headless", "scikit-image")
# DetectConfig device name -> OpenVINO device. Other devices use PyTorch.
OPENVINO_DEVICES = {"intel-gpu": "GPU"}


def detection_runtime(device):
    """Packages whose versions identify a detection run on this device."""
    return DETECTION_RUNTIME + (("openvino",) if device in OPENVINO_DEVICES else ())


CALIBRATION_RUNTIME = ("numpy", "opencv-python-headless", "scipy", "pykalman")


def runtime_versions(names):
    """Record versions only when a stage executes, never on import."""
    return {name: importlib.metadata.version(name) for name in names}
