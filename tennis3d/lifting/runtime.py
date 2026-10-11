"""Device selection and measured runtime identity, without perception imports."""

import platform

import numpy as np
import torch
import torch.version


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    return requested


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def environment(device):
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": device,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "torch_threads": torch.get_num_threads(),
    }
