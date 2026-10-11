"""Atomic, weights-only-loadable full training checkpoints and inference loading."""

import pickle
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from tennis3d.artifacts import atomic_file
from tennis3d.lifting.features import WORLD_SCALE_M
from tennis3d.lifting.model import ModelConfig, build_lifter

CHECKPOINT_VERSION = "1.0"


def random_state():
    numpy_state = np.random.get_state()
    assert isinstance(numpy_state, tuple)  # Legacy MT19937 state.
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_random_state(state, device):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"].cpu())
    if str(device).startswith("cuda") and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path, state):
    with atomic_file(path, overwrite=True) as stream:
        torch.save(state, stream)


def read_checkpoint(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError) as exc:
        raise ValueError("invalid or incompatible lifter checkpoint") from exc
    if not isinstance(state, dict):
        raise TypeError("lifter checkpoint must contain a state mapping")
    if (
        state.get("schema_version") != CHECKPOINT_VERSION
        or state.get("world_scale_m") != WORLD_SCALE_M
    ):
        raise ValueError("incompatible lifter checkpoint version or world scale")
    return state


def _same_dataset_identity(first, second):
    left, right = first.get("dataset_identity"), second.get("dataset_identity")
    if left is not None and right is not None:
        return left == right
    # A migrated v1 run may retain its original, unmodified best checkpoint.
    # The caller also checks the original manifest hashes in both checkpoints.
    return all(
        identity.get("format_version") == 1
        for identity in (left or right or {}).values()
    )


def carry_best_checkpoint(resume, state, destination):
    """Preserve the run's selected best, including a later improved best on recovery.

    A validation publishes best.pt before the next last.pt. A crash in between
    leaves a checkpoint that predates its run's recorded selection, possibly
    with no score at all; the published best is still the run's selection.
    """
    source = Path(resume).resolve().parent / "best.pt"
    if not source.is_file():
        if state["best_score"] is None:
            return None, None
        raise ValueError("resume requires the original run's selected best.pt")
    best = read_checkpoint(source)
    if (
        best["run_id"] != state["run_id"]
        or best["manifest_sha256"] != state["manifest_sha256"]
        or not _same_dataset_identity(best, state)
        or ModelConfig.model_validate(best["config"]["model"])
        != ModelConfig.model_validate(state["config"]["model"])
        or best.get("best_step", best["step"]) != best["step"]
        or best["best_score"] is None
        or not np.isfinite(best["best_score"])
        or (
            state["best_score"] is not None and best["best_score"] > state["best_score"]
        )
        or best.get("continuation") != state.get("continuation")
    ):
        raise ValueError("selected best checkpoint does not match the resumed run")
    if source != Path(destination).resolve():
        with (
            source.open("rb") as original,
            atomic_file(destination, overwrite=True) as target,
        ):
            shutil.copyfileobj(original, target)
    return best["best_score"], best["step"]


def load_lifter(path, device="cpu"):
    """Load final/selected EMA weights. Returns model and checkpoint metadata."""
    state = read_checkpoint(path)
    model = build_lifter(ModelConfig.model_validate(state["config"]["model"]))
    weights = {
        key.removeprefix("module."): value
        for key, value in state["ema"].items()
        if key.startswith("module.")
    }
    model.load_state_dict(weights, strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("checkpoint contains nonfinite model weights")
    return model.to(device).eval(), state
