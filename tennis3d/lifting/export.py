"""Compact EMA-only inference export in the current checkpoint schema.

The export keeps exactly what `load_lifter` reads: schema version, world scale,
the resolved model config and the EMA module tensors. Optimizer, sampler, RNG,
budget and training paths stay in the source checkpoint. Identity lives in an
external JSON manifest, verified before the weights are loaded.
"""

import hashlib
import io
from pathlib import Path

import numpy as np
import torch

from tennis3d.artifacts import (
    atomic_file,
    file_sha256,
    json_sha256,
    read_json,
    write_json,
)
from tennis3d.lifting.checkpoint import (
    CHECKPOINT_VERSION,
    load_lifter,
    read_checkpoint,
)
from tennis3d.lifting.features import WORLD_SCALE_M, build_features
from tennis3d.lifting.model import ModelConfig

MANIFEST_VERSION = "1.0"


def compact_state(state: dict) -> dict:
    """Select the inference fields of a full training checkpoint."""
    model = ModelConfig.model_validate(state["config"]["model"]).model_dump(mode="json")
    ema = {
        key: value.detach().cpu().clone()
        for key, value in state["ema"].items()
        if key.startswith("module.")
    }
    if not ema:
        raise ValueError("checkpoint has no EMA module weights")
    return {
        "schema_version": CHECKPOINT_VERSION,
        "world_scale_m": WORLD_SCALE_M,
        "config": {"model": model},
        "ema": ema,
    }


def tensor_sha256(ema: dict) -> str:
    """Hash tensor names, dtypes, shapes and values independently of file bytes."""
    digest = hashlib.sha256()
    for key in sorted(ema):
        array = ema[key].detach().cpu().contiguous().numpy()
        digest.update(f"{key}\0{array.dtype}\0{list(array.shape)}\0".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def export_lifter(source: Path, output: Path, *, overwrite=False) -> dict:
    """Write the compact export and its manifest; return the manifest."""
    source, output = Path(source), Path(output)
    if output.resolve() == source.resolve():
        raise ValueError("export output must differ from the source checkpoint")
    source_hash = file_sha256(source)
    state = read_checkpoint(source)
    compact = compact_state(state)
    buffer = io.BytesIO()
    torch.save(compact, buffer)
    with atomic_file(output, overwrite=overwrite) as stream:
        stream.write(buffer.getvalue())
    if file_sha256(source) != source_hash:
        output.unlink(missing_ok=True)
        raise ValueError("Source checkpoint changed during export")
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "file": output.name,
        "sha256": file_sha256(output),
        "bytes": output.stat().st_size,
        "checkpoint_schema_version": CHECKPOINT_VERSION,
        "world_scale_m": WORLD_SCALE_M,
        "weights": "EMA",
        "architecture_version": compact["config"]["model"]["architecture_version"],
        "model_config_sha256": json_sha256(compact["config"]["model"]),
        "tensor_sha256": tensor_sha256(compact["ema"]),
        "tensor_count": len(compact["ema"]),
        "tensor_elements": sum(value.numel() for value in compact["ema"].values()),
        "source_checkpoint_sha256": source_hash,
        "source_step": state.get("step"),
    }
    write_json(manifest_path(output), manifest, overwrite=overwrite)
    verify_export(output)
    return manifest


def manifest_path(export: Path) -> Path:
    return Path(export).with_suffix(".manifest.json")


def verify_export(export: Path, manifest: dict | None = None) -> dict:
    """Check file, config and tensor identity before trusting an export."""
    export = Path(export)
    manifest = manifest or read_json(manifest_path(export))
    if manifest.get("schema_version") != MANIFEST_VERSION:
        raise ValueError("unsupported lifter export manifest version")
    if (
        export.stat().st_size != manifest["bytes"]
        or file_sha256(export) != manifest["sha256"]
    ):
        raise ValueError("lifter export does not match its manifest SHA-256")
    state = read_checkpoint(export)
    if set(state) != {"schema_version", "world_scale_m", "config", "ema"} or set(
        state["config"]
    ) != {"model"}:
        raise ValueError("lifter export contains fields outside the inference schema")
    if (
        json_sha256(state["config"]["model"]) != manifest["model_config_sha256"]
        or tensor_sha256(state["ema"]) != manifest["tensor_sha256"]
    ):
        raise ValueError("lifter export weights do not match its manifest")
    return manifest


def qualify_export(source: Path, export: Path, *, seed=0, frames=250) -> dict:
    """Compare the export with its source: tensors and predictions must be equal."""
    manifest = verify_export(export)
    if file_sha256(source) != manifest["source_checkpoint_sha256"]:
        raise ValueError("source checkpoint differs from the export manifest")
    original, state = load_lifter(source)
    compact, _ = load_lifter(export)
    first, second = original.state_dict(), compact.state_dict()
    if first.keys() != second.keys() or not all(
        torch.equal(first[key], second[key]) for key in first
    ):
        raise ValueError("exported tensors differ from the source EMA weights")
    rng = np.random.default_rng(seed)
    size = np.array([1920.0, 1080.0])
    observed = rng.random(frames) > 0.2
    observed[0] = True
    features = build_features(
        t_s=np.cumsum(rng.uniform(0.015, 0.045, frames)),
        ball_uv=rng.uniform(0, 1, (frames, 2)) * size,
        ball_observed=observed,
        court_uv=rng.uniform(0, 1, (frames, 14, 2)) * size,
        camera_valid=np.ones(frames, dtype=bool),
        image_size=size,
    )
    with torch.inference_mode():
        expected, actual = original(features), compact(features)
    if not torch.equal(expected, actual):
        raise ValueError("exported model predictions differ from the source model")
    return {
        "status": "equal",
        "export_sha256": manifest["sha256"],
        "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
        "tensor_sha256": manifest["tensor_sha256"],
        "tensors_compared": len(first),
        "prediction_shape": list(expected.shape),
        "prediction_max_abs_difference": float((expected - actual).abs().max()),
        "prediction_sha256": tensor_sha256({"xyz": expected}),
        "probe": {"seed": seed, "frames": frames, "device": "cpu"},
        "source_step": state.get("step"),
        "torch": str(torch.__version__),
    }
