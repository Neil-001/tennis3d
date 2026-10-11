"""Fixed data, runtime and resolved-training identities for a comparison."""

import importlib.metadata
import platform
import subprocess
from contextlib import ExitStack
from pathlib import Path

from tennis3d.artifacts import canonical_json, file_sha256, json_sha256, read_json
from tennis3d.autoresearch.protocol import Protocol
from tennis3d.config import load_config

# Config fields that may vary. Sampling, seeds, data and all budgets are supervisor-owned.
TUNABLE = {"model", "batch_size", "learning_rate", "ema_decay", "gradient_clip"}
DEFAULT_LOSS_LABEL = (
    "Squared Euclidean position loss, XYZ divided by 10 m (dimensionless)"
)
# Reviewed baseline implementation. A changed objective/trainer must name its loss.
DEFAULT_OBJECTIVE_FILES = {
    "tennis3d/lifting/losses.py": "82d4ab351ae28ad33021887d85c508ee086ce86cf7e2c3e833ed375ae0fae00d",
    "tennis3d/lifting/train.py": "8f22eaad83de14813f43be55961eb345e1c777a2b9373b5957211db2087eff54",
}


def data_identity(protocol: Protocol) -> dict:
    from tennis3d.evaluation.synthetic import preflight
    from tennis3d.lifting.data import ViewDataset, validate_split_pair

    with ExitStack() as opened:
        train = opened.enter_context(ViewDataset(protocol.train_manifest))
        val = opened.enter_context(ViewDataset(protocol.val_manifest))
        validate_split_pair(train, val)
        files, counts, sources = {}, {}, {}
        # Reject remote sources before any numeric read. Search inputs must be
        # materialized as local source descriptors before segment initialization.
        paths = {
            "train": train.local_files(),
            "val": val.local_files(),
        }
        preflight(val)
        for label, dataset in (("train", train), ("val", val)):
            dataset.simulation_config()
            frames = 0
            for index in range(len(dataset)):
                arrays = dataset[index]
                frames += int((arrays["label_valid"] & arrays["camera_valid"]).sum())
            dataset.verify_identity()
            files.update(
                {
                    str(path.resolve()): file_sha256(path)
                    for path in sorted(paths[label])
                }
            )
            counts[label] = {
                "physical_rallies": len({r["rally_id"] for r in dataset.records}),
                "views": len(dataset),
                "supported_frames": frames,
            }
            sources[label] = {
                "path": str(dataset.path.resolve()),
                "identity": dataset.identity,
            }
        result = {
            "schema_version": 2,
            "files": files,
            "counts": counts,
            "sources": sources,
        }
        return {**result, "sha256": json_sha256(result)}


def verify_data(data: dict, protocol: Protocol | None = None) -> None:
    """Verify only local bytes; never open a reader that could fetch remotely.

    Older file-only identities cannot bind exposure counts or membership and
    require a fresh comparison segment under this schema.
    """
    if data.get("schema_version") != 2 or data.get("sha256") != json_sha256(
        {key: value for key, value in data.items() if key != "sha256"}
    ):
        raise ValueError("frozen data identity changed; initialize a new segment")
    if protocol is not None:
        for label in ("train", "val"):
            path = str(getattr(protocol, f"{label}_manifest").resolve())
            if path != data["sources"][label]["path"] or path not in data["files"]:
                raise ValueError("protocol source differs from frozen local data")
    for path, expected in data["files"].items():
        if file_sha256(path) != expected:
            raise ValueError(f"frozen data changed: {path}")


def hardware_identity(device: str) -> dict:
    import torch

    from tennis3d.lifting.runtime import environment, resolve_device

    result = environment(resolve_device(device))
    result.update(
        machine=platform.machine(),
        platform=platform.platform(),
        cpu=platform.processor(),
        torch_build=torch.__config__.show(),
    )
    if Path("/proc/cpuinfo").exists():
        result["cpu"] = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            result["cpu"],
        )
    result["packages"] = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    }
    if device == "cuda":
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        result["driver"] = driver.stdout.strip().splitlines()
        props = torch.cuda.get_device_properties(0)
        result.update(
            gpu_memory=props.total_memory,
            capability=list(torch.cuda.get_device_capability(0)),
        )
    return result


def resolve_training(
    root: Path, protocol: Protocol, reference: dict | None = None
) -> dict:
    from tennis3d.lifting.settings import TrainConfig

    candidate = load_config(TrainConfig, root / "configs/lifter.yaml").model_dump(
        mode="json"
    )
    if reference is not None:
        # Reject silent changes to the frozen sampling recipe. Budget/data fields
        # are overwritten below regardless of candidate values.
        for key in ("min_frames", "max_frames", "event_crop_probability"):
            if candidate[key] != reference[key]:
                raise ValueError(f"frozen sampling setting changed: {key}")
        resolved = {**reference, **{key: candidate[key] for key in TUNABLE}}
    else:
        resolved = candidate
    resolved.update(
        train_manifest=str(protocol.train_manifest.resolve()),
        val_manifest=str(protocol.val_manifest.resolve()),
        seed=protocol.seed,
        device=protocol.device,
        max_train_seconds=protocol.train_seconds,
        max_job_seconds=protocol.total_seconds,
        patience=None,
        log_every=1,
        max_steps=1,
        validation_every=1,
        artifact_label=f"{protocol.profile} autoresearch; fresh weights; accuracy not established",
    )
    return TrainConfig.model_validate(resolved).model_dump(mode="json")


def load_request(path: Path):
    value = read_json(path)
    return value, Protocol.model_validate_json(canonical_json(value["protocol"]))


def loss_label(root: Path, explicit: str | None) -> str:
    if explicit:
        if not explicit.strip():
            raise ValueError("loss label must name the objective and units")
        return explicit
    if any(
        file_sha256(root / path) != digest
        for path, digest in DEFAULT_OBJECTIVE_FILES.items()
    ):
        raise ValueError(
            "changed loss/trainer requires --loss-label naming its objective and units"
        )
    return DEFAULT_LOSS_LABEL
