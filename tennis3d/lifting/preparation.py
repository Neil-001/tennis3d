"""Pinned Stage 4 data preparation and bounded profiling, never full training."""

import argparse
import importlib
import math
import os
import resource
import shutil
import socket
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import yaml

from tennis3d.artifacts import file_sha256, json_sha256, read_json, write_json
from tennis3d.datasets.schema import Artifact
from tennis3d.datasets.source import SourceSpec, V2Source, VerifiedCache
from tennis3d.datasets.subset import read_descriptor

MAIN_COMMIT = "c2e782053694cfde305362e670ebbb4f4e70884c"
COMPACT_COMMIT = "a5420e23f6bc3a5391254de78fead7115420b9aa"
CACHE_BYTES = 4 << 30
TARGET_WINDOWS = 4_000_000  # Predeclared ten full view passes, one crop per view.
WALL_SECONDS = 172800


def publish_exact(path: Path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"existing preparation artifact differs: {path}")
    else:
        write_json(path, value)


def prepare(pins_path: Path, output: Path, cache_base: Path):
    pins = read_json(pins_path)
    if (
        pins["status"] != "verified-release"
        or pins["main"]["commit"] != MAIN_COMMIT
        or pins["compact"]["publication_commit"] != COMPACT_COMMIT
    ):
        raise ValueError("preparation requires the declared immutable release")
    output.mkdir(parents=True, exist_ok=True)
    main = pins["main"]
    common: dict[str, Any] = {
        "manifest": Artifact.model_validate(main["manifest_artifact"]),
        "repo_id": pins["repo_id"],
        "revision": MAIN_COMMIT,
        "cache_bytes": CACHE_BYTES,
        "max_group_bytes": main["max_group_bytes"],
    }
    compact_artifact = Artifact.model_validate(pins["compact"]["descriptor_artifact"])
    spec = SourceSpec.model_validate(
        {
            **common,
            "manifest": compact_artifact,
            "revision": COMPACT_COMMIT,
            "split": "val",
            "cache_dir": str(cache_base / "metadata"),
        }
    )
    cache = VerifiedCache(cache_base / "metadata", spec)
    try:
        compact_path = cache.get(compact_artifact)
        compact = read_descriptor(
            compact_path, expected_sha256=pins["compact"]["descriptor_sha256"]
        )
    finally:
        cache.close()
    if (
        compact.main_commit != MAIN_COMMIT
        or compact.main_manifest != common["manifest"]
        or compact.main_repo_id != pins["repo_id"]
        or json_sha256(compact.model_dump()) != pins["compact"]["descriptor_sha256"]
    ):
        raise ValueError("compact descriptor has a different main source identity")
    exclusions = sorted(
        row.record.rally_id
        for row in compact.subset.selection.records
        if row.record.split == "val"
    )
    if len(exclusions) != 256 or len(set(exclusions)) != 256:
        raise ValueError("compact validation must contain 256 physical rallies")
    receipts = {}
    for split, name, expected in (
        ("train", "train", (200000, 400000)),
        ("val", "reserved-val", (4744, 9488)),
    ):
        descriptor = output / f"{name}.source.json"
        source_spec = SourceSpec(
            **common,
            split=split,
            cache_dir=str(cache_base / name),
            exclude_rally_ids=exclusions if split == "val" else [],
        )
        publish_exact(descriptor, source_spec.model_dump(mode="json"))
        source = V2Source(descriptor, read_json(descriptor))
        try:
            ids = {row["rally_id"] for row in source.records}
            if (len(ids), len(source.records)) != expected:
                raise ValueError("full source counts differ from release contract")
            if split == "val" and ids.intersection(exclusions):
                raise ValueError("reserved validation overlaps compact search")
            receipts[name] = {
                "descriptor": str(descriptor),
                "descriptor_sha256": file_sha256(descriptor),
                "identity": source.identity,
                "rallies": len(ids),
                "views": len(source.records),
                "rally_ids_sha256": json_sha256(sorted(ids)),
            }
        finally:
            source.close()
    receipt = {
        "pins_sha256": file_sha256(pins_path),
        "compact_descriptor_sha256": file_sha256(compact_path),
        "compact_revision": COMPACT_COMMIT,
        "excluded_rally_ids": exclusions,
        "excluded_rally_ids_sha256": json_sha256(exclusions),
        "sources": receipts,
        "access": "bounded pinned streaming; no test arrays read",
        "full_training_authorized": False,
    }
    publish_exact(output / "sources.receipt.json", receipt)
    return receipt


def exact_state(first, second):
    """Compare continuation state, including tensor bytes and random streams."""
    import torch

    if isinstance(first, torch.Tensor):
        if not isinstance(second, torch.Tensor) or not torch.equal(first, second):
            raise ValueError("GPU continuation tensor differs")
    elif isinstance(first, dict):
        if first.keys() != second.keys():
            raise ValueError("GPU continuation mapping differs")
        for key in first:
            exact_state(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        if type(first) is not type(second) or len(first) != len(second):
            raise ValueError("GPU continuation sequence differs")
        for a, b in zip(first, second):
            exact_state(a, b)
    elif first != second:
        raise ValueError("GPU continuation value differs")


def profile(recipe: Path, sources: Path, output: Path):
    """Fresh full-trainer qualification: 50 warm-up +300 measured updates.

    Another 50 updates replay checkpoint50 through100, comparing the complete
    state to uninterrupted100. Reserved validation is timed without scoring.
    The selected 83334-update schedule is never compressed to the profile cap.
    """
    import torch

    from tennis3d.evaluation.synthetic import input_mode, preflight
    from tennis3d.inference.predict import predict_view
    from tennis3d.inference.windows import make_windows
    from tennis3d.lifting.backup import account_final, publish, verify_snapshot
    from tennis3d.lifting.checkpoint import read_checkpoint
    from tennis3d.lifting.data import ViewDataset, WindowSampler
    from tennis3d.lifting.model import Lifter
    from tennis3d.lifting.runtime import environment, synchronize
    from tennis3d.lifting.settings import TrainConfig
    from tennis3d.lifting.train import TrainControl
    from tennis3d.provenance import code_identity

    trainer = importlib.import_module("tennis3d.lifting.train")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("profile output must be empty; never reuse profile weights")
    raw = yaml.safe_load(recipe.read_text())
    raw["model"]["schedule_steps"] = math.ceil(TARGET_WINDOWS / 48)
    raw.update(
        train_manifest=str(sources / "train.source.json"),
        val_manifest=str(sources / "reserved-val.source.json"),
        max_steps=83334,
        max_train_seconds=2400.0,
        max_job_seconds=3600.0,
        device="cuda",
        seed=42,
        checkpoint_every=50,
        log_every=10,
        artifact_label="final fresh qualification; no quality ranking",
    )
    config = TrainConfig.model_validate(raw)
    if (config.batch_size, config.max_frames) != (48, 250):
        raise ValueError("profile requires selected batch48/context250")
    write_json(output / "profile.config.json", config.model_dump(mode="json"))
    started = time.perf_counter()
    timings: dict[str, Any] = {
        "loader_batches": [],
        "updates": [],
        "checkpoints": [],
        "trainer_preflights": [],
        "invocations": [],
        "backup_publications": [],
    }
    with ViewDataset(config.train_manifest) as dataset:
        sampler = WindowSampler(dataset, config)
        for i in range(100):
            begin = time.perf_counter()
            sampler.next_batch()
            timings["loader_batches"].append(
                {"batch": i + 1, "seconds": time.perf_counter() - begin}
            )
        train_identity = dataset.identity
        train_cache_peak = dataset._source.cache.peak_bytes
    loader_finished = time.perf_counter()
    torch.manual_seed(42)
    model = Lifter(config.model).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    with ViewDataset(config.val_manifest) as dataset:
        val_identity = dataset.identity
        begin = time.perf_counter()
        preflight(dataset)
        timings["validation_preflight_seconds"] = time.perf_counter() - begin
        begin = time.perf_counter()
        windows = 0
        for i in range(len(dataset)):
            arrays = dataset[i]
            for mode in ("corrupted", "clean"):
                predict_view(
                    model,
                    input_mode(arrays, mode),
                    device="cuda",
                    require_complete=True,
                )
                windows += len(make_windows(arrays["camera_valid"]))
        synchronize("cuda")
        timings["validation_forward_seconds"] = time.perf_counter() - begin
        timings["validation_views"] = len(dataset)
        timings["validation_windows"] = windows
        val_cache_peak = dataset._source.cache.peak_bytes
    del model
    validation_finished = time.perf_counter()
    validation_peak_allocated = torch.cuda.max_memory_allocated()
    validation_peak_reserved = torch.cuda.max_memory_reserved()
    original_batch, original_save, original_preflight = (
        WindowSampler.next_batch,
        trainer.save_checkpoint,
        trainer.preflight,
    )
    invocation = "uninterrupted"
    batch_times = []
    run_dir, replay = output / "fresh-run", output / "replay50"

    def measured_batch(self):
        begin = time.perf_counter()
        result = original_batch(self)
        batch_times.append(time.perf_counter() - begin)
        if tuple(result[1].shape) != (48, 250, 3):
            raise ValueError("profile batch differs from context250/batch48")
        return result

    def measured_preflight(dataset):
        begin = time.perf_counter()
        result = original_preflight(dataset)
        timings["trainer_preflights"].append(
            {"invocation": invocation, "seconds": time.perf_counter() - begin}
        )
        return result

    def measured_save(path, state):
        begin = time.perf_counter()
        original_save(path, state)
        timings["checkpoints"].append(
            {
                "invocation": invocation,
                "step": state["step"],
                "seconds": time.perf_counter() - begin,
                "bytes": Path(path).stat().st_size,
            }
        )
        if invocation == "uninterrupted" and state["step"] == 50:
            replay.mkdir()
            (replay / "checkpoints").mkdir()
            for name in (
                "run.json",
                "config.resolved.json",
                "loss.jsonl",
                "budget.json",
            ):
                shutil.copyfile(run_dir / name, replay / name)
            shutil.copyfile(path, replay / "checkpoints/last.pt")
        if invocation == "uninterrupted" and state["step"] == 100:
            shutil.copyfile(path, output / "uninterrupted100.pt")

    def capture(run):
        begin = time.monotonic()
        store = output / f"{run.name}.backups"
        checkpoint_hash = file_sha256(run / "checkpoints/last.pt")
        snapshot_id = publish(run, store, started=begin)
        snapshot = store / "snapshots" / snapshot_id
        manifest = verify_snapshot(snapshot, snapshot_id)
        account_final(run, store, event=f"profile-{run.name}", started=begin)
        if file_sha256(snapshot / "checkpoints/last.pt") != checkpoint_hash:
            raise ValueError("backup changed checkpoint bytes")
        timings["backup_publications"].append(
            {
                "step": manifest["step"],
                "seconds": time.monotonic() - begin,
                "capture_job_seconds": manifest["capture_job_seconds"],
                "wrapper_job_seconds": manifest["wrapper_job_seconds"],
                "snapshot_id": snapshot_id,
                "checkpoint_sha256": checkpoint_hash,
                "ledger_job_seconds": read_json(run / "budget.json")["job_seconds"],
            }
        )
        return snapshot

    def run_training(directory, stop, resume=None):
        control = TrainControl(validate_during_training=False, evaluate_at_end=False)

        def record(row):
            if row["phase"] == "training" and "step_seconds" in row:
                timings["updates"].append(
                    {**row, "invocation": invocation, "loader_seconds": batch_times[-1]}
                )
                if row["step"] == stop:
                    control.stop_requested = True

        control.on_update = record
        begin = time.perf_counter()
        result = trainer.train(config, directory, resume=resume, control=control)
        timings["invocations"].append(
            {
                "invocation": invocation,
                "wall_seconds": time.perf_counter() - begin,
                "summary": result,
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
        if result["step"] != stop or result["stop_reason"] != "requested":
            raise ValueError("profile did not complete its bounded graceful stop")
        return result

    with (
        patch.object(WindowSampler, "next_batch", measured_batch),
        patch.object(trainer, "save_checkpoint", measured_save),
        patch.object(trainer, "preflight", measured_preflight),
    ):
        final = run_training(run_dir, 350)
        capture(run_dir)
        checkpoint50 = capture(replay)
        invocation = "resumed50to100"
        resumed = run_training(replay, 100, checkpoint50 / "checkpoints/last.pt")
        reference = read_checkpoint(output / "uninterrupted100.pt")
        continued = read_checkpoint(resumed["last_checkpoint"])
        fields = (
            "model",
            "ema",
            "optimizer",
            "sampler",
            "rng",
            "training_objective",
            "step",
            "dataset_identity",
        )
        for field in fields:
            exact_state(reference[field], continued[field])
        charged = read_json(checkpoint50 / "budget.json")["job_seconds"]
        if (
            resumed["job_seconds"] < charged
            or resumed["train_seconds"]
            <= read_checkpoint(checkpoint50 / "checkpoints/last.pt")["budget"][
                "train_seconds"
            ]
        ):
            raise ValueError("continuation lost accumulated time or backup charge")
    receipt = {
        **code_identity(),
        "environment": environment("cuda"),
        "host": socket.gethostname(),
        "host_load": list(os.getloadavg()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "live_output": str(output),
        "recipe_sha256": file_sha256(recipe),
        "profile_config_sha256": file_sha256(output / "profile.config.json"),
        "sources_receipt_sha256": file_sha256(sources / "sources.receipt.json"),
        "train_identity": train_identity,
        "val_identity": val_identity,
        "seed": 42,
        "target_windows": TARGET_WINDOWS,
        "wall_cap_seconds": WALL_SECONDS,
        "warmup_updates": 50,
        "measured_updates": 300,
        "additional_replay_updates": 50,
        "profile_schedule_steps": config.model.schedule_steps,
        "loader_wall_seconds": loader_finished - started,
        "validation_wall_seconds": validation_finished - loader_finished,
        "training_wall_seconds": time.perf_counter() - validation_finished,
        "total_wall_seconds": time.perf_counter() - started,
        "initial_stop": final["stop_reason"],
        "interrupted_step": final["step"],
        "final_step": final["step"],
        "train_seconds": final["train_seconds"],
        "exact_continuation": {
            "from_step": 50,
            "to_step": 100,
            "fields": list(fields),
            "equal": True,
            "backup_charge_carried": True,
        },
        "gpu_peak_allocated_bytes": max(
            r["gpu_peak_allocated_bytes"] for r in timings["invocations"]
        ),
        "gpu_peak_reserved_bytes": max(
            r["gpu_peak_reserved_bytes"] for r in timings["invocations"]
        ),
        "validation_peak_allocated_bytes": validation_peak_allocated,
        "validation_peak_reserved_bytes": validation_peak_reserved,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cache_peak_bytes": {"train": train_cache_peak, "val": val_cache_peak},
        "timings": timings,
        "caveats": [
            "One fixed fresh profile; reserved validation has no scoring or ranking; synthetic test arrays untouched.",
            "Validation measures full reader/preflight and both forward modes; metric aggregation excluded and separately allowed for.",
            "Actual GPU TrainControl graceful stop and exact checkpoint continuation; real Slurm preemption and24h expiry remain untested.",
            "Actual local backup capture and charge continuation; existing private Hub fixture qualification reused without new uploads.",
            "One node/allocation; cold100-batch read and400 actual updates do not establish48h network or load reliability.",
        ],
        "full_training_authorized": False,
    }
    write_json(output / "profile.json", receipt)
    return receipt


def resolve(recipe: Path, sources: Path, profile_path: Path, output: Path):
    from tennis3d.lifting.settings import TrainConfig

    measurement = read_json(profile_path)
    raw = yaml.safe_load(recipe.read_text())
    if (raw["batch_size"], raw["learning_rate"], raw["ema_decay"]) != (
        48,
        0.0016,
        0.99,
    ):
        raise ValueError("planning requires the fixed selected batch48 recipe")
    if measurement["recipe_sha256"] != file_sha256(recipe):
        raise ValueError("profile recipe differs from the selected recipe")
    receipt_path = sources / "sources.receipt.json"
    if measurement["sources_receipt_sha256"] != file_sha256(receipt_path):
        raise ValueError("profile was measured on different prepared sources")
    receipt = read_json(receipt_path)
    for name, key in (("train", "train_identity"), ("reserved-val", "val_identity")):
        source = receipt["sources"][name]
        if source["identity"] != measurement[key] or source[
            "descriptor_sha256"
        ] != file_sha256(sources / f"{name}.source.json"):
            raise ValueError(
                "prepared source descriptor or identity differs from profile"
            )
    if measurement["target_windows"] != TARGET_WINDOWS or measurement["seed"] != 42:
        raise ValueError("profile differs from predeclared exposure and seed")
    rows = [
        r
        for r in measurement["timings"]["updates"]
        if r["invocation"] == "uninterrupted" and 50 < r["step"] <= 350
    ]
    if (
        len(rows) != 300
        or [
            r["step"]
            for r in measurement["timings"]["updates"]
            if r["invocation"] == "uninterrupted"
        ]
        != list(range(1, 351))
        or [r["step"] for r in rows] != list(range(51, 351))
        or measurement["warmup_updates"] < 50
        or measurement["measured_updates"] < 300
        or measurement["final_step"] != 350
        or measurement["profile_schedule_steps"] != 83334
        or not measurement["exact_continuation"]["equal"]
        or measurement["total_wall_seconds"] > 3600
    ):
        raise ValueError("missing or unauthorized final preparation measurement")
    loader = [row["seconds"] for row in measurement["timings"]["loader_batches"]]
    # Include cold transitions even when the measured trainer cache is warm.
    step_seconds = 1.25 * (
        float(np.percentile([r["step_seconds"] for r in rows], 95))
        + float(np.mean(loader))
    )
    validation_seconds = 1.5 * (
        measurement["timings"]["validation_preflight_seconds"]
        + measurement["timings"]["validation_forward_seconds"]
    )
    validation_every = max(
        1, math.ceil(max(3600, 10 * validation_seconds) / step_seconds)
    )
    checkpoint_every = max(1, math.floor(300 / step_seconds))
    checkpoint_seconds = 1.5 * max(
        r["seconds"] for r in measurement["timings"]["checkpoints"]
    )
    backup_seconds = 1.5 * max(
        r["seconds"] for r in measurement["timings"]["backup_publications"]
    )
    # Two allocations can cover the48h cap. Each setup includes full preflight,
    # source/model/optimizer initialization and wrapper staging. Queue is separate.
    setup_seconds = max(
        3600.0,
        2
        * max(
            measurement["validation_wall_seconds"],
            *(r["seconds"] for r in measurement["timings"]["trainer_preflights"]),
        ),
    )
    allocations = 2
    startup_allowance = allocations * setup_seconds
    backup_events = math.ceil(WALL_SECONDS / 3600) + allocations
    backup_allowance = backup_events * backup_seconds
    training_cap = 0.8 * WALL_SECONDS
    target_steps = math.ceil(TARGET_WINDOWS / 48)

    def cost(steps):
        return (
            startup_allowance
            + steps * step_seconds
            + (steps // validation_every + 1) * validation_seconds
            + (steps // checkpoint_every + 2 * allocations) * checkpoint_seconds
            + backup_allowance
        )

    planned = min(target_steps, math.floor(training_cap / step_seconds))
    while planned > 0 and cost(planned) > WALL_SECONDS:
        planned -= 1
    if planned < 1:
        raise ValueError("measured overheads leave no full-run update budget")
    available = min(
        training_cap, WALL_SECONDS - (cost(planned) - planned * step_seconds)
    )
    raw["model"]["schedule_steps"] = planned
    raw.update(
        train_manifest=str(sources / "train.source.json"),
        val_manifest=str(sources / "reserved-val.source.json"),
        max_steps=planned,
        max_train_seconds=available,
        max_job_seconds=float(WALL_SECONDS),
        validation_every=validation_every,
        checkpoint_every=checkpoint_every,
        patience=8,
        log_every=10,
        seed=42,
        device="cuda",
        artifact_label="Stage4 full training preparation; full training requires authorization",
    )
    config = TrainConfig.model_validate(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
    if output.exists() and output.read_text() != payload:
        raise ValueError("existing full-run config differs from resolved preparation")
    if not output.exists():
        output.write_text(payload)
    planning = {
        "profile_sha256": file_sha256(profile_path),
        "config_sha256": file_sha256(output),
        "target_windows": TARGET_WINDOWS,
        "target_steps": target_steps,
        "planned_steps": planned,
        "planned_windows": planned * 48,
        "planned_view_epochs": planned * 48 / 400000,
        "exposure_limited_by_wall": planned < target_steps,
        "step_allowance_seconds": step_seconds,
        "validation_allowance_seconds": validation_seconds,
        "startup_allowance_seconds": startup_allowance,
        "checkpoint_allowance_seconds_per_event": checkpoint_seconds,
        "checkpoint_events": planned // checkpoint_every + 2 * allocations,
        "backup_allowance_seconds_per_event": backup_seconds,
        "backup_events": backup_events,
        "backup_allowance_seconds": backup_allowance,
        "allocation_setup_allowance_seconds": setup_seconds,
        "planned_allocations": allocations,
        "training_workload_fraction_limit": 0.8,
        "training_workload_cap_seconds": training_cap,
        "estimated_training_workload_seconds": planned * step_seconds,
        "validation_events": planned // validation_every + 1,
        "validation_metric_allowance": "50% over full preflight+two-mode forwards; no quality ranking performed",
        "queue_accounting": "excluded; setup/loading/resume/checkpoint/backup work charged to accumulated job budget",
        "estimated_wall_seconds": cost(planned),
        "wall_cap_seconds": WALL_SECONDS,
        "full_training_authorized": False,
    }
    publish_exact(output.with_suffix(".plan.json"), planning)
    return planning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sources")
    p.add_argument(
        "--pins", type=Path, default=Path("configs/release-pins-main-v1.json")
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    for name in ("profile", "resolve"):
        p = sub.add_parser(name)
        p.add_argument("--recipe", type=Path, default=Path("configs/lifter.yaml"))
        p.add_argument("--sources", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if name == "resolve":
            p.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "sources":
        result = prepare(args.pins, args.output.resolve(), args.cache.resolve())
    elif args.command == "profile":
        result = profile(args.recipe, args.sources.resolve(), args.output.resolve())
    else:
        result = resolve(args.recipe, args.sources.resolve(), args.profile, args.output)
    print(result)


if __name__ == "__main__":
    main()
