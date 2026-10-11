"""Bounded subprocess stages. The supervisor selects all inputs and deadlines."""

import argparse
import os
import time
from pathlib import Path

from tennis3d.artifacts import canonical_json, file_sha256, read_json, write_json
from tennis3d.autoresearch.inputs import hardware_identity, load_request, verify_data
from tennis3d.autoresearch.protocol import verify_sources

MIN_PARAMETERS = 1_500_000  # inclusive floor for candidates and the reference
MAX_PARAMETERS = 2_000_000  # exclusive ceiling for every trial


def check_parameter_count(total: int, trainable: int, baseline: str | None) -> None:
    """Ceiling on all parameters, floor on trainable ones; small is floor-exempt."""
    floor = 0 if baseline == "small" else MIN_PARAMETERS
    if not (floor <= trainable <= total < MAX_PARAMETERS):
        raise ValueError(
            f"{trainable} trainable of {total} parameters outside "
            f"[{floor}, {MAX_PARAMETERS})"
        )


def count_parameters(model) -> tuple[int, int]:
    params = list(model.parameters())
    return sum(p.numel() for p in params), sum(
        p.numel() for p in params if p.requires_grad
    )


def training(request: dict, output: Path):
    from tennis3d.autoresearch.observer import Observer
    from tennis3d.lifting.settings import TrainConfig

    config = TrainConfig.model_validate(request["training"])
    from tennis3d.lifting.model import Lifter

    # Fail an out-of-band initial model before spending the training budget.
    check_parameter_count(
        *count_parameters(Lifter(config.model)), request.get("baseline")
    )
    observer = Observer(config, None)
    observer.install()
    # Import only after installing the frozen data/checkpoint/optimizer observers.
    from tennis3d.lifting.train import TrainControl, train

    previous = -float("inf")

    def update(row):
        nonlocal previous
        observer.update(row)
        if row["phase"] == "training" and "loss" not in row:
            return
        now = time.perf_counter()
        if (
            observer.control.stop_requested
            or now - previous >= request["protocol"]["loss_interval_seconds"]
            or row["phase"] != "training"
        ):
            progress = {**row, "train_seconds": observer.seconds, "time": time.time()}
            write_json(output / "progress.json", progress, overwrite=True)
            if "loss" in row:
                with (output / "curve.jsonl").open("ab") as stream:
                    stream.write(canonical_json(progress) + b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            previous = now

    control = TrainControl(
        use_step_limit=False,
        use_early_stopping=False,
        validate_during_training=False,
        evaluate_at_end=False,
        max_train_seconds=config.max_train_seconds,
        deadline=request["deadline"],
        on_update=update,
    )
    observer.control = control
    try:
        summary = train(config, output / "training", resume=None, control=control)
        observed = observer.finish(summary)
        write_json(output / "observed.json", observed)
    finally:
        observer.close()


def checks(request: dict, output: Path):
    import numpy as np
    import torch

    from tennis3d.lifting.checkpoint import load_lifter
    from tennis3d.lifting.features import build_features
    from tennis3d.lifting.losses import position_loss

    summary = read_json(output / "training/summary.json")
    observed = read_json(output / "observed.json")
    model, state = load_lifter(output / "training/checkpoints/last.pt", "cpu")
    if state["config"] != request["training"] or state["step"] != observed["steps"]:
        raise ValueError("checkpoint config/step differs from supervised training")
    if int(state["ema"]["n_averaged"]) != observed["steps"] + 1:
        raise ValueError("final EMA does not include every optimizer update")
    if summary["last_checkpoint_sha256"] != file_sha256(
        output / "training/checkpoints/last.pt"
    ):
        raise ValueError("final checkpoint differs from trainer summary")
    limits = summary["execution_limits"]
    if (
        any(
            limits[key] is not None
            for key in ("max_steps", "patience", "validation_every")
        )
        or limits["evaluate_at_end"]
    ):
        raise ValueError(
            "candidate enabled step/plateau/validation selection during timed training"
        )
    parameters, trainable = count_parameters(model)
    check_parameter_count(parameters, trainable, request.get("baseline"))
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("nonfinite model")
    # Functional CPU compatibility on 250 real timestamps, independent of crop settings.
    n = 250
    features = build_features(
        t_s=np.arange(n, dtype=np.float64) / 50,
        ball_uv=np.full((n, 2), 0.5),
        ball_observed=np.arange(n) % 3 != 0,
        court_uv=np.ones((n, 14, 2)),
        camera_valid=np.ones(n, dtype=bool),
        image_size=np.array([640, 480]),
    )
    predicted = model(features)
    if predicted.shape != (1, n, 3) or not torch.isfinite(predicted).all():
        raise ValueError("model violates dense XYZ contract")
    loss = position_loss(
        predicted,
        torch.zeros_like(predicted),
        torch.ones((1, n), dtype=torch.bool),
        features.padding,
    )
    loss.backward()
    if not torch.isfinite(loss) or not all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    ):
        raise ValueError("CPU loss/backward is nonfinite")
    metrics = read_json(output / "metrics.json")
    expected = request["data"]["counts"]["val"]
    source_identity = {
        label: source["identity"]
        for label, source in request["data"]["sources"].items()
    }
    if (
        state.get("dataset_identity") != source_identity
        or metrics.get("source_identity") != source_identity["val"]
    ):
        raise ValueError("checkpoint/evaluation source differs from frozen data")
    for mode in ("corrupted", "clean"):
        value = metrics["modes"][mode]
        if (
            value["valid_coverage"] != 1
            or value["count"] != expected["supported_frames"]
            or value["view_count"] != expected["views"]
            or value["physical_rally_count"] != expected["physical_rallies"]
        ):
            raise ValueError("evaluation denominator changed")
        if not np.isfinite(value["val_xyz_m"]):
            raise ValueError("nonfinite comparison score")
    if (
        metrics["checkpoint_step"] != state["step"]
        or metrics["checkpoint_sha256"] != summary["last_checkpoint_sha256"]
    ):
        raise ValueError("score does not belong to the final checkpoint EMA")
    verify_data(request["data"])
    return {
        "passed": True,
        "parameter_count": parameters,
        "cpu_forward_backward": True,
        "fresh_initial_weights_sha256": observed["initial_weights_sha256"],
        "final_ema_updates": int(state["ema"]["n_averaged"]),
        "validation_exposure": expected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "resolve", "train", "evaluate", "checks")
    )
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request, protocol = load_request(args.request)
    output = args.request.parent
    try:
        import torch

        torch.set_num_threads(2)
        if args.stage == "prepare":
            from tennis3d.autoresearch.inputs import data_identity, resolve_training

            result = {
                "data": data_identity(protocol),
                "hardware": hardware_identity(protocol.device),
                "training": resolve_training(Path(request["root"]), protocol),
            }
            write_json(output / "prepared.json", result)
            return
        verify_sources(
            Path(request["root"]), request["frozen_files"], request["candidate_files"]
        )
        if hardware_identity(protocol.device) != request["hardware"]:
            raise ValueError("hardware/software changed; initialize a new segment")
        verify_data(request["data"], protocol)
        if args.stage == "resolve":
            from tennis3d.autoresearch.inputs import resolve_training

            write_json(
                output / "config.resolved.json",
                resolve_training(Path(request["root"]), protocol, request["training"]),
            )
        elif args.stage == "train":
            training(request, output)
        elif args.stage == "evaluate":
            from tennis3d.evaluation.synthetic import evaluate_checkpoint

            result = evaluate_checkpoint(
                output / "training/checkpoints/last.pt",
                protocol.val_manifest,
                device=protocol.device,
                modes=("corrupted", "clean"),
            )
            write_json(output / "metrics.json", result)
        else:
            write_json(output / "checks.json", checks(request, output))
        verify_sources(
            Path(request["root"]), request["frozen_files"], request["candidate_files"]
        )
    except Exception as exc:
        write_json(
            output / "failure.json",
            {"stage": args.stage, "type": type(exc).__name__, "reason": str(exc)},
            overwrite=True,
        )
        raise


if __name__ == "__main__":
    main()
