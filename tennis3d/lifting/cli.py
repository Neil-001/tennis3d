"""Lazy train/evaluate command registration; base CLI does not import PyTorch."""

import argparse
import json
import signal
from pathlib import Path

import yaml

from tennis3d.artifacts import write_json
from tennis3d.config import load_config


def register_parser(commands):
    train = commands.add_parser("train", help="train or resume the dense XYZ lifter")
    train.add_argument("--config", type=Path)
    train.add_argument("--output", type=Path)
    train.add_argument("--resume", type=Path)
    train.add_argument(
        "--fork",
        type=Path,
        help="frozen continuation plan: start a new run from a finished checkpoint",
    )
    train.add_argument(
        "--expect-plan",
        type=Path,
        help="with --resume: require a run forked from this continuation plan",
    )
    train.add_argument("--device", choices=("auto", "cpu", "cuda"))
    train.add_argument("--max-steps", type=int)
    train.add_argument("--max-train-seconds", type=float)
    train.add_argument("--checkpoint-every", type=int)
    train.add_argument(
        "--job-setup-seconds",
        type=float,
        default=0.0,
        help="charge actual allocation staging before further updates",
    )
    train.set_defaults(handler=train_command)
    evaluate = commands.add_parser(
        "evaluate", help="score complete synthetic views with EMA weights"
    )
    evaluate.add_argument("--config", type=Path)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--manifest", type=str)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"))
    evaluate.add_argument("--output", type=Path)
    evaluate.set_defaults(handler=evaluate_command)


def train_command(args: argparse.Namespace) -> int:
    try:
        from tennis3d.lifting.checkpoint import read_checkpoint
        from tennis3d.lifting.settings import TrainConfig
        from tennis3d.lifting.train import TrainControl, train

        overrides = {
            key: getattr(args, key)
            for key in ("device", "max_steps", "max_train_seconds", "checkpoint_every")
            if getattr(args, key, None) is not None
        }
        if args.resume and not args.config:
            overrides = {**read_checkpoint(args.resume)["config"], **overrides}
        config = load_config(TrainConfig, args.config, overrides=overrides)
        output = args.output or (
            args.resume.resolve().parent.parent if args.resume else None
        )
        if output is None:
            raise ValueError("train requires --output for a new run")
        fork = getattr(args, "fork", None)
        if fork and not args.config:
            raise ValueError("a fork requires its own frozen --config")
        control = TrainControl(
            setup_job_seconds=getattr(args, "job_setup_seconds", 0.0)
        )
        previous = {}

        def stop(signum, frame):
            control.stop_requested = True
            # A repeated signal can interrupt a slow step or evaluation.
            signal.signal(signum, previous[signum])

        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, stop)
            result = train(
                config,
                output,
                resume=args.resume,
                fork=fork,
                expected_plan=getattr(args, "expect_plan", None),
                control=control,
            )
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        print(json.dumps(result, indent=2))
        return 0
    except (ImportError, FileNotFoundError, RuntimeError) as exc:
        print(f"Training runtime unavailable: {exc}")
        return 3
    except (TypeError, ValueError, OSError, yaml.YAMLError) as exc:
        print(f"Invalid training input: {exc}")
        return 2


def evaluate_command(args: argparse.Namespace) -> int:
    try:
        from tennis3d.evaluation.synthetic import evaluate_checkpoint
        from tennis3d.lifting.settings import EvaluationConfig

        overrides = {
            key: getattr(args, key)
            for key in ("device", "manifest")
            if getattr(args, key) is not None
        }
        config = load_config(EvaluationConfig, args.config, overrides=overrides)
        if args.output and args.output.exists():
            raise ValueError("evaluation output already exists")
        result = evaluate_checkpoint(
            args.checkpoint, config.manifest, device=config.device, modes=config.modes
        )
        result["config"] = config.model_dump(mode="json")
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return 0
    except (ImportError, FileNotFoundError, RuntimeError) as exc:
        print(f"Evaluation runtime unavailable: {exc}")
        return 3
    except (TypeError, ValueError, OSError, yaml.YAMLError) as exc:
        print(f"Invalid evaluation input: {exc}")
        return 2
