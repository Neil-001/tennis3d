"""Optional simulate command; parsing does not import the physics runtime."""

import argparse
import json
from pathlib import Path

import yaml

from tennis3d.config import load_config
from tennis3d.synthetic.config import SimulationConfig


def register_parser(subparsers):
    parser = subparsers.add_parser(
        "simulate", help="generate a small seeded numeric tennis dataset"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pool-cache", type=Path, help="reuse checksummed split-owned shot pools"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--format", choices=("npz", "parquet"), default="npz")
    parser.add_argument("--execution-config", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--phase",
        choices=("complete", "generate", "finalize"),
        default="complete",
        help="parquet execution phase; finalize requires complete generation and --resume",
    )
    parser.add_argument(
        "--restore-bundle",
        type=Path,
        help="restore a verified bundle into a fresh output before --resume",
    )
    parser.add_argument(
        "--recovery-bundle",
        type=Path,
        help="save a bounded portable bundle after draining",
    )
    parser.set_defaults(handler=run_command)


def run_command(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(
            SimulationConfig,
            args.config,
            overrides={"seed": args.seed} if args.seed is not None else None,
        )
        if args.format == "parquet":
            from tennis3d.synthetic.producer import generate_main
            from tennis3d.synthetic.producer_config import ProducerConfig
            from tennis3d.synthetic.producer_parallel import Stopped

            execution = load_config(ProducerConfig, args.execution_config)
            if args.restore_bundle:
                if not args.resume:
                    raise ValueError("--restore-bundle requires --resume")
                from tennis3d.synthetic.producer_recovery import restore_bundle

                restore_bundle(
                    args.restore_bundle, args.output, execution.recovery_bytes
                )

            def save_bundle(primary=None):
                if not (
                    args.recovery_bundle
                    and (args.output / "control/plan.json").exists()
                ):
                    return
                from tennis3d.synthetic.producer_bundle import recovery_bundle

                try:
                    recovery_bundle(
                        args.output, args.recovery_bundle, execution.recovery_bytes
                    )
                except Exception as bundle_error:
                    if primary is None:
                        raise
                    print(f"Recovery bundle failed: {bundle_error}")

            try:
                result = generate_main(
                    args.output,
                    cfg,
                    execution,
                    pool_cache=args.pool_cache,
                    resume=args.resume,
                    phase=getattr(args, "phase", "complete"),
                )
            except Stopped as error:
                save_bundle(error)
                print("Simulation stopped with incomplete recoverable planned output")
                return 4
            except Exception as error:
                save_bundle(error)
                raise
            save_bundle()
            print(json.dumps(result, indent=2))
            return 0
        if (
            getattr(args, "phase", "complete") != "complete"
            or args.execution_config
            or args.resume
            or args.recovery_bundle
            or args.restore_bundle
        ):
            raise ValueError(
                "execution/resume/recovery options require --format parquet"
            )
        from tennis3d.synthetic.generate import generate

        manifest = generate(args.output, cfg, args.pool_cache)
    except ImportError as error:
        print(
            f"Synthetic runtime unavailable: {error}. Install with uv sync --extra synthetic."
        )
        return 3
    except (ValueError, OSError, RuntimeError, yaml.YAMLError) as error:
        print(f"Simulation failed: {error}")
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "splits": manifest["splits"],
                "generation_seconds": manifest["generation_seconds"],
                "rallies_per_cpu_hour": manifest["rallies_per_cpu_hour"],
                "rallies_per_wall_hour": manifest["rallies_per_wall_hour"],
            },
            indent=2,
        )
    )
    return 0
