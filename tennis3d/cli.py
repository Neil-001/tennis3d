"""Small CLI. Add ordinary parser registrations here as stages are implemented."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

from tennis3d import __version__

# Shared process exit codes for future stage handlers.
INVALID_INPUT = 2
UNAVAILABLE_RUNTIME = 3
NO_TRAJECTORY = 4


def doctor(args: argparse.Namespace) -> int:
    """Inspect the installed runtime without fetching assets or allocating a GPU."""
    report = {
        "tennis3d": __version__,
        "python": platform.python_version(),
        "device_requested": args.device,
        "device": "cpu",
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pydantic", "PyYAML")
        },
        "optional_modules": {
            name: importlib.util.find_spec(name) is not None
            for name in ("torch", "cv2", "av", "mujoco")
        },
        "assets": "Use --perception for detector asset probes or --checkpoint PATH for a lifter probe.",
        "cuda": "not checked",
    }
    code = 0
    if args.device == "cuda":
        report["cuda"] = "CUDA verification requires --perception or --checkpoint"
        code = UNAVAILABLE_RUNTIME
    if args.device == "intel-gpu" and not args.perception:
        report["accelerator"] = "Intel GPU verification requires --perception"
        code = UNAVAILABLE_RUNTIME
    if args.perception:
        try:
            from tennis3d.perception.doctor import check

            report.update(check(args.device, args.assets_manifest))
            code = 0
        except (ImportError, RuntimeError, OSError, ValueError, KeyError) as error:
            report["perception_error"] = str(error)
            code = UNAVAILABLE_RUNTIME
    if args.checkpoint:
        try:
            from tennis3d.inference.doctor import check_lifter
            from tennis3d.inference.settings import lifting_device

            report["lifting"] = check_lifter(
                args.checkpoint, lifting_device(args.device)
            )
            if args.device != "intel-gpu":
                report["device"] = report["lifting"]["runtime"]["device"]
            report["cuda"] = (
                "verified by lifter probe"
                if report["device"] == "cuda"
                else "not requested"
            )
            if "perception_error" not in report:
                code = 0
        except (
            ImportError,
            RuntimeError,
            OSError,
            ValueError,
            KeyError,
            TypeError,
        ) as error:
            report["lifting_error"] = str(error)
            code = UNAVAILABLE_RUNTIME
    if args.device == "cuda" and code and (args.perception or args.checkpoint):
        report["cuda"] = "CUDA verification failed"
    if args.device == "intel-gpu" and not args.perception:
        # A CPU lifter probe says nothing about the graphics device.
        code = UNAVAILABLE_RUNTIME
    print(json.dumps(report, indent=2))
    return code


def version(args: argparse.Namespace) -> int:
    print(f"tennis3d {__version__}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tennis3d", description="Tennis 3D reconstruction tools"
    )
    parser.add_argument(
        "--version", action="version", version=f"tennis3d {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("version", help="show package version").set_defaults(
        handler=version
    )
    check = commands.add_parser(
        "doctor", help="inspect the local runtime without downloads"
    )
    check.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "intel-gpu"), default="auto"
    )
    check.add_argument(
        "--perception",
        action="store_true",
        help="load and probe perception checkpoints",
    )
    check.add_argument(
        "--assets-manifest", type=Path, default=Path("configs/assets.yaml")
    )
    check.add_argument(
        "--checkpoint", type=Path, help="load and probe an EMA lifter checkpoint"
    )
    check.set_defaults(handler=doctor)
    from tennis3d.app.cli import register_parser as register_app
    from tennis3d.inference.cli import register_parser as register_inference
    from tennis3d.lifting.cli import register_parser as register_lifting
    from tennis3d.perception.cli import register_parser as register_perception
    from tennis3d.synthetic.cli import register_parser as register_synthetic

    register_app(commands)
    if os.name == "posix":
        # Experiment tooling relies on POSIX file locks and signals.
        from tennis3d.autoresearch.state import register_parser as register_state
        from tennis3d.lifting.experiment import (
            register_parser as register_experiment,
        )

        register_state(commands)
        register_experiment(commands)
    register_inference(commands)
    register_lifting(commands)
    register_perception(commands)
    register_synthetic(commands)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code = args.handler(args)
    accelerate = sys.modules.get("tennis3d.perception.accelerate")
    if accelerate is not None and accelerate.STALLED:
        # Normal teardown would wait on the stalled device request forever.
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except OSError:
            pass
        finally:
            os._exit(code)
    return code
