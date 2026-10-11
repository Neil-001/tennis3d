"""Lightweight CLI parsing for reconstruction, raw lifting and filtered export."""

import json
from pathlib import Path

from tennis3d.config import load_config
from tennis3d.inference.quality import QualityConfig
from tennis3d.inference.settings import (
    LiftConfig,
    ReconstructionConfig,
    lifting_device,
)
from tennis3d.perception.assets import AssetIntegrityError, MissingAssetError


def run_command(args):
    try:
        if args.command == "reconstruct":
            from tennis3d.inference.pipeline import reconstruct

            overrides = {}
            if args.device is not None:
                overrides = {
                    "detection": {"device": args.device},
                    "lifting": {"device": lifting_device(args.device)},
                }
            if args.checkpoint is not None:
                overrides.setdefault("lifting", {})["checkpoint"] = str(args.checkpoint)
            if args.config is None and not Path("configs/assets.yaml").is_file():
                # Outside a source checkout, use the weights from `tennis3d setup`.
                from tennis3d.app.home import home, installed_config

                installed = installed_config(home(), args.device or "auto")
                installed["lifting"]["device"] = lifting_device(
                    installed["lifting"]["device"]
                )
                if args.checkpoint is not None:
                    installed["lifting"]["checkpoint"] = str(args.checkpoint)
                config = load_config(ReconstructionConfig, overrides=installed)
            else:
                config = load_config(
                    ReconstructionConfig, args.config, overrides=overrides
                )
            summary = reconstruct(
                args.video, args.output, config, overwrite=args.overwrite
            )
        elif args.command == "lift":
            from tennis3d.inference.lift import lift

            overrides = {
                name: str(getattr(args, name))
                for name in ("device", "checkpoint")
                if getattr(args, name) is not None
            }
            config = load_config(LiftConfig, args.config, overrides=overrides)
            summary = lift(args.run, config, overwrite=args.overwrite)
        else:
            from tennis3d.inference.export import export

            summary = export(
                args.run,
                load_config(QualityConfig, args.config),
                overwrite=args.overwrite,
            )
        print(json.dumps(summary, indent=2))
        return summary["exit_code"]
    except (ImportError, RuntimeError, MissingAssetError, AssetIntegrityError) as error:
        print(json.dumps({"status": "failed", "exit_code": 3, "error": str(error)}))
        return 3
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "failed", "exit_code": 2, "error": str(error)}))
        return 2


def register_parser(commands):
    reconstruct = commands.add_parser(
        "reconstruct", help="video to timestamped result using validated stage caches"
    )
    reconstruct.add_argument("video", type=Path)
    reconstruct.add_argument("--output", type=Path, required=True)
    lift = commands.add_parser(
        "lift", help="save raw EMA predictions using calibrated source observations"
    )
    lift.add_argument("--run", type=Path, required=True)
    export = commands.add_parser(
        "export", help="apply quality rules and write result/metrics JSON"
    )
    export.add_argument("--run", type=Path, required=True)
    reconstruct.add_argument("--device", choices=("auto", "cpu", "cuda", "intel-gpu"))
    lift.add_argument("--device", choices=("auto", "cpu", "cuda"))
    for parser in (reconstruct, lift):
        parser.add_argument("--checkpoint", type=Path)
    for parser in (reconstruct, lift, export):
        parser.add_argument("--config", type=Path)
        parser.add_argument("--overwrite", action="store_true")
        parser.set_defaults(handler=run_command)
