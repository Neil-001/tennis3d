"""Lightweight parser registration; optional numerical libraries load on execution."""

import json
from pathlib import Path

from tennis3d.config import load_config
from tennis3d.perception.assets import AssetIntegrityError, MissingAssetError
from tennis3d.perception.config import CalibrationConfig, DetectConfig


def run_command(args):
    try:
        if args.command == "assets":
            from tennis3d.perception.assets import fetch, manifest

            assets = manifest(args.manifest)
            for asset in assets.assets.values():
                fetch(asset, overwrite=args.overwrite)
            print(json.dumps({"status": "ok", "assets": list(assets.assets)}))
            return 0
        if args.command == "detect":
            from tennis3d.perception.detect import detect

            overrides = {
                key: getattr(args, key)
                for key in ("device", "assets_manifest")
                if getattr(args, key) is not None
            }
            config = load_config(DetectConfig, args.config, overrides=overrides)
            summary = detect(args.video, args.output, config, overwrite=args.overwrite)
        else:
            from tennis3d.perception.calibrate import calibrate

            config = load_config(
                CalibrationConfig,
                args.config,
                overrides={"mode": args.mode} if args.mode else {},
            )
            summary = calibrate(args.run, config, overwrite=args.overwrite)
        print(json.dumps(summary, indent=2))
        return summary.get("exit_code", 0 if summary["status"] != "failed" else 4)
    except (ImportError, RuntimeError, MissingAssetError, AssetIntegrityError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 3
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 2


def register_parser(commands):
    assets = commands.add_parser(
        "assets", help="explicitly fetch verified local inference assets"
    )
    actions = assets.add_subparsers(dest="asset_action", required=True)
    fetch = actions.add_parser("fetch")
    fetch.add_argument("--manifest", type=Path, default=Path("configs/assets.yaml"))
    fetch.add_argument("--overwrite", action="store_true")
    fetch.set_defaults(handler=run_command)
    detect = commands.add_parser(
        "detect", help="decode timestamps and save raw tennis detector evidence"
    )
    detect.add_argument("video", type=Path)
    detect.add_argument("--output", type=Path, required=True)
    detect.add_argument("--device", choices=("auto", "cpu", "cuda", "intel-gpu"))
    detect.add_argument("--assets-manifest")
    calibrate = commands.add_parser(
        "calibrate", help="estimate and filter metric cameras from raw landmarks"
    )
    calibrate.add_argument("--run", type=Path, required=True)
    calibrate.add_argument("--mode", choices=("static", "moving"))
    for parser in (detect, calibrate):
        parser.add_argument("--config", type=Path)
        parser.add_argument("--overwrite", action="store_true")
        parser.set_defaults(handler=run_command)
