"""User-facing commands: the local app, weight setup and the compact export."""

import json
import os
from pathlib import Path

from tennis3d.perception.assets import AssetIntegrityError, MissingAssetError


def dashboard(args) -> int:
    if args.run is not None:
        # Compatible spelling of `tennis3d monitor`.
        if os.name != "posix":
            print("The experiment monitor needs Linux or macOS.")
            return 2
        from tennis3d.autoresearch.dashboard import command

        args.port = 8765 if args.port is None else args.port
        return command(args)
    from tennis3d.app.home import home
    from tennis3d.app.server import serve

    try:
        return serve(
            home(args.home),
            port=args.port,
            gallery=args.gallery,
            open_browser=not args.no_open,
        )
    except (OSError, ValueError, KeyError) as error:
        print(f"Tennis3D could not start: {error}")
        return 2


def setup(args) -> int:
    from tennis3d.app import home

    root = home.home(args.home)
    try:
        if args.check:
            missing = home.problems(root)
            print(
                json.dumps(
                    {"home": str(root), "ready": not missing, "problems": missing},
                    indent=2,
                )
            )
            return 3 if missing else 0
        report = home.setup(root, weights=args.weights, offline=args.offline)
        print(json.dumps(report | {"status": "ok"}, indent=2))
        return 0
    except (MissingAssetError, AssetIntegrityError, OSError) as error:
        print(json.dumps({"status": "failed", "exit_code": 3, "error": str(error)}))
        return 3


def export_lifter(args) -> int:
    try:
        from tennis3d.lifting.export import export_lifter, qualify_export

        manifest = export_lifter(args.checkpoint, args.output, overwrite=args.overwrite)
        equality = qualify_export(args.checkpoint, args.output)
        print(json.dumps({"manifest": manifest, "qualification": equality}, indent=2))
        return 0
    except ImportError as error:
        print(json.dumps({"status": "failed", "exit_code": 3, "error": str(error)}))
        return 3
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "failed", "exit_code": 2, "error": str(error)}))
        return 2


def gallery(args) -> int:
    from tennis3d.app.gallery import Gallery
    from tennis3d.app.home import home

    try:
        states = Gallery(args.manifest, home(args.home) / "gallery-playback").prepare()
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "failed", "exit_code": 2, "error": str(error)}))
        return 2
    print(json.dumps({"status": "ok", "videos": states}, indent=2))
    return 0


def register_parser(commands):
    if os.name == "posix":
        from tennis3d.autoresearch.dashboard import (
            register_parser as register_monitor,
        )

        register_monitor(commands, "monitor")
    app = commands.add_parser(
        "dashboard",
        help="open the local reconstruction app (--run serves the experiment monitor)",
    )
    app.add_argument(
        "--port", type=int, help="loopback port; default 8770 or a free one"
    )
    app.add_argument("--home", type=Path, help="app directory; default ~/.tennis3d")
    app.add_argument("--gallery", type=Path, help="precomputed examples manifest")
    app.add_argument("--no-open", action="store_true", help="do not open a browser")
    app.add_argument(
        "--run", type=Path, help="experiment run for the read-only monitor"
    )
    app.add_argument("--archive-run", type=Path, action="append", default=[])
    app.add_argument("--kind", choices=("accuracy", "throughput"), default="accuracy")
    app.set_defaults(handler=dashboard)
    install = commands.add_parser(
        "setup", help="download or import the pinned model weights"
    )
    install.add_argument("--home", type=Path, help="app directory; default ~/.tennis3d")
    install.add_argument(
        "--weights",
        type=Path,
        action="append",
        default=[],
        help="local weight file or directory to import instead of downloading",
    )
    install.add_argument("--offline", action="store_true", help="never download")
    install.add_argument("--check", action="store_true", help="only report readiness")
    install.set_defaults(handler=setup)
    export = commands.add_parser(
        "export-lifter", help="write a compact EMA inference export and its manifest"
    )
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--overwrite", action="store_true")
    export.set_defaults(handler=export_lifter)
    examples = commands.add_parser(
        "gallery", help="prepare browser playback copies for a precomputed gallery"
    )
    examples.add_argument("--manifest", type=Path, required=True)
    examples.add_argument("--home", type=Path)
    examples.set_defaults(handler=gallery)
