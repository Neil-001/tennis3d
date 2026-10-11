"""Explicit search stop/resume. The browser never calls these mutations."""

import argparse
import json
from pathlib import Path

from tennis3d.autoresearch.log import append, lock, reconcile, snapshot
from tennis3d.autoresearch.segments import publish_best


def command(args: argparse.Namespace) -> int:
    run = args.run.resolve()
    try:
        if args.action == "stop":
            append(run, "control", action="stop")
        elif args.action == "resume":
            with lock(run / ".trial.lock", blocking=False):
                reconcile(run)
                publish_best(run)
                append(run, "control", action="resume")
        print(json.dumps(snapshot(run), indent=2, allow_nan=False))
    except (ValueError, OSError) as exc:
        print(f"experiment state: {exc}")
        return 2
    return 0


def register_parser(commands) -> None:
    parser = commands.add_parser(
        "experiment-state", help="inspect, stop or explicitly resume a search"
    )
    parser.add_argument("action", choices=("status", "stop", "resume"))
    parser.add_argument("--run", type=Path, required=True)
    parser.set_defaults(handler=command)
