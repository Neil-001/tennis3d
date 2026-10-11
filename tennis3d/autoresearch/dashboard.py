"""Local GET-only monitoring. No evaluation, reconciliation or controls."""

import argparse
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from tennis3d.artifacts import read_json
from tennis3d.autoresearch.log import EventTail, Projection
from tennis3d.autoresearch.segments import safe_name
from tennis3d.autoresearch.throughput_dashboard import ThroughputHistory

ASSETS = {
    "/": "dashboard.html",
    "/dashboard.js": "dashboard.js",
    "/style.css": "style.css",
}
TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}


def trial_details(
    run: Path, trial_id: str, state: dict, segment: str | None = None
) -> dict:
    matches = [
        t
        for t in state["trials"]
        if t["trial"] == trial_id and (segment is None or t["segment"] == segment)
    ]
    if not matches:
        raise ValueError("unknown trial")
    if len(matches) != 1:
        raise ValueError("ambiguous trial ID; include the segment from /api/snapshot")
    trial = matches[0]
    directory = (run / safe_name(trial_id)).resolve()
    if directory.parent != run.resolve():
        raise ValueError("trial directory must be inside this run")
    details = {"trial": trial, "curve": []}
    curve = directory / "curve.jsonl"
    if curve.exists():
        for line in curve.read_text().splitlines():
            try:
                point = json.loads(line)
            except ValueError:
                # A concurrently appended final line can be incomplete.
                details["curve_warning"] = "Incomplete or malformed loss curve record"
                continue
            details["curve"].append(point)
    for name in ("metrics", "config.resolved", "checks"):
        path = directory / f"{name}.json"
        if path.exists():
            details[name] = read_json(path)
    return details


class RunHistory:
    """Keep each directory's event identities and reader cache independent."""

    def __init__(self, run: Path, *, archived: bool):
        self.run = run.resolve()
        self.archived = archived
        # Archive keys stay stable across restarts and CLI argument ordering.
        identity = hashlib.sha256(str(self.run).encode()).hexdigest()
        self.prefix = f"archive:{identity}:" if archived else ""
        self.reader = EventTail(self.run / "experiments.jsonl")
        self.projection = Projection()

    def segment_key(self, segment: str) -> str:
        return self.prefix + segment

    def view(self):
        try:
            reset, events = self.reader.poll()
            if reset:
                self.projection = Projection()
            self.projection.apply(events)
            state = self.projection.view(self.reader.warnings)
        except (ValueError, OSError):
            # A failed update must never leave a partly applied cached state.
            self.reader = EventTail(self.run / "experiments.jsonl")
            self.projection = Projection()
            raise
        metadata = {
            "archived": self.archived,
            "source_run": hashlib.sha256(str(self.run).encode()).hexdigest(),
            "kind": "accuracy",
        }
        state["segments"] = [
            {
                **s,
                **metadata,
                "segment": self.segment_key(s["segment"]),
                "segment_name": s["segment"],
                "stopped": state["stopped"],
                "last_update": state["last_update"],
            }
            for s in state["segments"]
        ]
        state["trials"] = [
            {**t, **metadata, "segment": self.segment_key(t["segment"])}
            for t in state["trials"]
        ]
        state["best"] = {
            self.segment_key(segment): best for segment, best in state["best"].items()
        }
        return state


def make_server(
    run: Path,
    port: int = 8765,
    *,
    archive_runs: list[Path] | None = None,
    kind: str = "accuracy",
) -> ThreadingHTTPServer:
    history_lock = threading.Lock()
    current = (ThroughputHistory if kind == "throughput" else RunHistory)(
        run, archived=False
    )
    # Resolve aliases and repeated arguments without displaying a run twice.
    archives = dict.fromkeys(path.resolve() for path in archive_runs or [])
    archives.pop(current.run, None)
    histories = [RunHistory(path, archived=True) for path in archives] + [current]
    sources = {
        hashlib.sha256(str(history.run).encode()).hexdigest(): history
        for history in histories
    }

    def current_state():
        states: list[dict] = []
        for history in histories:
            try:
                states.append(history.view())
            except (ValueError, OSError):
                if not history.archived:
                    raise
                states.append(
                    {
                        "segments": [],
                        "trials": [],
                        "best": {},
                        "warnings": [
                            "Archive unavailable: unreadable or inconsistent logs"
                        ],
                    }
                )
        # Stop state, last update and server clock always describe --run.
        state = {
            **states[-1],
            "segments": [s for source in states for s in source["segments"]],
            "trials": [t for source in states for t in source["trials"]],
            "best": {k: v for source in states for k, v in source["best"].items()},
            "warnings": [
                f"{'Archived' if history.archived else 'Current'} campaign: {warning}"
                for history, source in zip(histories, states)
                for warning in source["warnings"]
            ],
        }
        return state

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlparse(self.path)
            try:
                if request.path in ASSETS:
                    asset = Path(__file__).with_name(ASSETS[request.path])
                    self.respond(asset.read_bytes(), TYPES[asset.suffix])
                elif request.path == "/api/snapshot":
                    with history_lock:
                        body = json.dumps(current_state(), allow_nan=False).encode()
                    self.respond(body, "application/json")
                elif request.path == "/api/trial":
                    query = parse_qs(request.query)
                    with history_lock:
                        state = current_state()
                        trial_id = query.get("id", [""])[0]
                        segment = query.get("segment", [None])[0]
                        matches = [
                            t
                            for t in state["trials"]
                            if t["trial"] == trial_id
                            and (segment is None or t["segment"] == segment)
                        ]
                        if not matches:
                            raise ValueError("unknown trial")
                        if len(matches) != 1:
                            raise ValueError(
                                "ambiguous trial ID; include the segment from /api/snapshot"
                            )
                        source = sources[matches[0]["source_run"]]
                        details = (
                            source.details(matches[0])
                            if isinstance(source, ThroughputHistory)
                            else trial_details(
                                source.run, trial_id, {"trials": matches}, segment
                            )
                        )
                        body = json.dumps(details, allow_nan=False).encode()
                    self.respond(body, "application/json")
                else:
                    self.send_error(404)
            except (ValueError, OSError) as exc:
                message = (
                    str(exc)
                    if isinstance(exc, ValueError)
                    and str(exc)
                    in (
                        "unknown trial",
                        "ambiguous trial ID; include the segment from /api/snapshot",
                        "trial directory must be inside this run",
                    )
                    else "Unreadable or inconsistent campaign artifacts"
                )
                self.respond_json({"error": message}, status=422)

        def respond_json(self, value, status=200):
            self.respond(
                json.dumps(value, allow_nan=False).encode(), "application/json", status
            )

        def respond(self, body: bytes, mime: str, status=200):
            self.send_response(status)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; connect-src 'self'; script-src 'self'; style-src 'self'",
            )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def command(args: argparse.Namespace) -> int:
    with make_server(
        args.run, args.port, archive_runs=args.archive_run, kind=args.kind
    ) as server:
        print(f"Read-only dashboard: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


def register_parser(commands, name="dashboard") -> None:
    parser = commands.add_parser(
        name, help="serve the local read-only experiment dashboard"
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--archive-run",
        type=Path,
        action="append",
        default=[],
        help="also display this archived run directory; repeat for more archives",
    )
    parser.add_argument(
        "--kind",
        choices=("accuracy", "throughput"),
        default="accuracy",
        help="explicit type of --run, even before it exists; archives are accuracy campaigns",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.set_defaults(handler=command)
