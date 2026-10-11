"""Loopback HTTP server for the single-page reconstruction app.

Only 127.0.0.1 is bound. Requests must name this loopback origin in `Host`, and
state-changing requests must carry a custom header that a cross-site page
cannot send without a preflight, which is never granted. Uploaded names are
display text only; every stored path is derived from a generated identifier.
"""

import json
import os
import re
import sys
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tennis3d import __version__
from tennis3d.app.gallery import Gallery
from tennis3d.app.home import cuda_available, problems, resource
from tennis3d.app.runs import AlreadyRunning, RunError, RunStore
from tennis3d.artifacts import read_json, write_json

DEFAULT_PORT = 8770
MAX_UPLOAD_BYTES = 8 << 30
RANGE = re.compile(r"bytes=(\d*)-(\d*)")
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cache-Control": "no-store",
}


class App(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, root: Path, port: int, gallery: Path | None = None):
        super().__init__(("127.0.0.1", port), Handler)
        self.root = root
        try:
            self.runs = RunStore(root)
        except BaseException:
            super().server_close()
            raise
        self.gallery = Gallery(gallery, root / "gallery-playback")
        self.html = resource("dashboard.html")
        self.url = f"http://127.0.0.1:{self.server_port}/"
        self.hosts = {
            f"{name}:{self.server_port}" for name in ("127.0.0.1", "localhost")
        }
        self.origins = {f"http://{host}" for host in self.hosts}

    def state(self) -> dict:
        missing = problems(self.root)
        return {
            "version": __version__,
            "ready": not missing,
            "problems": missing,
            "cuda": cuda_available(),
            "intel_gpu": self.runs.intel.snapshot(),
            "runs": self.runs.list(),
            "gallery": self.gallery.describe(),
        }

    def handle_error(self, request, client_address):
        # A browser drops its video connections whenever it seeks. Reporting
        # each one would fill the terminal with tracebacks.
        if not isinstance(sys.exception(), ConnectionError):
            super().handle_error(request, client_address)

    def server_close(self):
        # The base class calls this when the port cannot be bound, before any
        # run store exists.
        runs = getattr(self, "runs", None)
        if runs is not None:
            runs.close()
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    server: App
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def reply(self, status, body=b"", content_type="application/json", headers=None):
        self.send_response(status)
        for key, value in (SECURITY_HEADERS | (headers or {})).items():
            self.send_header(key, value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def json(self, value, status=HTTPStatus.OK):
        self.reply(status, json.dumps(value).encode())

    def error(self, status, message):
        self.close_connection = True
        self.json({"error": message}, status)

    def file(self, path: Path, content_type: str, download: str | None = None):
        """Serve a file, honouring one byte range so browsers can seek video."""
        size = path.stat().st_size
        start, end, status = 0, size - 1, HTTPStatus.OK
        if match := RANGE.fullmatch(self.headers.get("Range", "")):
            first, last = match.groups()
            if first:
                start, end = int(first), min(int(last) if last else end, end)
            elif last:
                start = max(0, size - int(last))
            if start > end or start >= size:
                return self.reply(
                    HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                    headers={"Content-Range": f"bytes */{size}"},
                )
            status = HTTPStatus.PARTIAL_CONTENT
        self.send_response(status)
        headers = SECURITY_HEADERS | {
            "Content-Type": content_type,
            "Content-Length": str(end - start + 1),
            "Accept-Ranges": "bytes",
        }
        if status == HTTPStatus.PARTIAL_CONTENT:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        if download:
            headers["Content-Disposition"] = f'attachment; filename="{download}"'
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining and (block := stream.read(min(1 << 18, remaining))):
                self.wfile.write(block)
                remaining -= len(block)

    def route(self, method):
        if self.headers.get("Host") not in self.server.hosts:
            return self.error(HTTPStatus.FORBIDDEN, "Unrecognized host")
        url = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(url.query, keep_blank_values=True)
        parts = [urllib.parse.unquote(part) for part in url.path.split("/") if part]
        if method != "GET":
            origin = self.headers.get("Origin")
            if self.headers.get("X-Tennis3D") != "1" or (
                origin and origin not in self.server.origins
            ):
                return self.error(HTTPStatus.FORBIDDEN, "Request not allowed")
        try:
            if method == "GET" and not parts:
                return self.reply(
                    HTTPStatus.OK, self.server.html, "text/html; charset=utf-8"
                )
            if parts[:1] != ["api"]:
                raise KeyError(url.path)
            if method == "GET" and parts[1:] == ["state"]:
                return self.json(self.server.state())
            if parts[1:2] == ["runs"]:
                return self.runs(method, parts[2:], query)
            if method == "GET" and parts[1:2] == ["gallery"] and len(parts) == 4:
                return self.gallery(parts[2], parts[3], query)
            raise KeyError(url.path)
        except RunError as error:
            self.error(HTTPStatus.BAD_REQUEST, str(error))
        except (KeyError, FileNotFoundError):
            self.error(HTTPStatus.NOT_FOUND, "Not found")
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as error:  # noqa: BLE001 - request boundary
            self.error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))

    def runs(self, method, parts, query):
        store = self.server.runs
        if method == "POST" and not parts:
            length = self.headers.get("Content-Length", "")
            if not length.isdigit() or not 0 < int(length) <= MAX_UPLOAD_BYTES:
                raise RunError("Choose a non-empty video file smaller than 8 GB.")
            run = store.create(
                query.get("name", [""])[0],
                self.rfile,
                int(length),
                query.get("device", ["auto"])[0],
            )
            return self.json(run, HTTPStatus.CREATED)
        if not parts:
            raise KeyError("runs")
        run_id, action = parts[0], parts[1:]
        if method == "DELETE" and not action:
            store.delete(run_id)
            return self.json({"deleted": run_id})
        if method == "POST" and action == ["start"]:
            return self.json(store.start(run_id, query.get("device", [None])[0]))
        if method == "POST" and action == ["cancel"]:
            return self.json(store.cancel(run_id))
        if method != "GET":
            raise KeyError(run_id)
        path = store.path(run_id)
        if not action:
            return self.json(store.get(run_id))
        if action == ["result"]:
            name = f"tennis3d-{run_id}.json" if "download" in query else None
            return self.file(
                path / "inference" / "result.json", "application/json", name
            )
        if action == ["video"] and store.get(run_id)["has_video"]:
            return self.file(path / "playback.mp4", "video/mp4")
        raise KeyError(run_id)

    def gallery(self, clip_id, action, query):
        gallery = self.server.gallery
        if action == "timeline":
            return self.file(gallery.timeline(clip_id), "application/json")
        if action == "result":
            download = f"tennis3d-{clip_id}.json" if "download" in query else None
            return self.file(gallery.result(clip_id), "application/json", download)
        if action == "video":
            return self.file(gallery.video(clip_id), "video/mp4")
        raise KeyError(clip_id)

    def do_GET(self):
        self.route("GET")

    do_HEAD = do_GET

    def do_POST(self):
        self.route("POST")

    def do_DELETE(self):
        self.route("DELETE")


def serve(root: Path, *, port=None, gallery=None, open_browser=True) -> int:
    """Run the app until interrupted. Records its own process for lifecycle checks."""
    import signal
    import threading
    import webbrowser

    root.mkdir(parents=True, exist_ok=True)
    gallery = gallery or (
        root / "gallery.json" if (root / "gallery.json").is_file() else None
    )
    try:
        try:
            server = App(root, DEFAULT_PORT if port is None else port, gallery)
        except OSError:
            if port is not None:
                raise
            server = App(root, 0, gallery)  # Default port is busy; take a free one.
    except AlreadyRunning:
        try:
            url = read_json(root / "server.json")["url"]
        except (OSError, ValueError, KeyError):
            url = None
        print(f"Tennis3D is already running{f' at {url}' if url else ''}.", flush=True)
        if url and open_browser:
            webbrowser.open(url)
        return 0
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    restored = server.gallery.restore_line_endings()
    if restored:
        print(f"Restored the line endings of {len(restored)} example files.")
    record = root / "server.json"
    write_json(record, {"pid": os.getpid(), "url": server.url}, overwrite=True)
    print(f"Tennis3D is running at {server.url}\nPress Ctrl+C to stop.", flush=True)
    threading.Thread(target=server.gallery.prepare, daemon=True).start()
    if open_browser:
        threading.Thread(
            target=webbrowser.open, args=(server.url,), daemon=True
        ).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        record.unlink(missing_ok=True)
    return 0
