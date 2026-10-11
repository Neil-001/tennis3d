"""Qualify an installed Tennis3D: real weights, CLI reconstruction and the app.

Run with the installed environment's Python after `tennis3d setup`:

    python scripts/qualify_install.py --output qualification --device cpu

It renders a short synthetic broadcast-style clip (drawn court lines and a
moving ball; no real footage), reconstructs it with the installed detector and
lifter weights through the CLI, then repeats through the loopback app:
upload, progress, cancel, resume, result download and video range requests.
The clip checks that the whole pipeline executes on this machine. Its coverage
is recorded as measured and is not an accuracy result.
"""

import argparse
import http.client
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from tennis3d.artifacts import file_sha256, read_json, read_npz

HEADERS = {"X-Tennis3D": "1", "Content-Type": "application/octet-stream"}


def synthetic_clip(path: Path, frames=48, size=(1280, 720)):
    import av
    import cv2

    from tennis3d.court.model import (
        DOUBLES_WIDTH_M,
        LENGTH_M,
        SERVICE_LINE_DISTANCE_M,
        SINGLES_WIDTH_M,
    )
    from tennis3d.geometry.camera import Camera
    from tennis3d.geometry.image import ImageSize
    from tennis3d.geometry.projection import project

    width, height = size
    center = np.array([0.0, -30.0, 14.0])
    forward = (np.array([0.0, 2.0, 0.0]) - center) / np.linalg.norm(
        np.array([0.0, 2.0, 0.0]) - center
    )
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    rotation = np.array([right, np.cross(forward, right), forward])
    focal = 1.9 * width
    camera = Camera(
        np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]),
        rotation,
        -rotation @ center,
        ImageSize(width, height),
    )
    half, wide, narrow = LENGTH_M / 2, DOUBLES_WIDTH_M / 2, SINGLES_WIDTH_M / 2
    lines = [
        ((-wide, -half), (wide, -half)),
        ((-wide, half), (wide, half)),
        ((-wide, -half), (-wide, half)),
        ((wide, -half), (wide, half)),
        ((-narrow, -half), (-narrow, half)),
        ((narrow, -half), (narrow, half)),
        ((-narrow, -SERVICE_LINE_DISTANCE_M), (narrow, -SERVICE_LINE_DISTANCE_M)),
        ((-narrow, SERVICE_LINE_DISTANCE_M), (narrow, SERVICE_LINE_DISTANCE_M)),
        ((0, -SERVICE_LINE_DISTANCE_M), (0, SERVICE_LINE_DISTANCE_M)),
    ]
    background = np.full((height, width, 3), (150, 95, 40), dtype=np.uint8)
    apron = np.array(
        [[-wide - 4, -half - 6], [wide + 4, -half - 6], [wide + 4, half + 6]]
        + [[-wide - 4, half + 6]]
    )
    corners, _ = project(camera, np.c_[apron, np.zeros(4)])
    cv2.fillPoly(background, [np.rint(corners).astype(np.int32)], (170, 120, 50))
    for start, end in lines:
        ends, _ = project(camera, np.array([[*start, 0.0], [*end, 0.0]]))
        a, b = (tuple(int(v) for v in np.rint(p)) for p in ends)
        cv2.line(background, a, b, (255, 255, 255), 3, cv2.LINE_AA)
    net, _ = project(camera, np.array([[-wide - 0.9, 0, 1.0], [wide + 0.9, 0, 1.0]]))
    a, b = (tuple(int(v) for v in np.rint(p)) for p in net)
    cv2.line(background, a, b, (235, 235, 235), 2, cv2.LINE_AA)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"crf": "12", "preset": "veryfast"}
        stream.codec_context.time_base = Fraction(1, 12800)
        for index in range(frames):
            t = index / 25
            xyz = np.array(
                [[-2.0 + 1.5 * t, -9.0 + 9.0 * t, 1.0 + 7.0 * t - 4.9 * t * t]]
            )
            uv, _ = project(camera, xyz)
            image = background.copy()
            point = tuple(int(v) for v in np.rint(uv[0]))
            cv2.circle(image, point, 5, (60, 235, 215), -1, cv2.LINE_AA)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base = index * 512, Fraction(1, 12800)
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return path


def request(port, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    try:
        connection.request(method, path, body=body, headers=headers or HEADERS)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def wait(port, run_id, accept, timeout=3600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = json.loads(request(port, "GET", f"/api/runs/{run_id}")[2])
        if accept(run):
            return run
        if run["status"] in ("complete", "failed"):
            raise SystemExit(f"Run ended before the expected state: {run}")
        time.sleep(0.5)
    raise SystemExit(f"Run did not reach the expected state: {run}")


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def source_alignment(video, run, result):
    """Require every decoded source row, with its exact integer PTS/time base."""
    from tennis3d.video.decode import decode

    rows = [
        (frame.index, frame.pts, frame.time_base.as_integer_ratio(), frame.t_s)
        for frame in decode(video)
    ]
    arrays, _ = read_npz(run / "frames.npz")
    if result.source.sha256 != file_sha256(video) or len(result.frames) != len(rows):
        raise ValueError("Result does not retain all rows of the qualified source")
    for key, expected in (
        ("frame_index", [row[0] for row in rows]),
        ("pts", [row[1] for row in rows]),
        ("time_base", [row[2] for row in rows]),
        ("t_s", [row[3] for row in rows]),
    ):
        if not np.array_equal(arrays[key], expected):
            raise ValueError(f"Detection changed the source {key}")
    if [(f.frame_index, f.t_s) for f in result.frames] != [
        (row[0], row[3]) for row in rows
    ]:
        raise ValueError("Export changed source row indices or times")
    return {"rows": len(rows), "exact_pts_time_bases_and_export_times": True}


def result_parity(left, right):
    """Compare the full row states and gaps before measuring numerical parity."""
    if len(left.frames) != len(right.frames):
        raise ValueError("CLI and app row counts differ")
    fields = (
        "frame_index",
        "t_s",
        "observation",
        "reconstruction",
        "camera_valid",
        "reason_codes",
        "duplicate_of",
    )
    same = (
        left.status == right.status
        and left.failure_reasons == right.failure_reasons
        and all(
            all(getattr(a, key) == getattr(b, key) for key in fields)
            and (a.xyz_m is None) == (b.xyz_m is None)
            for a, b in zip(left.frames, right.frames, strict=True)
        )
    )
    difference = max(
        (
            abs(a - b)
            for x, y in zip(left.frames, right.frames, strict=True)
            if x.xyz_m is not None and y.xyz_m is not None
            for a, b in zip(x.xyz_m, y.xyz_m, strict=True)
        ),
        default=0.0,
    )
    return same, difference


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "intel-gpu"), default="cpu"
    )
    parser.add_argument("--video", type=Path, help="use this clip, not the synthetic")
    parser.add_argument("--runtime-timeout", type=float, default=600)
    parser.add_argument("--reconstruction-timeout", type=float, default=3600)
    parser.add_argument(
        "--keep-run", action="store_true", help="retain the qualified app run"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True)
    from tennis3d import __version__
    from tennis3d.app.home import detector_manifest, home, lifter_manifest, problems
    from tennis3d.app.server import App
    from tennis3d.perception.provenance import code_identity
    from tennis3d.schemas import read_result

    root = home()
    if missing := problems(root):
        raise SystemExit(" ".join(missing))
    import torch

    report = {
        "tennis3d": __version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "cpu_count": os.cpu_count(),
        "device_requested": args.device,
        "code_identity": code_identity(),
        "dependencies": {
            package: importlib.metadata.version(package)
            for package in (
                "numpy",
                "av",
                "torch",
                "opencv-python-headless",
                "scipy",
                "pykalman",
                "scikit-image",
            )
            + (("openvino",) if args.device == "intel-gpu" else ())
        },
        "process_runtime_environment": {
            key: os.environ.get(key) for key in ("OCL_ICD_VENDORS", "LD_LIBRARY_PATH")
        },
        "lifter_sha256": lifter_manifest()["sha256"],
        "environment_bytes": tree_bytes(Path(sys.prefix)),
        "weights_bytes": tree_bytes(root / "weights"),
    }
    video = (
        args.video.resolve() if args.video else synthetic_clip(output / "synthetic.mp4")
    )
    report["clip"] = "supplied" if args.video else "synthetic rendered court and ball"
    report["source_sha256"] = file_sha256(video)
    # The outer timeout also bounds native device discovery and model compilation.
    doctor = subprocess.run(
        [
            sys.executable,
            "-m",
            "tennis3d",
            "doctor",
            "--perception",
            "--device",
            args.device,
            "--assets-manifest",
            str(detector_manifest(root)),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=args.runtime_timeout,
        cwd=output,
    )
    (output / "doctor.stdout.txt").write_text(doctor.stdout)
    (output / "doctor.stderr.txt").write_text(doctor.stderr)
    if doctor.returncode:
        raise SystemExit(f"Model/device check failed: {doctor.stdout}\n{doctor.stderr}")
    report["doctor"] = json.loads(doctor.stdout)

    def cli(run):
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, "-m", "tennis3d", "reconstruct", str(video)]
            + ["--output", str(run), "--device", args.device],
            capture_output=True,
            text=True,
            cwd=output,
            check=False,
            timeout=args.reconstruction_timeout,
        )
        return completed, time.perf_counter() - started

    completed, seconds = cli(output / "cli-run")
    if completed.returncode not in (0, 4):
        raise SystemExit(f"CLI reconstruction failed:\n{completed.stdout}")
    summary = json.loads(completed.stdout)
    result = read_result(output / "cli-run" / "result.json")
    report["cli_source_alignment"] = source_alignment(video, output / "cli-run", result)
    lift = json.loads((output / "cli-run" / "lift.json").read_text())
    report["cli"] = {
        "exit_code": completed.returncode,
        "status": result.status,
        "frames": len(result.frames),
        "valid_frame_fraction": result.summary.valid_frame_fraction,
        "detected_frame_fraction": result.summary.detected_frame_fraction,
        "seconds": seconds,
        "stage_seconds": summary["invocation_timings"],
        "device": lift["runtime"]["device"],
        "checkpoint_sha256": lift["checkpoint_sha256"],
        "detection": read_json(output / "cli-run" / "detect.json"),
    }
    if lift["checkpoint_sha256"] != report["lifter_sha256"]:
        raise SystemExit("CLI did not use the installed lifter weights")
    again, seconds = cli(output / "cli-run")
    if again.returncode not in (0, 4):
        raise SystemExit(f"Cached CLI reconstruction failed: {again.stdout}")
    reused = json.loads(again.stdout)["reused_stages"]
    if reused != ["detect", "calibrate", "lift", "export"]:
        raise SystemExit(f"Second CLI run did not reuse its stages: {reused}")
    report["cli"]["cached_seconds"] = seconds

    import threading

    server = App(root, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_port
    try:
        if args.device == "intel-gpu":
            server.runs.intel.thread.join(timeout=30)
            if server.runs.intel.snapshot()["status"] != "available":
                raise SystemExit(
                    f"App Intel runtime unavailable: {server.runs.intel.snapshot()}"
                )
        state = json.loads(request(port, "GET", "/api/state")[2])
        report["app_capabilities"] = {
            "cuda": state["cuda"],
            "intel_gpu": state["intel_gpu"],
        }
        if not state["ready"]:
            raise SystemExit(f"App is not ready: {state['problems']}")
        status, _, body = request(
            port,
            "POST",
            f"/api/runs?name=qualify.mp4&device={args.device}",
            video.read_bytes(),
        )
        if status != 201:
            raise SystemExit(f"Upload failed: {body!r}")
        run_id = json.loads(body)["id"]
        started = time.perf_counter()
        wait(
            port,
            run_id,
            lambda r: (
                r.get("stage") == "detect"
                and r.get("progress", {}).get("completed", 0) >= 2
            ),
        )
        worker = server.runs.process
        assert worker is not None
        request(port, "POST", f"/api/runs/{run_id}/cancel")
        worker.wait(timeout=60)
        cancelled = wait(port, run_id, lambda r: r["status"] == "cancelled")
        request(port, "POST", f"/api/runs/{run_id}/start")
        done = wait(port, run_id, lambda r: r["status"] in ("complete", "failed"))
        if not done["has_result"] or not done["has_video"]:
            raise SystemExit(f"App run produced no result or video: {done}")
        served = request(port, "GET", f"/api/runs/{run_id}/result")[2]
        app_result = read_result(root / "runs" / run_id / "inference" / "result.json")
        app_run = root / "runs" / run_id / "inference"
        report["app_source_alignment"] = source_alignment(video, app_run, app_result)
        status, _, part = request(
            port, "GET", f"/api/runs/{run_id}/video", headers={"Range": "bytes=0-99"}
        )
        if status != 206 or len(part) != 100:
            raise SystemExit("Video range request failed")
        same, difference = result_parity(app_result, result)
        report["app"] = {
            "cancelled_status": cancelled["status"],
            "worker_exit_after_cancel": worker.returncode,
            "final_status": done["status"],
            "result_status": app_result.status,
            "seconds_including_cancel": time.perf_counter() - started,
            "result_bytes": len(served),
            "same_row_states_as_cli": same,
            "max_xyz_difference_from_cli_m": difference,
            "run_bytes": tree_bytes(root / "runs" / run_id),
            "run_id": run_id,
            "source_sha256": app_result.source.sha256,
            "detection": read_json(app_run / "detect.json"),
            "result_sha256": file_sha256(app_run / "result.json"),
        }
        if not same or difference > 1e-3:
            raise SystemExit(f"App and CLI results differ: {report['app']}")
        # A completed app rerun must reuse every stage and preserve exported bytes.
        request(port, "POST", f"/api/runs/{run_id}/start")
        cached = wait(port, run_id, lambda r: r["status"] in ("complete", "failed"))
        if (cached.get("summary") or {}).get("reused_stages") != [
            "detect",
            "calibrate",
            "lift",
            "export",
        ]:
            raise SystemExit(f"Completed app run did not reuse its cache: {cached}")
        if file_sha256(app_run / "result.json") != report["app"]["result_sha256"]:
            raise SystemExit("Cached app run changed the exported result")
        report["app"]["cached_reused_stages"] = cached["summary"]["reused_stages"]
        if not args.keep_run:
            request(port, "DELETE", f"/api/runs/{run_id}")
            if (root / "runs" / run_id).exists():
                raise SystemExit("Deleted run still exists")
    finally:
        server.shutdown()
        server.server_close()
    (output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
