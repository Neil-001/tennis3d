"""Detection/calibration stages prepared once and shared by several lifters.

`prepare` runs the stock `tennis3d detect` and `tennis3d calibrate` commands on
every eligible clip of a frozen inventory. Each detection subprocess holds a
shared file lock for its whole life, has a time limit fixed from the clip's
frame count before anything starts, and is never retried or replaced by
another device.

`freeze` copies finished stage directories into a new gallery-shaped root and
writes the `real-artifact-interface/v1` manifest that `cached_real.py`
requires. It runs the checks `cached_real.py` applies to cached stages, plus
the frozen source timeline, on the copies. The recorded producer identity of
the stages is kept verbatim; nothing is recomputed or relabelled. No lifter
runs here and no model output is read.
"""

import argparse
import fcntl
import importlib.util
import json
import os
import resource
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import numpy as np

from tennis3d.artifacts import atomic_file, file_sha256, read_json, write_json
from tennis3d.config import load_config
from tennis3d.evaluation.delivery import fresh_output
from tennis3d.evaluation.inventory import read_inventory, verify_clip
from tennis3d.inference.settings import ReconstructionConfig
from tennis3d.perception.config import CalibrationConfig, DetectConfig
from tennis3d.perception.provenance import OPENVINO_DEVICES
from tennis3d.provenance import code_identity as package_identity

_SPEC = importlib.util.spec_from_file_location(
    "cached_real", Path(__file__).with_name("cached_real.py")
)
assert _SPEC and _SPEC.loader
cached_real = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cached_real)

# Detection options that select assets, hardware and diagnostics, not numerics.
OPERATIONAL = ("assets_manifest", "device", "threads", "save_heatmaps", "overlay_every")
KERNEL_EVENTS = ("GPU HANG", "Fence expiration")
INTERRUPTIONS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
STATEMENT = (
    "Detection and calibration ran once per clip and are shared unchanged by "
    "every model evaluated on this root. Stage producer identities, runtimes "
    "and timings are those recorded inside the stage files."
)


class StageError(Exception):
    """The stages cannot be prepared or trusted; nothing is published."""


def pinned_inventory(path, expected_sha256):
    if file_sha256(path) != expected_sha256:
        raise StageError("inventory differs from --inventory-sha256")
    return read_inventory(path)


def now():
    return datetime.now(UTC).isoformat()


def kernel_events(since):
    """Graphics driver complaints in the kernel log since a local time, if readable."""
    try:
        done = subprocess.run(
            ["journalctl", "-k", "-b", "--no-pager", "--since", since],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"readable": False, "error": str(error), "lines": []}
    return {
        "readable": done.returncode == 0,
        "lines": [
            line
            for line in done.stdout.splitlines()
            if any(event in line for event in KERNEL_EVENTS)
        ],
    }


def group_running(pgid):
    """Whether any member of a process group still runs; zombies do not count."""
    proc = Path("/proc")
    if not proc.is_dir():
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        return True
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # The command name may hold spaces; fields follow its closing bracket.
            state, _, group = (entry / "stat").read_text().rsplit(")", 1)[1].split()[:3]
        except (OSError, IndexError, ValueError):
            continue
        if int(group) == pgid and state != "Z":
            return True
    return False


def stop_group(process):
    """Signal a command's whole process group and wait until none of it runs.

    A member that survives SIGKILL is stuck in the kernel and may still hold
    the device, so the wait, and with it the lock, lasts as long as it does.
    """
    for sign, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, None)):
        try:
            os.killpg(process.pid, sign)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=grace)
            break
        except subprocess.TimeoutExpired:
            continue
    # The leader may exit first; nothing it started outlives this call.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()
    while group_running(process.pid):
        time.sleep(0.05)


def supervised(command, limit_s, log, lock=None):
    """Run one command in its own process group, under the lock when given.

    The lock is taken before the process starts and released only after the
    whole group has exited: after a normal exit, a timeout, or an interruption
    of this supervisor. From before the process starts until after the lock is
    released, interruptions are only recorded, never raised, so none can skip
    or cut short the cleanup; one is raised at the end. Handlers are replaced,
    not masked: threads of numerical libraries would still receive a masked
    process signal.
    """
    record = {"command": [str(part) for part in command], "limit_s": limit_s}
    held = None
    if lock is not None:
        held = open(lock, "a")  # noqa: SIM115
        then = time.perf_counter()
        fcntl.flock(held, fcntl.LOCK_EX)
        record["lock_wait_s"] = time.perf_counter() - then
    received, previous, process = [], {}, None
    try:
        for sign in INTERRUPTIONS:
            previous[sign] = signal.signal(
                sign, lambda number, _: received.append(number)
            )
        record["started_utc"] = now()
        since = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        then = time.perf_counter()
        with (
            open(f"{log}.stdout", "wb") as out,
            open(f"{log}.stderr", "wb") as err,
        ):
            try:
                process = subprocess.Popen(
                    command, stdout=out, stderr=err, start_new_session=True
                )
                record["timed_out"] = False
                while not received:
                    left = limit_s - (time.perf_counter() - then)
                    if left <= 0:
                        record["timed_out"] = True
                        break
                    try:
                        process.wait(timeout=min(left, 0.2))
                        break
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                if process is not None:
                    stop_group(process)
        if not received:
            record["wall_s"] = time.perf_counter() - then
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            record["children_cpu_s"] = (
                after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
            )
            record["returncode"] = process.returncode
            record["ended_utc"] = now()
            if lock is not None:
                record["kernel_events"] = kernel_events(since)
    finally:
        if held is not None:
            fcntl.flock(held, fcntl.LOCK_UN)
            held.close()
        for sign, handler in previous.items():
            signal.signal(sign, handler)
    if received:
        raise KeyboardInterrupt
    return record


def finished(record, summary, graphics=False):
    """The command exited by itself with the status its own summary records.

    A timeout, a signal, a runtime failure (exit status 3), an exit status the
    summary does not explain, or, for a stage on the graphics device, a kernel
    graphics event while the lock was held is a failure of the run, not of the
    clip. A CPU stage records such events without being stopped by them.
    """
    if record["timed_out"] or summary is None:
        return False
    # A graphics fault in the kernel log during the locked stage, or a log
    # that cannot be read, stops the run even if the command reported success.
    events = record.get("kernel_events") if graphics else None
    if events is not None and (events["lines"] or not events["readable"]):
        return False
    expected = summary.get("exit_code", 4 if summary["status"] == "failed" else 0)
    return record["returncode"] == expected and expected in (0, 2, 4)


def stage_recorded(run, name, source_sha256=None):
    """A stage summary the stock command itself published for this run."""
    try:
        summary = read_json(run / f"{name}.json")
    except (OSError, ValueError):
        return None
    if not isinstance(summary, dict) or summary.get("stage") != name:
        return None
    if source_sha256 is not None and summary.get("source_sha256") != source_sha256:
        return None
    return summary


def prepare(
    inventory_path,
    inventory_sha256,
    output,
    executable,
    detect_config,
    lock,
    *,
    base_s,
    per_frame_s,
    calibrate_s,
):
    """Stock detect and calibrate once per eligible clip, bounded and locked.

    A stage that published its own summary keeps its recorded status, failed or
    not. A detection that timed out, died or left no summary stops the run with
    its files and logs in place; later clips are not attempted.
    """
    inventory = pinned_inventory(inventory_path, inventory_sha256)
    clips = [clip for clip in inventory.clips if clip.eligible]
    for clip in clips:
        verify_clip(clip)
    config = load_config(DetectConfig, detect_config)
    # Only a detector on the graphics device is judged by kernel graphics events.
    graphics = config.device in OPENVINO_DEVICES
    output = fresh_output(output)
    bounds = {
        "frozen_utc": now(),
        "rule": "detect limit = base_s + per_frame_s * frame_count; fixed before any stage starts",
        "base_s": base_s,
        "per_frame_s": per_frame_s,
        "calibrate_limit_s": calibrate_s,
        "inventory": str(Path(inventory_path).resolve()),
        "inventory_sha256": inventory_sha256,
        "executable": str(executable),
        "detect_config": config.model_dump(mode="json"),
        "detect_config_file_sha256": file_sha256(detect_config),
        "calibrate_mode": "moving",
        "lock": str(lock),
        "kernel_graphics_event_stops_run": graphics,
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "package": {
            key: value
            for key, value in package_identity().items()
            if key != "source_files"
        },
        "script_sha256": file_sha256(__file__),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "LD_LIBRARY_PATH",
                "OCL_ICD_VENDORS",
                "ZE_ENABLE_ALT_DRIVERS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "clips": {
            clip.clip_id: {
                "frame_count": clip.frame_count,
                "detect_limit_s": base_s + per_frame_s * (clip.frame_count or 0),
            }
            for clip in clips
        },
    }
    write_json(output / "bounds.json", bounds)
    (output / "logs").mkdir()
    started, records, blocker = time.perf_counter(), [], None
    for clip in clips:
        run = output / "stages" / clip.clip_id
        record = {"clip_id": clip.clip_id, "frame_count": clip.frame_count}
        record["detect"] = supervised(
            [
                executable,
                "detect",
                clip.source_path,
                "--output",
                run,
                "--config",
                detect_config,
            ],
            bounds["clips"][clip.clip_id]["detect_limit_s"],
            output / "logs" / f"{clip.clip_id}.detect",
            lock,
        )
        detection = stage_recorded(run, "detect", clip.source_sha256)
        if not finished(record["detect"], detection, graphics):
            blocker = (
                f"{clip.clip_id}: detection did not finish cleanly with its summary"
            )
        else:
            assert detection is not None
            record["detect"]["status"] = detection["status"]
            record["calibrate"] = supervised(
                [executable, "calibrate", "--run", run, "--mode", "moving"],
                calibrate_s,
                output / "logs" / f"{clip.clip_id}.calibrate",
            )
            calibration = stage_recorded(run, "calibrate")
            if not finished(record["calibrate"], calibration):
                blocker = f"{clip.clip_id}: calibration did not finish with its summary"
            else:
                assert calibration is not None
                record["calibrate"]["status"] = calibration["status"]
        records.append(record)
        with open(output / "records.jsonl", "a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if blocker:
            break
    summary = {
        "status": "blocked" if blocker else "complete",
        "blocker": blocker,
        "declaration": "No retry, resubmission or other device was attempted.",
        "clips_requested": len(clips),
        "clips_attempted": len(records),
        "process_wall_s": time.perf_counter() - started,
        "lock_wait_s": sum(row["detect"]["lock_wait_s"] for row in records),
        "detect_wall_s": sum(row["detect"]["wall_s"] for row in records),
        "detect_children_cpu_s": sum(
            row["detect"]["children_cpu_s"] for row in records
        ),
        "calibrate_wall_s": sum(
            row["calibrate"]["wall_s"] for row in records if "calibrate" in row
        ),
        "calibrate_children_cpu_s": sum(
            row["calibrate"]["children_cpu_s"] for row in records if "calibrate" in row
        ),
        "kernel_event_lines": sum(
            len(row["detect"]["kernel_events"]["lines"]) for row in records
        ),
        "bounds_sha256": file_sha256(output / "bounds.json"),
        "records": records,
    }
    write_json(output / "prepare.json", summary)
    return summary


def frozen_rows(clip, frames):
    """The stage's source rows are exactly the inventory's frozen timeline."""
    rows = read_json(clip.timeline_path)["frames"]
    pts = [
        Fraction(row["pts"] * row["time_base"][0], row["time_base"][1]) for row in rows
    ]
    expected = np.array([float(value - pts[0]) for value in pts])
    if (
        len(frames["t_s"]) != clip.frame_count
        or list(map(int, frames["image_size"])) != [clip.width, clip.height]
        or not np.array_equal(frames["pts"], [row["pts"] for row in rows])
        or not np.array_equal(frames["time_base"], [row["time_base"] for row in rows])
        or not np.allclose(frames["t_s"], expected, atol=1e-9, rtol=0)
    ):
        raise ValueError("stage rows differ from the frozen source timeline")


def released_entries(path, expected_sha256):
    """Released files by absolute path, from a pinned list of path, size and hash."""
    if file_sha256(path) != expected_sha256:
        raise StageError("released file list differs from --released-sha256")
    return {row["path"]: row for row in read_json(path)}


def freeze(
    inventory_path,
    inventory_sha256,
    stages,
    output,
    manifest_path,
    bindings,
    released=None,
):
    """Copy verified stage files into a new root and write its artifact manifest.

    Every eligible clip must have all nine files, one detection configuration,
    one calibration configuration and one recorded producer. Thresholds must
    equal the reviewed defaults; only asset, device and diagnostic options may
    differ. With `released`, a (path, SHA-256) pair naming a released file
    list, every stage file must be listed there with the same size and hash.
    The manifest is written last, so a stopped freeze leaves none.
    """
    inventory = pinned_inventory(inventory_path, inventory_sha256)
    stages = Path(stages).resolve()
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        raise StageError("artifact manifest already exists")
    bound = {}
    for name, path in bindings.items():
        path = Path(path).resolve()
        bound[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    clips = [clip for clip in inventory.clips if clip.eligible]
    if not clips:
        raise StageError("inventory has no eligible clip")
    listed = None
    if released is not None:
        listed = released_entries(*released)
        bound["released_files"] = {
            "path": str(Path(released[0]).resolve()),
            "sha256": released[1],
            "bytes": Path(released[0]).stat().st_size,
        }
    resolved = Path(output).resolve()
    if resolved == stages or stages in resolved.parents or resolved in stages.parents:
        raise StageError("output must be separate from the stage directories")
    output = fresh_output(output)
    config, rows = None, []
    for clip in clips:
        source, run = stages / clip.clip_id, output / "clips" / clip.clip_id
        try:
            verify_clip(clip)
            for name in cached_real.UPSTREAM_FILES:
                data = (source / name).read_bytes()
                if listed is not None:
                    row = listed.get(str(source / name))
                    if row is None or (row["bytes"], row["sha256"]) != (
                        len(data),
                        cached_real._sha256(data),
                    ):
                        raise ValueError(f"{name} differs from the released file list")
                with atomic_file(run / name) as stream:
                    stream.write(data)
            recorded = ReconstructionConfig(
                detection=load_config(DetectConfig, run / "detect.config.json"),
                calibration=load_config(
                    CalibrationConfig, run / "calibrate.config.json"
                ),
            )
            if config is None:
                config = recorded
            if recorded != config:
                raise ValueError("stage configuration differs between clips")
            detection, frames, _, detector = cached_real.cached_detection(
                run, config.detection
            )
            if detection["source_sha256"] != clip.source_sha256:
                raise ValueError("stage belongs to a different source")
            if not len(frames["t_s"]):
                raise ValueError("stage has no decoded rows")
            frozen_rows(clip, frames)
            calibration, _, calibrator = cached_real.cached_cameras(
                run, detection, frames, config.calibration
            )
            if calibrator["code_revision"] != detector["code_revision"]:
                raise ValueError("stages have different producers")
        except (ValueError, OSError, KeyError, TypeError) as error:
            raise StageError(f"{clip.clip_id}: {error}") from error
        rows.append(
            {
                "clip_id": clip.clip_id,
                "source_sha256": clip.source_sha256,
                "timeline_sha256": clip.timeline_sha256,
                "frame_count": clip.frame_count,
                "image_size": [clip.width, clip.height],
                "detection_status": detection["status"],
                "calibration_status": calibration.get("status"),
                "producer": {
                    "code_revision": detector["code_revision"],
                    "detection": detector,
                    "calibration": calibrator,
                    "accelerator": detection.get("accelerator"),
                },
            }
        )
    assert config is not None
    defaults = ReconstructionConfig()
    detect, default_detect = (
        value.detection.model_dump(mode="json") for value in (config, defaults)
    )
    changed = sorted(key for key in detect if detect[key] != default_detect[key])
    if set(changed) - set(OPERATIONAL) or config.calibration != defaults.calibration:
        raise StageError("stage thresholds differ from the reviewed defaults")
    write_json(output / "config.json", config.model_dump(mode="json"))
    files = {
        str(path.relative_to(output)): {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    for row in rows:
        prefix = f"clips/{row['clip_id']}/"
        row["artifacts"] = {
            name.removeprefix(prefix): entry
            for name, entry in files.items()
            if name.startswith(prefix)
        }
        if set(row["artifacts"]) != set(cached_real.UPSTREAM_FILES):
            raise StageError(f"{row['clip_id']}: unexpected files in the new root")
    package = package_identity()
    manifest = {
        "schema": cached_real.MANIFEST_SCHEMA,
        "kind": "common_stage_root",
        "original_inventory_sha256": inventory_sha256,
        "files": files,
        "clips": rows,
        "common_stages": {
            "declaration": STATEMENT,
            "frozen_utc": now(),
            "root": str(output.resolve()),
            "stage_source": str(stages),
            "inventory": str(Path(inventory_path).resolve()),
            "frame_count": sum(row["frame_count"] for row in rows),
            "producer_code_revisions": sorted(
                {row["producer"]["code_revision"] for row in rows}
            ),
            "detection_options_differing_from_defaults": {
                key: detect[key] for key in changed
            },
            "freezing_package": {
                "source_sha256": package["source_sha256"],
                "code_revision": package["code_revision"],
            },
            "freezing_scripts_sha256": {
                "common_stages.py": file_sha256(__file__),
                "cached_real.py": file_sha256(
                    Path(__file__).with_name("cached_real.py")
                ),
            },
            "stage_files_checked_against_released_list": listed is not None,
            "bindings": bound,
        },
    }
    for clip in clips:
        cached_real.verified_upstream(output, manifest, clip.clip_id)
    write_json(manifest_path, manifest)
    cached_real.checked_manifest(
        manifest_path, file_sha256(manifest_path), inventory_path
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    for name in ("inventory", "output", "executable", "detect-config", "lock"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--inventory-sha256", required=True)
    for name in ("base-seconds", "seconds-per-frame", "calibrate-seconds"):
        p.add_argument(f"--{name}", type=float, required=True)
    f = commands.add_parser("freeze")
    for name in ("inventory", "stages", "output", "artifact-manifest"):
        f.add_argument(f"--{name}", type=Path, required=True)
    f.add_argument("--inventory-sha256", required=True)
    f.add_argument(
        "--released",
        type=Path,
        help="JSON list of released files (path, bytes, sha256) every stage file must match",
    )
    f.add_argument("--released-sha256")
    f.add_argument(
        "--bind",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="evidence file whose path, size and SHA-256 the manifest records",
    )
    args = parser.parse_args(argv)
    if bool(getattr(args, "released", None)) != bool(
        getattr(args, "released_sha256", None)
    ):
        parser.error("--released and --released-sha256 go together")
    try:
        if args.command == "prepare":
            # A terminated supervisor stops its detector before the lock is freed.
            for sign in INTERRUPTIONS:
                signal.signal(sign, signal.default_int_handler)
            summary = prepare(
                args.inventory,
                args.inventory_sha256,
                args.output,
                args.executable,
                args.detect_config,
                args.lock,
                base_s=args.base_seconds,
                per_frame_s=args.seconds_per_frame,
                calibrate_s=args.calibrate_seconds,
            )
            print(
                json.dumps(
                    {key: value for key, value in summary.items() if key != "records"},
                    indent=2,
                )
            )
            return 3 if summary["blocker"] else 0
        manifest = freeze(
            args.inventory,
            args.inventory_sha256,
            args.stages,
            args.output,
            args.artifact_manifest,
            dict(item.split("=", 1) for item in args.bind),
            (args.released, args.released_sha256) if args.released else None,
        )
    except (StageError, ValueError, OSError, KeyError, TypeError) as error:
        print(f"Common stages unavailable: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact_manifest_sha256": file_sha256(args.artifact_manifest),
                "clips": len(manifest["clips"]),
                **{
                    key: manifest["common_stages"][key]
                    for key in ("frame_count", "producer_code_revisions")
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
