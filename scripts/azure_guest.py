"""Run a pinned command with guest-local generation and drain deadlines."""

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


class Refusal(Exception):
    """A manifest or source verification failure with a safe public reason."""


def deadline(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", value
    ):
        raise Refusal("invalid_deadline")
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        raise Refusal("invalid_deadline") from None


def command(value):
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(part, str) or not part or "\0" in part for part in value)
    ):
        raise Refusal("invalid_argv")
    return value


def git(cwd, *args):
    return (
        subprocess.check_output(
            ["git", "-C", str(cwd), *args], stderr=subprocess.DEVNULL, timeout=10
        )
        .decode()
        .strip()
    )


def validate(manifest):
    if not isinstance(manifest, dict):
        raise Refusal("invalid_manifest")
    try:
        run_id = str(uuid.UUID(manifest["run_id"]))
    except (KeyError, TypeError, ValueError, AttributeError):
        raise Refusal("invalid_run_id") from None
    cwd_value = manifest.get("cwd")
    if not isinstance(cwd_value, str) or not Path(cwd_value).is_absolute():
        raise Refusal("invalid_cwd")
    cwd = Path(cwd_value).resolve()
    commit = manifest.get("source_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise Refusal("invalid_source_commit")
    generate_until = deadline(manifest.get("generate_until"))
    drain_until = deadline(manifest.get("drain_until"))
    if generate_until <= time.time() or drain_until <= generate_until:
        raise Refusal("expired_or_unordered_deadlines")
    argv = command(manifest.get("argv"))
    drain_argv = None
    if "drain_argv" in manifest:
        drain_argv = command(manifest["drain_argv"])
    files = manifest.get("files")
    config = manifest.get("generator_config")
    if (
        not isinstance(files, dict)
        or "uv.lock" not in files
        or not isinstance(config, str)
        or config == "uv.lock"
        or config not in files
    ):
        raise Refusal("missing_pinned_files")
    if Path(git(cwd, "rev-parse", "--show-toplevel")).resolve() != cwd:
        raise Refusal("cwd_is_not_checkout_root")
    if git(cwd, "rev-parse", "HEAD").lower() != commit.lower():
        raise Refusal("source_commit_mismatch")
    if git(cwd, "status", "--porcelain", "--untracked-files=all"):
        raise Refusal("dirty_checkout")
    for name, expected in files.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", expected)
        ):
            raise Refusal("invalid_file_pin")
        path = (cwd / name).resolve()
        if not path.is_relative_to(cwd) or not path.is_file():
            raise Refusal("invalid_pinned_file")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected.lower():
            raise Refusal("file_hash_mismatch")
    if generate_until <= time.time():
        raise Refusal("expired_or_unordered_deadlines")
    return run_id, cwd, argv, drain_argv, generate_until, drain_until


def signal_group(process, number):
    try:
        os.killpg(process.pid, number)
        return True
    except ProcessLookupError:
        return False


def wait_until(process, until, stopped=None):
    while process.poll() is None:
        if stopped is not None and stopped[0] is not None:
            return "signal"
        remaining = until - time.time()
        if remaining <= 0:
            return "deadline"
        time.sleep(min(0.02, remaining))
    return "exit"


def run(manifest, stopped, result):
    run_id, cwd, argv, drain_argv, generate_until, drain_until = validate(manifest)
    result["run_id"] = run_id
    environment = os.environ.copy()
    environment.update(dict.fromkeys(THREAD_VARIABLES, "1"))
    environment.pop("DISPLAY", None)
    environment.pop("MUJOCO_GL", None)

    def start(argv):
        return subprocess.Popen(
            argv,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    process = None
    try:
        if stopped[0] is None:
            process = start(argv)
            reason = wait_until(process, generate_until, stopped)
            result["generation_deadline_reached"] = reason == "deadline"
            if reason != "exit":
                signal_group(process, signal.SIGTERM)
                if wait_until(process, drain_until) == "deadline":
                    result["drain_deadline_reached"] = True
            # A command can exit while its children still occupy the group.
            result["forced_kill"] = signal_group(process, signal.SIGKILL)
            result["generator_exit"] = process.wait()
            process = None
        if drain_argv is not None:
            if time.time() >= drain_until:
                result["drain_deadline_reached"] = True
            else:
                process = start(drain_argv)
                if wait_until(process, drain_until) == "deadline":
                    result["drain_deadline_reached"] = True
                result["forced_kill"] |= signal_group(process, signal.SIGKILL)
                result["drain_exit"] = process.wait()
                process = None
        result["stop_signal"] = stopped[0]
        if result["drain_deadline_reached"]:
            result["status"] = "drain_deadline"
        elif stopped[0] is not None:
            result["status"] = "stopped"
        elif result["generation_deadline_reached"]:
            result["status"] = "generation_deadline"
        elif result["generator_exit"] or result["drain_exit"]:
            result["status"] = "failed"
        else:
            result["status"] = "completed"
    finally:
        if process is not None:
            signal_group(process, signal.SIGKILL)
            process.wait()


def write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=".guest-result-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    stopped = [None]

    def stop(number, _frame):
        stopped[0] = number

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    result = {
        "run_id": None,
        "status": "refused",
        "error": None,
        "generator_exit": None,
        "drain_exit": None,
        "generation_deadline_reached": False,
        "drain_deadline_reached": False,
        "forced_kill": False,
        "stop_signal": None,
    }
    try:
        with args.manifest.open() as stream:
            manifest = json.load(stream)
        run(manifest, stopped, result)
    except Refusal as error:
        result["error"] = str(error)
    except (ValueError, TypeError):
        result["error"] = "invalid_manifest"
    except (OSError, subprocess.SubprocessError):
        result["error"] = "command_or_io_error"
        if result["run_id"] is not None:
            result["status"] = "failed"
    write_result(args.result, result)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
