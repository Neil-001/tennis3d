"""Exercise the guest runner with tiny local subprocesses and throwaway Git repos."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "azure_guest.py"


def timestamp(seconds):
    return (
        datetime.fromtimestamp(time.time() + seconds, UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )


@pytest.fixture
def checkout(tmp_path):
    cwd = tmp_path / "source"
    cwd.mkdir()
    (cwd / "uv.lock").write_text("pinned dependencies\n")
    (cwd / "generator.yaml").write_text("pinned config\n")
    (cwd / ".gitignore").write_text("state/\n")
    for argv in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "pin",
        ],
    ):
        subprocess.run(["git", "-C", str(cwd), *argv], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(cwd), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "run_id": str(uuid.uuid4()),
        "cwd": str(cwd),
        "source_commit": commit,
        "generator_config": "generator.yaml",
        "files": {
            name: hashlib.sha256((cwd / name).read_bytes()).hexdigest()
            for name in ("uv.lock", "generator.yaml")
        },
        "argv": [sys.executable, "-c", "pass"],
        "generate_until": timestamp(10),
        "drain_until": timestamp(12),
    }
    return cwd, manifest, tmp_path


def launch(checkout, **kwargs):
    _cwd, manifest, tmp_path = checkout
    manifest_path = tmp_path / "manifest.json"
    result_path = tmp_path / "result.json"
    manifest_path.write_text(json.dumps(manifest))
    process = subprocess.Popen(
        [
            sys.executable,
            str(RUNNER),
            "--manifest",
            str(manifest_path),
            "--result",
            str(result_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **kwargs,
    )
    return process, result_path


def finish(process, result_path):
    stdout, stderr = process.communicate(timeout=8)
    assert not stdout
    assert not stderr
    return json.loads(result_path.read_text())


def wait_file(path):
    until = time.monotonic() + 4
    while not path.exists():
        assert time.monotonic() < until, f"child did not create {path.name}"
        time.sleep(0.01)


def alive(pid):
    # A killed orphan can remain a zombie until the container's PID 1 reaps it.
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


def test_exact_argv_environment_and_ignored_state(checkout):
    cwd, manifest, _ = checkout
    (cwd / "state").mkdir()
    marker = cwd / "state" / "observed.json"
    manifest["argv"] = [
        sys.executable,
        "-c",
        (
            "import json, os, sys; from pathlib import Path; "
            "names = ('OMP_NUM_THREADS OMP_THREAD_LIMIT OPENBLAS_NUM_THREADS MKL_NUM_THREADS '"
            "'NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS BLIS_NUM_THREADS DISPLAY MUJOCO_GL').split(); "
            "env = {k: os.environ[k] for k in names if k in os.environ}; "
            "Path('state/observed.json').write_text(json.dumps([sys.argv[1:], env]))"
        ),
        "literal; $(touch should-not-exist)",
    ]
    env = os.environ.copy()
    env.update(
        DISPLAY=":999", MUJOCO_GL="egl", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4"
    )
    process, path = launch(checkout, env=env)
    result = finish(process, path)
    arguments, observed = json.loads(marker.read_text())
    assert arguments == manifest["argv"][3:]
    for name in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        assert observed[name] == "1"
    assert "DISPLAY" not in observed
    assert "MUJOCO_GL" not in observed
    assert result["status"] == "completed"
    assert result["run_id"] == manifest["run_id"]
    assert process.returncode == 0
    assert path.stat().st_mode & 0o777 == 0o600
    assert not list(path.parent.glob(".guest-result-*"))


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("source_commit", "0" * 40, "source_commit_mismatch"),
        ("source_commit", "bad", "invalid_source_commit"),
        ("run_id", "bad", "invalid_run_id"),
        ("cwd", "relative", "invalid_cwd"),
        ("argv", [], "invalid_argv"),
        ("argv", [""], "invalid_argv"),
        ("drain_argv", "echo", "invalid_argv"),
        ("generator_config", "missing.yaml", "missing_pinned_files"),
        ("generate_until", "2000-01-01T00:00:00Z", "expired_or_unordered_deadlines"),
        ("generate_until", "2100-01-01T00:00:00+00:00", "invalid_deadline"),
        ("generate_until", "2100-99-01T00:00:00Z", "invalid_deadline"),
        ("drain_until", "2000-01-01T00:00:00Z", "expired_or_unordered_deadlines"),
    ],
)
def test_refuses_invalid_manifest(checkout, field, value, error):
    checkout[1][field] = value
    process, path = launch(checkout)
    result = finish(process, path)
    assert result["status"] == "refused"
    assert result["error"] == error
    assert result["generator_exit"] is None
    assert process.returncode == 1


@pytest.mark.parametrize("name", ["uv.lock", "untracked.txt"])
def test_refuses_dirty_source(checkout, name):
    (checkout[0] / name).write_text("changed\n")
    result = finish(*launch(checkout))
    assert result["error"] == "dirty_checkout"


def test_refuses_hash_mismatch(checkout):
    checkout[1]["files"]["uv.lock"] = "0" * 64
    assert finish(*launch(checkout))["error"] == "file_hash_mismatch"


def test_refuses_missing_lock(checkout):
    del checkout[1]["files"]["uv.lock"]
    assert finish(*launch(checkout))["error"] == "missing_pinned_files"


def test_refuses_pin_outside_checkout(checkout):
    checkout[1]["files"]["../secret"] = "0" * 64
    assert finish(*launch(checkout))["error"] == "invalid_file_pin"


def test_nonzero_exit_is_recorded_without_child_logs(checkout):
    checkout[1]["argv"] = [sys.executable, "-c", "print('secret'); raise SystemExit(7)"]
    result = finish(*launch(checkout))
    assert result["status"] == "failed"
    assert result["generator_exit"] == 7
    assert "secret" not in json.dumps(result)


def test_guest_deadlines_survive_controller_exit_and_kill_process_group(checkout):
    cwd, manifest, tmp_path = checkout
    (cwd / "state").mkdir()
    manifest["generate_until"] = timestamp(0.6)
    manifest["drain_until"] = timestamp(1.0)
    manifest["argv"] = [
        sys.executable,
        "-c",
        (
            "import os, signal, time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "pid=os.fork(); "
            "Path('state/child' if pid == 0 else 'state/parent').write_text(str(os.getpid())); "
            "time.sleep(30)"
        ),
    ]
    manifest_path = tmp_path / "manifest.json"
    result_path = tmp_path / "result.json"
    manifest_path.write_text(json.dumps(manifest))
    # The controller launches a detached guest and exits immediately.
    controller = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import subprocess,sys; subprocess.Popen(sys.argv[1:], start_new_session=True, "
                "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)"
            ),
            sys.executable,
            str(RUNNER),
            "--manifest",
            str(manifest_path),
            "--result",
            str(result_path),
        ],
        timeout=3,
        check=True,
    )
    assert controller.returncode == 0
    wait_file(cwd / "state" / "child")
    wait_file(result_path)
    result = json.loads(result_path.read_text())
    assert result["generation_deadline_reached"]
    assert result["drain_deadline_reached"]
    assert result["forced_kill"]
    assert result["generator_exit"] == -signal.SIGKILL
    for name in ("child", "parent"):
        assert not alive(int((cwd / "state" / name).read_text()))


@pytest.mark.parametrize("trigger", ["wrapper_sigterm", "generation_deadline"])
def test_graceful_generation_cleanup_and_drain(checkout, trigger):
    cwd, manifest, _ = checkout
    (cwd / "state").mkdir()
    manifest["argv"] = [
        sys.executable,
        "-c",
        (
            "import signal,time; from pathlib import Path\n"
            "def stop(*_):\n"
            " Path('state/cleanup').write_text('preserved')\n"
            " raise SystemExit(0)\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            "Path('state/ready').touch()\n"
            "time.sleep(30)\n"
        ),
    ]
    manifest["drain_argv"] = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('state/drained').touch()",
    ]
    if trigger == "generation_deadline":
        manifest["generate_until"] = timestamp(0.5)
        manifest["drain_until"] = timestamp(2)
    process, path = launch(checkout)
    wait_file(cwd / "state" / "ready")
    if trigger == "wrapper_sigterm":
        process.send_signal(signal.SIGTERM)
    result = finish(process, path)
    if trigger == "wrapper_sigterm":
        assert result["status"] == "stopped"
        assert result["stop_signal"] == signal.SIGTERM
    else:
        assert result["status"] == "generation_deadline"
        assert result["generation_deadline_reached"]
        assert result["stop_signal"] is None
    assert result["generator_exit"] == 0
    assert result["drain_exit"] == 0
    assert (cwd / "state" / "cleanup").read_text() == "preserved"
    assert (cwd / "state" / "drained").exists()


def test_early_exit_kills_lingering_children_before_drain(checkout):
    cwd, manifest, _ = checkout
    (cwd / "state").mkdir()
    manifest["argv"] = [
        sys.executable,
        "-c",
        (
            "import os, signal, time; from pathlib import Path\n"
            "pid=os.fork()\n"
            "if pid == 0:\n"
            " signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            " Path('state/child').write_text(str(os.getpid()))\n"
            " time.sleep(30)\n"
            "else:\n"
            " while not Path('state/child').exists(): time.sleep(0.01)\n"
        ),
    ]
    manifest["drain_argv"] = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import time; "
            "pid=Path('state/child').read_text(); time.sleep(0.05); "
            "p=Path('/proc') / pid / 'stat'; "
            "assert not p.exists() or p.read_text().split()[2] == 'Z'; "
            "Path('state/drained').touch()"
        ),
    ]
    result = finish(*launch(checkout))
    assert result["status"] == "completed"
    assert result["forced_kill"]
    assert result["drain_exit"] == 0
    assert (cwd / "state" / "drained").exists()


def test_drain_has_its_own_deadline(checkout):
    checkout[1]["generate_until"] = timestamp(0.4)
    checkout[1]["drain_until"] = timestamp(0.7)
    checkout[1]["drain_argv"] = [sys.executable, "-c", "import time; time.sleep(30)"]
    result = finish(*launch(checkout))
    assert result["status"] == "drain_deadline"
    assert result["generator_exit"] == 0
    assert result["drain_exit"] == -signal.SIGKILL
