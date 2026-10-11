"""Local shell tests for durable Slurm submission and reconciliation."""

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "throughput-wait.sh"
MOCK_SLURM = """
import json, os, sys, time
from pathlib import Path

command = Path(sys.argv[0]).name
arguments = sys.argv[1:]
line = json.dumps({'command': command, 'arguments': arguments}) + '\\n'
descriptor = os.open(os.environ['MOCK_CALLS'], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
try:
    os.write(descriptor, line.encode())
finally:
    os.close(descriptor)
if command == 'sbatch':
    print(os.environ.get('MOCK_SUBMISSION_ID', '123'), flush=True)
    if os.environ.get('MOCK_BLOCK_SUBMISSION'):
        release = Path(os.environ['MOCK_RELEASE'])
        deadline = time.monotonic() + 10
        while not release.exists():
            if time.monotonic() > deadline:
                raise SystemExit('mock submission release timed out')
            time.sleep(0.01)
elif command == 'squeue':
    pass
elif command == 'sacct':
    job = arguments[arguments.index('-j') + 1]
    if '-o' in arguments and arguments[arguments.index('-o') + 1] == 'State':
        print('COMPLETED')
    elif '-P' in arguments:
        print(f'{job}|COMPLETED|0:0|60|2026-10-06T09:00:00|2026-10-06T09:02:03|2026-10-06T09:03:03|node-test|4|32Gn')
    else:
        print(f'{job} COMPLETED 0:0 00:01:00 node-test')
else:
    raise SystemExit(f'unexpected mock command: {command}')
"""


@pytest.fixture
def waiter(tmp_path):
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    for command in ("sbatch", "squeue", "sacct"):
        executable = mock_bin / command
        executable.write_text(f"#!{sys.executable}\n{MOCK_SLURM}")
        executable.chmod(0o755)
    # The real script calls python3 for its stdlib accounting parser. Use the
    # test interpreter regardless of the host's default Python installation.
    python = mock_bin / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        f"os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
    )
    python.chmod(0o755)
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    output = tmp_path / "output"
    record = tmp_path / "jobs" / "qualification.id"
    calls = tmp_path / "calls.jsonl"
    environment = {
        **os.environ,
        "PATH": f"{mock_bin}{os.pathsep}{os.environ['PATH']}",
        "THROUGHPUT_ROOT": str(root),
        "THROUGHPUT_OUTPUT": str(output),
        "THROUGHPUT_JOB_RECORD": str(record),
        "THROUGHPUT_MODE": "qualify",
        "MOCK_CALLS": str(calls),
        "MOCK_RELEASE": str(tmp_path / "release"),
    }
    return environment, record, output, calls


def recorded_calls(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def run_waiter(environment):
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def assert_accounting(output):
    report = json.loads((output / "slurm-accounting.json").read_text())
    assert report == {
        "basis": "slurm",
        "job_id": "123",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "compute_s": 60.0,
        "queue_s": 123.0,
        "submit": "2026-10-06T09:00:00",
        "start": "2026-10-06T09:02:03",
        "end": "2026-10-06T09:03:03",
        "node": "node-test",
        "cpus": "4",
        "requested_memory": "32Gn",
    }


def test_simultaneous_waiters_submit_same_record_once(waiter):
    environment, record, output, calls = waiter
    environment["MOCK_BLOCK_SUBMISSION"] = "1"
    processes = []
    try:
        first = subprocess.Popen(
            ["bash", str(SCRIPT)],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(first)
        assert first.stdout is not None
        # The mock sbatch publishes its ID, then remains blocked like --wait.
        assert first.stdout.readline() == "123\n"
        second = subprocess.Popen(
            ["bash", str(SCRIPT)],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(second)
        with pytest.raises(subprocess.TimeoutExpired):
            second.communicate(timeout=0.2)
        assert [call["command"] for call in recorded_calls(calls)] == ["sbatch"]
        Path(environment["MOCK_RELEASE"]).touch()
        for process in processes:
            _, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, stderr
    finally:
        Path(environment["MOCK_RELEASE"]).touch()
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)
    invocations = recorded_calls(calls)
    submitted = [call for call in invocations if call["command"] == "sbatch"]
    assert len(submitted) == 1
    assert "--wait" in submitted[0]["arguments"]
    assert "--parsable" in submitted[0]["arguments"]
    assert record.read_text() == "123\n"
    assert sum(call["command"] == "squeue" for call in invocations) == 2
    assert_accounting(output)


def test_existing_empty_record_fails_closed_without_submission(waiter):
    environment, record, output, calls = waiter
    record.parent.mkdir()
    record.touch()
    result = run_waiter(environment)
    assert result.returncode == 2
    assert "uncertain submission" in result.stderr
    assert record.read_bytes() == b""
    assert not calls.exists()
    assert not (output / "slurm-accounting.json").exists()


@pytest.mark.parametrize("recorded", ["123\n456\n", "123;alpha-1.example\n456\n"])
def test_multiline_record_fails_closed_without_slurm_calls(waiter, recorded):
    environment, record, output, calls = waiter
    record.parent.mkdir()
    record.write_text(recorded)
    result = run_waiter(environment)
    assert result.returncode == 2
    assert "invalid recorded Slurm ID" in result.stderr
    assert record.read_text() == recorded
    assert not calls.exists()
    assert not (output / "slurm-accounting.json").exists()


@pytest.mark.parametrize("cluster", [None, "alpha-1.example"])
def test_recorded_id_reconciles_without_resubmission_and_preserves_cluster(
    waiter, cluster
):
    environment, record, output, calls = waiter
    record.parent.mkdir()
    recorded = "123" if cluster is None else f"123;{cluster}"
    record.write_text(recorded + "\n")
    result = run_waiter(environment)
    assert result.returncode == 0, result.stderr
    invocations = recorded_calls(calls)
    assert [call["command"] for call in invocations] == [
        "squeue",
        "sacct",
        "sacct",
        "sacct",
    ]
    for call in invocations:
        arguments = call["arguments"]
        assert arguments[arguments.index("-j") + 1] == "123"
        if cluster is None:
            assert not any(value.startswith("--clusters") for value in arguments)
        else:
            assert f"--clusters={cluster}" in arguments
    assert record.read_text() == recorded + "\n"
    assert_accounting(output)


def test_successful_trial_submission_writes_trial_accounting(waiter):
    environment, record, output, calls = waiter
    environment.update(
        THROUGHPUT_MODE="trial",
        THROUGHPUT_TRIAL="000001",
        MOCK_SUBMISSION_ID="123;alpha-1.example",
    )
    result = run_waiter(environment)
    assert result.returncode == 0, result.stderr
    invocations = recorded_calls(calls)
    submission = invocations[0]
    assert submission["command"] == "sbatch"
    assert submission["arguments"][-2:] == ["trial", "000001"]
    assert record.read_text() == "123;alpha-1.example\n"
    for call in invocations[1:]:
        assert "--clusters=alpha-1.example" in call["arguments"]
    assert not (output / "slurm-accounting.json").exists()
    assert_accounting(output / "trials" / "000001")


@pytest.mark.parametrize(
    "mode,trial",
    [
        ("trial", ""),
        ("trial", "../escape"),
        ("trial", "candidate-001"),
        ("other", ""),
        ("qualify", "000001"),
    ],
)
def test_invalid_mode_or_trial_is_rejected_before_record_or_submission(
    waiter, mode, trial
):
    environment, record, output, calls = waiter
    environment.update(THROUGHPUT_MODE=mode, THROUGHPUT_TRIAL=trial)
    result = run_waiter(environment)
    assert result.returncode == 2
    assert not record.exists()
    assert not calls.exists()
    assert not output.exists()


def test_failed_qualification_syncs_evidence_without_source_cache(tmp_path):
    if shutil.which("rsync") is None:
        pytest.skip("local rsync is required for the evidence sync test")
    root, durable = tmp_path / "source", tmp_path / "durable"
    root.mkdir()
    interpreter = tmp_path / "qualification"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import sys\nfrom pathlib import Path\n"
        "output=Path(sys.argv[sys.argv.index('--output')+1])\n"
        "(output/'source-cache').mkdir(parents=True)\n"
        "(output/'source-cache'/'partial').write_bytes(b'owned-cache')\n"
        "(output/'failure.json').write_text('{\"passed\":false}')\n"
        "raise SystemExit(2)\n"
    )
    interpreter.chmod(0o755)
    job = "test-" + uuid.uuid4().hex
    user = os.environ.get("USER", "tester")
    owned = Path("/tmp") / user / f"throughput-{job}"
    assert not owned.exists()
    try:
        result = subprocess.run(
            ["bash", str(SCRIPT.with_name("throughput.slurm")), "qualify"],
            env={
                **os.environ,
                "USER": user,
                "SLURM_JOB_ID": job,
                "TENNIS3D_ROOT": str(root),
                "TENNIS3D_THROUGHPUT_OUTPUT": str(durable),
                "TENNIS3D_PYTHON": str(interpreter),
            },
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert result.returncode == 2, result.stderr
        assert (durable / "failure.json").read_text() == '{"passed":false}'
        assert not (durable / "source-cache").exists()
        assert (owned / "run/source-cache/partial").read_bytes() == b"owned-cache"
    finally:
        if owned.exists():
            shutil.rmtree(owned)
