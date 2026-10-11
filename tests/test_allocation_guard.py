"""Fake-Slurm tests for the fail-closed GPU allocation budget guard."""

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "allocation_guard.py"
FAKE_SLURM = """
import json, os, sys, time
from pathlib import Path

command = Path(sys.argv[0]).name
arguments = sys.argv[1:]
root = Path(os.environ['FAKE_SLURM'])
state = json.loads((root / 'state.json').read_text())
with open(root / 'calls.jsonl', 'a') as stream:
    stream.write(json.dumps([command, *arguments]) + '\\n')
if command == 'sbatch':
    plan = state['sbatch']
    latest = sorted((root / 'ledger').glob('intent-*.json'))[-1]
    if json.loads(latest.read_text())['status'] != 'intent':
        raise SystemExit('the intent was not durable before sbatch ran')
    while plan.get('block') and not (root / 'release').exists():
        time.sleep(0.01)
    sys.stderr.write(plan.get('stderr', ''))
    sys.stdout.write(plan.get('stdout', ''))
    raise SystemExit(plan.get('exit', 0))
if command == 'squeue' and '-u' in arguments:
    if arguments[-2] != '-O' or state.get('queue_exit'):
        raise SystemExit(1)
    print('\\n'.join(state.get('queue', [])))
    raise SystemExit(0)
job = arguments[-1] if command == 'scancel' else arguments[arguments.index('-j') + 1]
if command == 'sacct':
    if state.get('sacct_exit'):
        raise SystemExit(state['sacct_exit'])
    pending = [f'{job}|PENDING|Unknown|Unknown|0||None assigned']
    print('\\n'.join(state.get('sacct', {}).get(job, pending)))
elif command == 'squeue':
    print(state.get('squeue', {}).get(job, ''))
elif command == 'scancel':
    queue = state.setdefault('squeue', {})
    if '--signal=USR1' not in arguments:
        queue.pop(job, None)
    elif job in state.get('after_usr1', {}):
        queue[job] = state['after_usr1'][job]
    (root / 'state.json').write_text(json.dumps(state))
else:
    raise SystemExit(f'unexpected fake command: {command}')
"""


def row(job, state, elapsed, tres="billing=8,cpu=8,gres/gpu:a100=1,gres/gpu=1,mem=32G"):
    return f"{job}|{state}|2026-10-09T01:00:00|Unknown|{elapsed}|{tres}|node-a"


class Slurm:
    def __init__(self, root, monkeypatch):
        self.root, self.ledger = root, root / "ledger"
        binaries = root / "bin"
        binaries.mkdir()
        for command in ("sbatch", "sacct", "squeue", "scancel"):
            executable = binaries / command
            executable.write_text(f"#!{sys.executable}\n{FAKE_SLURM}")
            executable.chmod(0o755)
        (root / "calls.jsonl").touch()
        self.script = str(root / "job.sh")
        Path(self.script).write_text("#!/bin/sh\n#SBATCH --time=24:00:00 --requeue\n")
        self.state = {}
        monkeypatch.setenv("PATH", f"{binaries}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setenv("FAKE_SLURM", str(root))

    @property
    def state(self):
        return json.loads((self.root / "state.json").read_text())

    @state.setter
    def state(self, value):
        (self.root / "state.json").write_text(json.dumps(value))

    def update(self, **changes):
        self.state = self.state | changes

    def calls(self, command=None):
        lines = (self.root / "calls.jsonl").read_text().splitlines()
        return [c for c in map(json.loads, lines) if command in (None, c[0])]

    def guard(self, command, *arguments):
        argv = [sys.executable, str(SCRIPT), command, "--ledger", str(self.ledger)]
        return subprocess.run(
            [*argv, *arguments], capture_output=True, text=True, check=False
        )

    def submit_argv(self, gpus=1, limit=100, cap=1000, extra=()):
        options = ["--cap", cap, "--label", "run", "--gpus", gpus, "--time-limit"]
        options += [limit, "--user", "tester", *extra, "--", "sbatch"]
        options += [f"--gpus=a100:{gpus}"] * bool(gpus)
        options += [f"--time={limit // 60}:{limit % 60:02d}", self.script]
        return [str(option) for option in options]

    def submit(self, job=None, **options):
        if job is not None:
            self.update(sbatch={"stdout": f"{job}\n"})
        return self.guard("submit", *self.submit_argv(**options))

    def record(self, job, gpus=1, status="submitted"):
        self.ledger.mkdir(mode=0o700, exist_ok=True)
        name = f"{len(self.intents()) + 1:04d}"
        value = {"status": status, "label": f"job-{job}", "gpus": gpus}
        value["created_at"] = "2026-10-09T05:29:33+00:00"
        if status == "submitted":
            value["job_id"] = str(job)
        (self.ledger / f"intent-{name}.json").write_text(json.dumps(value))

    def intents(self):
        paths = sorted(self.ledger.glob("intent-*.json"))
        return [json.loads(path.read_text()) for path in paths]


@pytest.fixture
def slurm(tmp_path, monkeypatch):
    return Slurm(tmp_path, monkeypatch)


@pytest.fixture
def guard(slurm, monkeypatch):
    spec = importlib.util.spec_from_file_location("allocation_guard", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    now, sleeps = [0.0], []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(module, "sleep", sleep)
    monkeypatch.setattr(module, "monotonic", lambda: now[0])
    monkeypatch.setattr(module, "sleeps", sleeps, raising=False)
    return module


def watch(guard, slurm, *extra):
    options = ["--cap", "1000", "--margin", "100", "--grace", "30", *extra]
    return guard.main(["watch", "--ledger", str(slurm.ledger), *options])


def test_submit_writes_intent_before_sbatch_and_records_job_once(slurm):
    slurm.update(sbatch={"stdout": "cluster notice\n101;cluster\n"})
    result = slurm.guard("submit", *slurm.submit_argv())
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert slurm.intents() == [record]
    assert (record["status"], record["job_id"]) == ("submitted", "101")
    assert record["argv"] == [
        "sbatch",
        "--parsable",
        "--gpus=a100:1",
        "--time=1:40",
        slurm.script,
    ]
    assert record["charged_before"] == 0 and record["queue_at_intent"] == []
    assert record["cwd"] == os.getcwd() and record["created_at"].endswith("+00:00")
    assert slurm.calls("sbatch") == [record["argv"]]
    assert (slurm.ledger.stat().st_mode & 0o777) == 0o700
    ledger = json.loads(slurm.guard("status", "--cap", "1000").stdout)
    assert [job["job_id"] for job in ledger["jobs"]] == ["101"]

    bad = slurm.guard("submit", *slurm.submit_argv()[:-2], "srun", "job.sh")
    assert bad.returncode != 0 and len(slurm.intents()) == 1


def test_charges_sum_every_allocation_of_every_job(slurm):
    for job, gpus in ((101, 1), (102, 2), (103, 4), (104, 0), (105, 1)):
        slurm.record(job, gpus)
    slurm.record(0, status="not_submitted")
    two, four = "cpu=8,gres/gpu=2", "cpu=8,gres/gpu:a100=4,gres/gpu=4"
    slurm.update(
        sacct={
            "101": [row(101, "REQUEUED", 100), row(101, "NODE_FAIL", 200)]
            + [row(101, "COMPLETED", 300)],
            "102": [row(102, "CANCELLED by 1000", 50, two)],
            "103": [row(103, "FAILED", 10, four), row(103, "PENDING", 0, "")],
            "104": [row(104, "COMPLETED", 999, "billing=8,cpu=8,mem=32G")],
            "105": [row(105, "RUNNING", 30)],
        },
        squeue={"105": "RUNNING"},
    )
    result = slurm.guard("status", "--cap", "1000")
    assert result.returncode == 0, result.stderr
    ledger = json.loads(result.stdout)
    assert ledger == json.loads((slurm.ledger / "ledger.json").read_text())
    assert [job["gpu_seconds"] for job in ledger["jobs"]] == [600, 100, 40, 0, 30]
    assert ledger["charged_gpu_seconds"] == 770
    assert ledger["remaining_gpu_seconds"] == 230
    assert ledger["unresolved_intents"] == []
    assert ledger["jobs"][4]["active_state"] == "RUNNING"
    assert ledger["jobs"][1]["allocations"] == [
        {
            "job": "102",
            "state": "CANCELLED by 1000",
            "start": "2026-10-09T01:00:00",
            "end": "Unknown",
            "elapsed": 50,
            "gpus": 2,
            "node": "node-a",
            "gpu_seconds": 100,
        }
    ]
    call = slurm.calls("sacct")[0]
    assert call[1:5] == ["-X", "-n", "-P", "-D"] and call[7:10] == ["-j", "101", "-o"]


def test_submit_refuses_a_request_that_could_exceed_the_cap(slurm):
    slurm.record(101)
    slurm.update(sacct={"101": [row(101, "FAILED", 300)]})
    refused = slurm.submit(job=102, gpus=2, limit=351)
    assert refused.returncode != 0 and "exceeds cap" in refused.stderr
    assert refused.stdout == "" and len(slurm.intents()) == 1
    assert not slurm.calls("sbatch")
    allowed = slurm.submit(job=102, gpus=2, limit=350)
    assert allowed.returncode == 0, allowed.stderr
    assert json.loads(allowed.stdout)["charged_before"] == 300


def test_one_gpu_job_at_a_time_except_cpu_only_and_allow_active(slurm):
    slurm.record(101)
    slurm.update(squeue={"101": "PENDING"})
    refused = slurm.submit(job=102)
    assert refused.returncode != 0 and "still active" in refused.stderr
    assert len(slurm.intents()) == 1 and not slurm.calls("sbatch")
    assert slurm.submit(job=102, gpus=0).returncode == 0
    slurm.update(
        squeue={"101": "PENDING", "102": "RUNNING"},
        sacct={"102": [row(102, "RUNNING", 5, "billing=8,cpu=8")]},
    )
    assert slurm.submit(job=103).returncode != 0
    assert slurm.submit(job=103, extra=["--allow-active"]).returncode == 0
    slurm.update(squeue={"102": "RUNNING"})
    assert slurm.submit(job=104).returncode == 0  # only the CPU-only job is active
    assert [r.get("job_id") for r in slurm.intents()] == ["101", "102", "103", "104"]


def test_failed_sbatch_is_not_submitted_and_does_not_block(slurm):
    slurm.update(sbatch={"exit": 1, "stderr": "sbatch: error: invalid partition\n"})
    result = slurm.guard("submit", *slurm.submit_argv())
    record = json.loads(result.stdout)
    assert result.returncode != 0 and record["status"] == "not_submitted"
    assert record["sbatch_exit"] == 1 and "invalid partition" in record["stderr_tail"]
    assert "job_id" not in record
    rewrite = ["--intent", "1", "--not-submitted", "--evidence", "again"]
    assert slurm.guard("reconcile", *rewrite).returncode != 0
    assert slurm.intents() == [record]
    assert slurm.submit(job=102).returncode == 0


def test_uncertain_submission_blocks_until_reconciled(slurm):
    slurm.record(101)
    slurm.update(sbatch={"stdout": "Submitted batch job maybe\n"})
    result = slurm.guard("submit", *slurm.submit_argv())
    assert result.returncode != 0
    assert json.loads(result.stdout)["status"] == "uncertain"
    refused = slurm.submit(job=103)
    assert refused.returncode != 0 and "unreconciled" in refused.stderr
    assert len(slurm.intents()) == 2 and len(slurm.calls("sbatch")) == 1
    assert json.loads(slurm.guard("status", "--cap", "9").stdout)[
        "unresolved_intents"
    ] == ["0002"]

    def reconcile(*arguments):
        return slurm.guard("reconcile", *arguments, "--evidence", "sacct by name")

    assert reconcile("--intent", "2", "--job-id", "101").returncode != 0
    assert reconcile("--intent", "2", "--job-id", "1x").returncode != 0
    assert slurm.intents()[1]["status"] == "uncertain"
    resolved = reconcile("--intent", "0002", "--job-id", "102")
    assert resolved.returncode == 0, resolved.stderr
    record = slurm.intents()[1]
    assert (record["status"], record["job_id"]) == ("submitted", "102")
    assert record["evidence"] == "sacct by name" and record["reconciled_at"]
    assert reconcile("--intent", "2", "--job-id", "104").returncode != 0
    assert reconcile("--intent", "1", "--not-submitted").returncode != 0
    assert slurm.intents()[1] == record
    assert slurm.submit(job=103).returncode == 0

    slurm.record(0, status="intent")  # a guard killed between intent and outcome
    assert slurm.submit(job=105).returncode != 0
    assert reconcile("--intent", "4", "--not-submitted").returncode == 0
    assert slurm.intents()[3]["status"] == "not_submitted"
    assert slurm.submit(job=105).returncode == 0


def test_row_without_gpu_tres_is_charged_at_declared_gpus(slurm):
    slurm.record(101, gpus=2)
    slurm.update(sacct={"101": [row(101, "RUNNING", 40, ""), row(101, "FAILED", 5)]})
    ledger = json.loads(slurm.guard("status", "--cap", "1000").stdout)
    assert [a["gpus"] for a in ledger["jobs"][0]["allocations"]] == [2, 1]
    assert ledger["charged_gpu_seconds"] == 85


@pytest.mark.parametrize(
    "broken",
    [
        {"sacct_exit": 1},
        {"sacct": {"101": ["101|RUNNING|only|four"]}},
        {"sacct": {"101": [row(101, "RUNNING", "12:00")]}},
        {"sacct": {"101": [row(999, "RUNNING", 5)]}},
        {"sacct": {}, "squeue": {"101": "RUNNING"}},
    ],
)
def test_accounting_failure_keeps_previous_ledger(slurm, broken):
    slurm.record(101)
    slurm.update(sacct={"101": [row(101, "RUNNING", 60)]})
    assert slurm.guard("status", "--cap", "1000").returncode == 0
    before = (slurm.ledger / "ledger.json").read_bytes()
    slurm.update(**broken)
    assert slurm.guard("status", "--cap", "1000").returncode != 0
    assert slurm.guard("watch", "--cap", "1000", "--margin", "0", "--once").returncode
    assert (slurm.ledger / "ledger.json").read_bytes() == before
    assert sorted(path.name for path in slurm.ledger.iterdir()) == [
        "guard.lock",
        "intent-0001.json",
        "ledger.json",
    ]
    assert slurm.submit(job=102, extra=["--allow-active"]).returncode != 0
    assert not slurm.calls("sbatch") and not slurm.calls("scancel")


def test_watch_under_the_cap_takes_no_action(slurm):
    slurm.record(101)
    slurm.update(sacct={"101": [row(101, "RUNNING", 899)]}, squeue={"101": "RUNNING"})
    options = ["--cap", "1000", "--margin", "100", "--once"]
    assert slurm.guard("watch", *options).returncode == 0
    ledger = json.loads((slurm.ledger / "ledger.json").read_text())
    assert ledger["charged_gpu_seconds"] == 899
    assert not slurm.calls("scancel") and not slurm.calls("sbatch")
    assert not (slurm.ledger / "cap-stop.json").exists()


def test_watch_at_margin_stops_once_then_cancels_reappearing_jobs(slurm, guard):
    for job in (101, 102, 103):
        slurm.record(job)
    slurm.update(
        sacct={"101": [row(101, "RUNNING", 900)], "103": [row(103, "PENDING", 0)]},
        squeue={"101": "RUNNING", "103": "PENDING"},
        after_usr1={"101": "PENDING"},  # the wrapper checkpoints and requeues
    )
    assert watch(guard, slurm, "--once") == 0
    usr1 = ["scancel", "--batch", "--signal=USR1", "101"]
    assert slurm.calls("scancel") == [["scancel", "103"], usr1, ["scancel", "101"]]
    stop = json.loads((slurm.ledger / "cap-stop.json").read_text())
    assert (stop["reason"], stop["charged_gpu_seconds"]) == ("cap_margin", 900)
    assert [action["argv"] for action in stop["actions"]] == slurm.calls("scancel")
    assert stop["final"]["101"] == {"squeue": "", "sacct": ["RUNNING"]}
    assert guard.sleeps == [] and slurm.state["squeue"] == {}

    assert watch(guard, slurm, "--once") == 0
    assert len(slurm.calls("scancel")) == 3  # nothing is active: nothing to do
    slurm.update(squeue={"101": "RUNNING"})  # a requeue the cancel raced with
    assert watch(guard, slurm, "--once") == 0
    assert slurm.calls("scancel")[3:] == [["scancel", "101"]]
    stop = json.loads((slurm.ledger / "cap-stop.json").read_text())
    assert len(stop["actions"]) == 4 and stop["reason"] == "cap_margin"
    ledger = json.loads((slurm.ledger / "ledger.json").read_text())
    assert ledger["charged_gpu_seconds"] == 900
    assert slurm.submit(job=104, gpus=0).returncode != 0  # stopped ledgers stay shut
    assert not slurm.calls("sbatch")


def test_watch_waits_the_grace_period_before_cancelling(slurm, guard):
    slurm.record(101)
    slurm.update(sacct={"101": [row(101, "RUNNING", 950)]}, squeue={"101": "RUNNING"})
    assert watch(guard, slurm, "--once") == 0
    assert guard.sleeps == [10, 10, 10]
    usr1 = ["scancel", "--batch", "--signal=USR1", "101"]
    assert slurm.calls("scancel") == [usr1, ["scancel", "101"]]


def test_five_failed_ticks_stop_jobs_fail_closed(slurm, guard, monkeypatch):
    slurm.record(101)
    slurm.update(sacct_exit=1, squeue={"101": "RUNNING"}, after_usr1={"101": ""})
    ticks = []

    def sleep(seconds):
        ticks.append(len(slurm.calls("scancel")))
        if len(ticks) == 5:
            raise KeyboardInterrupt

    monkeypatch.setattr(guard, "sleep", sleep)
    with pytest.raises(KeyboardInterrupt):
        watch(guard, slurm, "--interval", "60")
    assert ticks == [0, 0, 0, 0, 2]
    usr1 = ["scancel", "--batch", "--signal=USR1", "101"]
    assert slurm.calls("scancel") == [usr1, ["scancel", "101"]]
    stop = json.loads((slurm.ledger / "cap-stop.json").read_text())
    assert stop["reason"] == "accounting_unavailable"
    assert stop["charged_gpu_seconds"] is None
    assert not (slurm.ledger / "ledger.json").exists()


def test_concurrent_submit_fails_on_the_ledger_lock(slurm):
    slurm.update(sbatch={"stdout": "101\n", "block": True})
    argv = [sys.executable, str(SCRIPT), "submit", "--ledger", str(slurm.ledger)]
    first = subprocess.Popen(
        [*argv, *slurm.submit_argv()], stdout=subprocess.PIPE, text=True
    )
    try:
        deadline = time.monotonic() + 10
        while not slurm.calls("sbatch"):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        second = slurm.guard("submit", *slurm.submit_argv(gpus=0))
        assert second.returncode != 0 and "locked" in second.stderr
        blocked = ["--intent", "1", "--not-submitted", "--evidence", "x"]
        assert slurm.guard("reconcile", *blocked).returncode != 0
        assert [record["status"] for record in slurm.intents()] == ["intent"]
    finally:
        (slurm.root / "release").touch()
        output, _ = first.communicate(timeout=10)
    assert first.returncode == 0 and json.loads(output)["job_id"] == "101"
    assert len(slurm.calls("sbatch")) == 1


QUEUE = {
    "own": "101   RUNNING 1 N/A gres:gpu:geforce_rtx_2080:1 N/A N/A ours ",
    "foreign": "7001 PENDING 1 N/A gres/gpu=2 N/A N/A sweep a",
    "cpu": "7002 RUNNING 1 N/A N/A N/A N/A notebook",
    "array": "7003_[1-4] PENDING 1 N/A N/A N/A N/A cpu-array",
    "wide": "7004 RUNNING 2 gres:gpu:1 N/A N/A N/A wide",
}


def queued_submit(slurm, *lines, extra=(), gpus=1):
    slurm.update(queue=[QUEUE.get(line, line) for line in lines])
    return slurm.submit(job=102, gpus=gpus, extra=["--allow-active", *extra])


def test_unrecorded_gpu_job_of_the_user_blocks_submission(slurm):
    slurm.record(101)
    slurm.update(squeue={"101": "RUNNING"}, sacct={"101": [row(101, "RUNNING", 1)]})
    refused = queued_submit(slurm, "own", "foreign", "cpu")
    assert refused.returncode != 0
    assert '{"id": "7001", "name": "sweep a", "state": "PENDING"}' in refused.stderr
    assert "7002" not in refused.stderr and "101" not in refused.stderr
    assert queued_submit(slurm, "foreign", extra=["--known-job", "700"]).returncode
    assert queued_submit(slurm, "foreign", extra=["--known-job", "70011"]).returncode
    assert len(slurm.intents()) == 1 and refused.stdout == ""
    assert not slurm.calls("sbatch") and not slurm.calls("scancel")
    assert any(
        call[:5] == ["squeue", "-h", "-u", "tester", "-O"] for call in slurm.calls()
    )


def test_known_and_cpu_only_jobs_coexist_and_are_recorded(slurm):
    slurm.record(101)
    slurm.update(squeue={"101": "RUNNING"}, sacct={"101": [row(101, "RUNNING", 1)]})
    known = ["--known-job", "7001", "--known-job", "9999"]
    result = queued_submit(slurm, "own", "foreign", "cpu", "array", extra=known)
    assert result.returncode == 0, result.stderr
    assert slurm.intents()[1]["queue_at_intent"] == [
        {"id": "101", "state": "RUNNING", "name": "ours", "gpus": 1, "relation": "own"},
        {
            "id": "7001",
            "state": "PENDING",
            "name": "sweep a",
            "gpus": 2,
            "relation": "known",
        },
        {
            "id": "7002",
            "state": "RUNNING",
            "name": "notebook",
            "gpus": 0,
            "relation": "unrecorded",
        },
        {
            "id": "7003_[1-4]",
            "state": "PENDING",
            "name": "cpu-array",
            "gpus": 0,
            "relation": "unrecorded",
        },
    ]


def test_user_gpu_ceiling_blocks_submission(slurm):
    known = ["--known-job", "7001", "--known-job", "7004"]
    refused = queued_submit(slurm, "foreign", "wide", extra=known)  # 2 + 2x1 + 1 > 4
    assert refused.returncode != 0 and "exceeds the user's 4" in refused.stderr
    assert slurm.intents() == [] and not slurm.calls("sbatch")
    raised = [*known, "--max-user-gpus", "5"]
    assert queued_submit(slurm, "foreign", "wide", extra=raised).returncode == 0
    assert [job["gpus"] for job in slurm.intents()[0]["queue_at_intent"]] == [2, 2]


@pytest.mark.parametrize(
    "line",
    [
        "7001 PENDING 1 N/A gres:gpu:2 N/A N/A",
        "7001 PENDING one N/A gres:gpu:2 N/A N/A sweep",
        "7001 PENDING 1 N/A gres:gpumem:4 N/A N/A sweep",
        "7001 PENDING 1 N/A N/A gres:gpu:1 N/A sweep",
        "7001 PENDING 1 N/A N/A N/A gres:gpu:1 sweep",
        "7001_[1-4] PENDING 1 gres:gpu:1 N/A N/A N/A sweep",
        "garbage",
    ],
)
def test_unparsable_queue_line_blocks_submission(slurm, line):
    refused = queued_submit(slurm, line, extra=["--known-job", "7001"])
    assert refused.returncode != 0 and "unparsable" in refused.stderr
    assert slurm.intents() == [] and not slurm.calls("sbatch")
    slurm.update(queue=[], queue_exit=1)
    assert slurm.submit(job=102).returncode != 0 and not slurm.calls("sbatch")
    assert slurm.submit(job=102, gpus=0).returncode == 0  # CPU-only skips the check
    assert slurm.intents()[0]["queue_at_intent"] is None
    assert len([call for call in slurm.calls("squeue") if "-u" in call]) == 2


def test_missing_or_shrinking_accounting_never_refunds_budget(slurm):
    slurm.record(101)
    slurm.update(sacct={"101": [row(101, "COMPLETED", 950)]})
    assert slurm.guard("status", "--cap", "1000").returncode == 0
    written = (slurm.ledger / "ledger.json").read_bytes()
    for lost in ([], [row(101, "COMPLETED", 949)], [row(101, "COMPLETED", "")]):
        slurm.update(sacct={"101": lost})
        assert slurm.guard("status", "--cap", "1000").returncode != 0
        refused = slurm.submit(job=102)
        assert refused.returncode != 0 and "sacct" in refused.stderr
        assert (slurm.ledger / "ledger.json").read_bytes() == written
    assert len(slurm.intents()) == 1 and not slurm.calls("sbatch")
    slurm.update(sacct={"101": [row(101, "COMPLETED", 950)]})
    assert "exceeds cap" in slurm.submit(job=102).stderr


@pytest.mark.parametrize(
    "declared,sbatch",
    [
        ((0, 60), ["--gpus=8", "--time=24:00:00"]),
        ((1, 60), ["--gpus=2", "--time=1"]),
        ((1, 60), ["--gpus=1", "--time=2"]),
        ((1, 60), ["--gpus=1"]),
        ((1, 60), ["--gpus=1", "--time=1", "--time=1"]),
        ((1, 60), ["--time=1", "--gres=gpu:1"]),
        ((1, 60), ["--gpus=1", "--time=1", "--gpus-per-node=1"]),
        ((1, 60), ["--gpus=1", "--time=1", "--array=0-3"]),
        ((1, 60), ["--gpus=1", "--time=1", "-a", "0-3"]),
        ((1, 60), ["--gpus=1", "--time=UNLIMITED"]),
        ((1, 60), ["-G", "many", "-t", "1"]),
        ((0, 60), ["--time=1", "gres.sh"]),
        ((0, 60), ["--time=1", "array.sh"]),
        ((0, 60), ["--time=1", "glued.sh"]),
        ((0, 60), ["--time=1", "marker.sh"]),
        ((0, 60), ["--time=1", "pbs.sh"]),
        ((1, 60), ["--gpus=1", "--time=1", "--output", "log", "--gpus=8"]),
        ((1, 60), ["--gpus=1", "--time=1", "--output", "log", "--time=24:00:00"]),
        ((1, 60), ["--gpus=1", "--time=1", "--ntasks=4"]),
        ((1, 60), ["--gpus=1", "--time=1", "--mystery=1"]),
        ((1, 60), ["-G1", "-t1"]),
        ((0, 60), ["--time=1", "--wrap=true", "job.sh"]),
        ((0, 60), ["--time=1", "missing.sh"]),
    ],
)
def test_declared_resources_must_match_the_sbatch_request(slurm, declared, sbatch):
    (slurm.root / "gres.sh").write_text("#!/bin/sh\n#SBATCH --gres=gpu:4\n")
    (slurm.root / "array.sh").write_text("#!/bin/sh\n#SBATCH -a0-3\n")
    (slurm.root / "glued.sh").write_text("#!/bin/sh\n#SBATCH--gres=gpu:8\n")
    (slurm.root / "marker.sh").write_text("#!/bin/sh\n#SLURM --gres=gpu:8\n")
    (slurm.root / "pbs.sh").write_text("#!/bin/sh\n#PBS -l gpus=8\n")
    (slurm.root / "log").write_text("an existing output file is not the script\n")
    slurm.update(sbatch={"stdout": "101\n"})
    options = ["--cap", "1000", "--label", "run", "--user", "tester"]
    options += ["--gpus", str(declared[0]), "--time-limit", str(declared[1])]
    sbatch = [
        str(slurm.root / a) if a.endswith(".sh") or a == "log" else a for a in sbatch
    ]
    if not sbatch[-1].endswith(".sh") and "--wrap=true" not in sbatch:
        sbatch.append(slurm.script)
    refused = slurm.guard("submit", *options, "--", "sbatch", *sbatch)
    assert refused.returncode == 2 and refused.stdout == ""
    assert not slurm.calls("sbatch") and slurm.intents() == []


@pytest.mark.parametrize(
    "sbatch,expected",
    [
        (["--gpus=geforce_rtx_2080:1", "--time=22:00:00"], (1, 79200)),
        (["-G", "2", "-t", "1-01:02:03", "job.sh"], (2, 90123)),
        (["-t", "90", "--wrap=true"], (0, 5400)),
        (["-A", "gpu-students", "--time=2-0", "job.sh", "--gpus=9"], (0, 172800)),
    ],
)
def test_sbatch_request_parsing(guard, tmp_path, monkeypatch, sbatch, expected):
    monkeypatch.chdir(tmp_path)
    Path("job.sh").write_text("#!/bin/sh\n#SBATCH --time=24:00:00 --ntasks=1\n")
    if "job.sh" not in sbatch and "--wrap=true" not in sbatch:
        sbatch = [*sbatch, "job.sh"]
    assert guard.sbatch_request(["sbatch", *sbatch]) == expected
    monkeypatch.setenv("SBATCH_GPUS", "8")
    with pytest.raises(guard.GuardError, match="environment"):
        guard.sbatch_request(["sbatch", *sbatch])


def test_overlapping_refreshes_cannot_publish_a_stale_charge(slurm, guard):
    slurm.record(101)
    slurm.update(sacct={"101": [row(101, "RUNNING", 200)]})
    stale = guard.compute_ledger(guard.load_intents(slurm.ledger), 1000, None)
    slurm.update(sacct={"101": [row(101, "RUNNING", 300)]})
    assert guard.refresh(slurm.ledger, 1000)["charged_gpu_seconds"] == 300
    # A refresh that started earlier must not finish its write unserialized.
    guard.LOCK_WAIT = 0
    with guard.lock(slurm.ledger), pytest.raises(guard.GuardError, match="locked"):
        guard.refresh(slurm.ledger, 1000)
    assert stale["charged_gpu_seconds"] == 200
    assert (
        json.loads((slurm.ledger / "ledger.json").read_text())["charged_gpu_seconds"]
        == 300
    )
    slurm.update(sacct={"101": [row(101, "RUNNING", 250)]})
    assert slurm.guard("status", "--cap", "1000").returncode != 0


def test_accounting_ignores_an_older_job_with_a_reused_id(slurm, guard, monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Jerusalem")
    time.tzset()
    slurm.record(101)
    slurm.update(sacct={"101": [row(101, "COMPLETED", 5)]})
    assert slurm.guard("status", "--cap", "1000").returncode == 0
    call = slurm.calls("sacct")[-1]
    # One hour before the intent, in the scheduler host's local time.
    assert call[call.index("-S") + 1] == "2026-10-09T07:29:33"
    for invalid in (None, "yesterday", "2026-10-09T05:29:33"):
        with pytest.raises(guard.GuardError):
            guard.accounting_start(invalid)
