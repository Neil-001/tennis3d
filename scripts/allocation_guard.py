"""Fail-closed GPU allocation budget guard for one Slurm ledger directory.

Charge rule: every allocation row of every recorded job, as reported by
`sacct -X -n -P -D -j ID` (one row per requeued or restarted allocation), costs
gpu_count * ElapsedRaw seconds, where gpu_count is the `gres/gpu=N` item of
AllocTRES. Running, failed, cancelled, requeued and completed rows all count.
Slurm reuses job ids, so accounting is read only from shortly before the
intent was created; an older job with the same id is not this allocation.

Charges never shrink: a recorded job without accounting rows, or with fewer
charged seconds than the last written ledger, is an error rather than a refund.

`submit` requires the sbatch command to request exactly the declared GPUs and
time limit, writes a durable intent before sbatch runs and refuses while any
earlier intent has no settled outcome. `watch` only signals and cancels: this
guard never resubmits anything. Stdlib only, so it runs on a login host.
"""

import argparse
import contextlib
import fcntl
import getpass
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

SACCT_FORMAT = "JobID,State,Start,End,ElapsedRaw,AllocTRES,NodeList"
SACCT_FIELDS = ("job", "state", "start", "end", "elapsed", "tres", "node")
JOB_LINE = re.compile(r"\d+(;\S+)?")
GPU_TRES = re.compile(r"(?:^|,)gres/gpu=(\d+)(?:,|$)")
GPU_REQUEST = re.compile(r"(?:gres[:/])?gpu(?:[:=][A-Za-z_][\w.-]*)?(?:[:=](\d+))?")
# Whitespace-separated columns; the job name comes last and may contain spaces.
# A GPU request appears per node or per job depending on how it was submitted.
QUEUE_FORMAT = (
    "JobID:30,State:30,NumNodes:10,tres-per-node:100,tres-per-job:100,"
    "tres-per-task:100,tres-per-socket:100,Name:256"
)
STATUSES = ("intent", "uncertain", "submitted", "not_submitted")
UNRESOLVED = STATUSES[:2]
MAX_FAILED_TICKS = 5
POLL_SECONDS = 10
COMMAND_TIMEOUT = 120
LOCK_WAIT = 300
# The only sbatch options a guarded submission may use. Anything else, including
# every other way to request GPUs, tasks or arrays, is refused, not interpreted.
SHORT_OPTIONS = {"-A", "-p", "-J", "-o", "-e", "-w", "-c", "-t", "-G", "-D"}
LONG_OPTIONS = {
    "--account",
    "--partition",
    "--job-name",
    "--output",
    "--error",
    "--nodelist",
    "--cpus-per-task",
    "--mem",
    "--time",
    "--gpus",
    "--chdir",
    "--signal",
    "--open-mode",
    "--ntasks",
    "--wrap",
}
FLAGS = {"--parsable", "--requeue"}
DIRECTIVES = LONG_OPTIONS - {"--gpus", "--wrap", "--chdir"}
sleep = time.sleep
monotonic = time.monotonic


class GuardError(Exception):
    pass


def utc_now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def run(*argv):
    try:
        return subprocess.run(
            argv, capture_output=True, text=True, timeout=COMMAND_TIMEOUT, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GuardError(f"{argv[0]} could not run: {error}") from error


def write_json(path, value, exclusive=False):
    target = path if exclusive else path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(target, "x" if exclusive else "w") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    if not exclusive:
        os.replace(target, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def lock(ledger, wait=0.0):
    """Exclusive ledger lock; every read-compute-write of the ledger holds it."""
    deadline = monotonic() + wait
    with open(ledger / "guard.lock", "a") as stream:
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as error:
                if monotonic() >= deadline:
                    raise GuardError(
                        "ledger is locked by another guard command"
                    ) from error
                sleep(0.2)
        yield


def load_intents(ledger):
    intents = {}
    for path in sorted(ledger.glob("intent-*.json")):
        try:
            record = json.loads(path.read_text())
            known = record["status"] in STATUSES
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise GuardError(f"{path.name} is unreadable: {error}") from error
        if not known or (record["status"] == "submitted") != ("job_id" in record):
            raise GuardError(f"{path.name} has an inconsistent status")
        intents[path.stem.removeprefix("intent-")] = record
    return intents


def active_state(job):
    result = run("squeue", "-h", "-j", job, "-o", "%T")
    if result.returncode == 0:
        return ",".join(sorted(set(result.stdout.split())))
    if "Invalid job id" in result.stderr:  # purged from the controller: finished
        return ""
    raise GuardError(f"squeue failed for job {job}: {result.stderr.strip()}")


def is_running(state):
    return bool({"RUNNING", "UNKNOWN"} & set(state.split(",")))


def accounting_start(created_at):
    """sacct -S value in the host's local time, an hour before the intent."""
    try:
        created = datetime.fromisoformat(created_at)
    except (TypeError, ValueError) as error:
        raise GuardError(f"intent creation time {created_at!r} is invalid") from error
    if created.tzinfo is None:
        raise GuardError(f"intent creation time {created_at!r} has no time zone")
    local = (created - timedelta(hours=1)).astimezone()
    return local.strftime("%Y-%m-%dT%H:%M:%S")


def allocations(job, declared_gpus, created_at):
    since = accounting_start(created_at)
    result = run(
        "sacct", "-X", "-n", "-P", "-D", "-S", since, "-j", job, "-o", SACCT_FORMAT
    )
    if result.returncode:
        raise GuardError(f"sacct failed for job {job}: {result.stderr.strip()}")
    rows = []
    for line in filter(None, map(str.strip, result.stdout.splitlines())):
        fields = line.split("|")
        if (
            len(fields) != len(SACCT_FIELDS)
            or not re.fullmatch(rf"{job}(\D.*)?", fields[0])
            or not fields[4].isdecimal()
        ):
            raise GuardError(f"sacct row for job {job} is unparsable: {line!r}")
        row = dict(zip(SACCT_FIELDS, fields))
        found = GPU_TRES.search(row.pop("tres"))
        elapsed = int(row["elapsed"])
        # A row without GPU TRES is charged as declared rather than as free.
        gpus = int(found[1]) if found else declared_gpus
        rows.append(row | {"elapsed": elapsed, "gpus": gpus, "node": row.pop("node")})
        rows[-1]["gpu_seconds"] = gpus * elapsed
    return rows


def compute_ledger(intents, cap, previous=None):
    """Charges of every recorded job; `previous` is the last written ledger."""
    floor = {
        job["job_id"]: job["gpu_seconds"] for job in (previous or {}).get("jobs", [])
    }
    jobs = []
    for name, record in intents.items():
        if record["status"] != "submitted":
            continue
        job = record["job_id"]
        state = active_state(job)
        rows = allocations(job, record["gpus"], record.get("created_at"))
        charged = sum(row["gpu_seconds"] for row in rows)
        # Missing or shrinking accounting must never hand budget back.
        if not rows:
            raise GuardError(f"sacct reports no allocation row for recorded job {job}")
        if charged < floor.get(job, 0):
            raise GuardError(
                f"sacct charge for job {job} fell from {floor[job]} to {charged}"
            )
        jobs.append(
            {
                "intent": name,
                "label": record["label"],
                "job_id": job,
                "declared_gpus": record["gpus"],
                "active_state": state,
                "allocations": rows,
                "gpu_seconds": charged,
            }
        )
    charged = sum(job["gpu_seconds"] for job in jobs)
    return {
        "at": utc_now(),
        "cap_gpu_seconds": cap,
        "charged_gpu_seconds": charged,
        "remaining_gpu_seconds": cap - charged,
        "unresolved_intents": [
            name for name, record in intents.items() if record["status"] in UNRESOLVED
        ],
        "jobs": jobs,
    }


def requested_gpus(request):
    """GPUs named by one squeue TRES request column; None when unparsable."""
    if request in ("", "N/A"):
        return 0
    total = 0
    for item in request.split(","):
        found = GPU_REQUEST.fullmatch(item)
        if "gpu" in item and not found:
            return None
        total += int(found[1] or 1) if found else 0
    return total


def queued_gpus(nodes, per_node, per_job, per_task, per_socket):
    """Total GPUs of one queue line; None when they cannot be counted exactly."""
    counts = [requested_gpus(column) for column in (per_node, per_job)]
    # Per-task and per-socket GPU requests have no exact total here.
    if None in counts or not nodes.isdecimal() or "gpu" in per_task + per_socket:
        return None
    return counts[0] * int(nodes) + counts[1]


def user_queue(user, own, known):
    """Every queued or running job of the user, with GPU jobs classified."""
    result = run("squeue", "-h", "-u", user, "-O", QUEUE_FORMAT)
    if result.returncode:
        raise GuardError(f"squeue failed for user {user}: {result.stderr.strip()}")
    queue = []
    for line in filter(None, map(str.strip, result.stdout.splitlines())):
        fields = line.split(None, 7)
        gpus = queued_gpus(*fields[2:7]) if len(fields) == 8 else None
        # A GPU array or het component id is not an exact id: refuse to guess.
        if gpus is None or not re.fullmatch(r"[0-9]+" if gpus else r"\S+", fields[0]):
            raise GuardError(f"squeue line for user {user} is unparsable: {line!r}")
        job = fields[0]
        relation = "own" if job in own else "known" if job in known else "unrecorded"
        queue.append(
            {
                "id": job,
                "state": fields[1],
                "name": fields[7],
                "gpus": gpus,
                "relation": relation,
            }
        )
    return queue


def check_coexistence(queue, gpus, ceiling):
    strangers = [
        {key: job[key] for key in ("id", "name", "state")}
        for job in queue
        if job["gpus"] and job["relation"] == "unrecorded"
    ]
    if strangers:
        raise GuardError(
            "GPU jobs neither recorded in this ledger nor passed as --known-job: "
            + json.dumps(strangers)
        )
    held = sum(job["gpus"] for job in queue)
    if held + gpus > ceiling:
        raise GuardError(f"queued {held} + {gpus} GPUs exceeds the user's {ceiling}")


def previous_ledger(ledger):
    path = ledger / "ledger.json"
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except (OSError, ValueError) as error:
        raise GuardError(f"ledger.json is unreadable: {error}") from error


def refresh(ledger, cap):
    # Two unserialized refreshes could publish an older, smaller charge last.
    with lock(ledger, wait=LOCK_WAIT):
        state = compute_ledger(load_intents(ledger), cap, previous_ledger(ledger))
        write_json(ledger / "ledger.json", state)
    return state


def slurm_seconds(text):
    """Seconds of a Slurm time limit, or None when it is not a plain limit."""
    found = re.fullmatch(r"(?:(\d+)-)?(\d+)(?::(\d+))?(?::(\d+))?", text)
    if not found:
        return None
    days, *parts = (int(part) if part else None for part in found.groups())
    parts = [part for part in parts if part is not None]
    if days is not None:  # D-HH[:MM[:SS]]
        hours, minutes, seconds = (parts + [0, 0])[:3]
    elif len(parts) == 3:  # HH:MM:SS
        hours, minutes, seconds = parts
    else:  # MM[:SS]
        hours, (minutes, seconds) = 0, (parts + [0])[:2]
    return (((days or 0) * 24 + hours) * 60 + minutes) * 60 + seconds


def sbatch_request(argv):
    """(gpus, time limit seconds) that an sbatch command line asks for.

    The command may use only the listed options, long ones as `--name=value`.
    The first other word is the batch script and the rest are its arguments.
    Environment defaults and script directives that could change the request
    are refused, so the declared GPUs and time are the whole request.
    """
    inherited = sorted(name for name in os.environ if name.startswith("SBATCH_"))
    if inherited:
        raise GuardError(
            f"sbatch environment defaults are not accountable: {inherited}"
        )
    values = {"gpus": [], "time": [], "ntasks": []}
    names = {"--gpus": "gpus", "-G": "gpus", "--time": "time", "-t": "time"}
    names["--ntasks"] = "ntasks"
    options, script, wrapped = argv[1:], None, False
    index = 0
    while index < len(options):
        token = options[index]
        index += 1
        name, joined, value = token.partition("=")
        if token in FLAGS:
            continue
        if token in SHORT_OPTIONS and index < len(options):
            name, value, index = token, options[index], index + 1
        elif not (name in LONG_OPTIONS and joined):
            if token.startswith("-"):
                raise GuardError(f"sbatch option {token!r} is not accountable")
            script = Path(token)
            break
        wrapped = wrapped or name == "--wrap"
        if name in names:
            values[names[name]].append(value)
    if (script is None) == (not wrapped) or (script and not script.is_file()):
        raise GuardError("sbatch needs one existing batch script or one --wrap")
    for line in script.read_text(errors="replace").splitlines() if script else ():
        # sbatch also reads glued "#SBATCH--opt" and other markers; allow neither.
        if re.match(r"#(SLURM|PBS|BSUB)|#SBATCH\S", line):
            raise GuardError(f"batch script directive is not accountable: {line}")
        if line.startswith("#SBATCH"):
            for word in line.split()[1:]:
                if word.partition("=")[0] not in DIRECTIVES | FLAGS:
                    raise GuardError(
                        f"batch script directive is not accountable: {line}"
                    )
                if word.startswith("--ntasks="):
                    values["ntasks"].append(word.partition("=")[2])
    if len(values["time"]) != 1 or len(values["gpus"]) > 1:
        raise GuardError("sbatch needs exactly one --time and at most one --gpus")
    if any(count != "1" for count in values["ntasks"]):
        raise GuardError("a guarded submission is a single task")
    gpus = 0
    if values["gpus"]:
        found = re.fullmatch(r"(?:[A-Za-z_][\w.-]*:)?(\d+)", values["gpus"][0])
        if not found:
            raise GuardError(f"sbatch --gpus value {values['gpus'][0]!r} is unparsable")
        gpus = int(found[1])
    return gpus, slurm_seconds(values["time"][0])


def sbatch_outcome(argv, known_jobs):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
    except (OSError, ValueError, KeyboardInterrupt) as error:
        return {"status": "uncertain", "error": repr(error)}
    sys.stderr.write(result.stderr)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    outcome = {
        "sbatch_exit": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }
    job = lines[-1].split(";")[0] if lines and JOB_LINE.fullmatch(lines[-1]) else None
    if result.returncode == 0 and job and job not in known_jobs:
        return outcome | {"status": "submitted", "job_id": job}
    if result.returncode > 0 and not any(map(JOB_LINE.fullmatch, lines)):
        return outcome | {"status": "not_submitted"}
    return outcome | {"status": "uncertain"}


def submit(args):
    if args.sbatch[:1] != ["sbatch"]:
        raise GuardError("the command after -- must start with sbatch")
    if args.gpus < 0 or args.time_limit <= 0:
        raise GuardError("--gpus must be >= 0 and --time-limit must be > 0")
    if sbatch_request(args.sbatch) != (args.gpus, args.time_limit):
        raise GuardError(
            f"sbatch requests {sbatch_request(args.sbatch)} (GPUs, seconds), not "
            f"the declared {(args.gpus, args.time_limit)}"
        )
    args.user = args.user or getpass.getuser()
    ledger = args.ledger
    ledger.mkdir(mode=0o700, parents=True, exist_ok=True)
    with lock(ledger):
        intents = load_intents(ledger)
        state = compute_ledger(intents, args.cap, previous_ledger(ledger))
        write_json(ledger / "ledger.json", state)
        if state["unresolved_intents"]:
            raise GuardError(f"unreconciled intents: {state['unresolved_intents']}")
        if (ledger / "cap-stop.json").exists():
            raise GuardError("cap-stop.json exists: the watcher would cancel this job")
        busy = [
            job["job_id"]
            for job in state["jobs"]
            if job["declared_gpus"] > 0 and job["active_state"]
        ]
        if args.gpus > 0 and busy and not args.allow_active:
            raise GuardError(f"GPU jobs are still active: {busy}")
        charged = state["charged_gpu_seconds"]
        if charged + args.gpus * args.time_limit > args.cap:
            raise GuardError(
                f"charged {charged} + {args.gpus} x {args.time_limit} exceeds "
                f"cap {args.cap}"
            )
        queue = None
        if args.gpus > 0:
            own = {job["job_id"] for job in state["jobs"]}
            queue = user_queue(args.user, own, set(args.known_job))
            check_coexistence(queue, args.gpus, args.max_user_gpus)
        argv = list(args.sbatch)
        if "--parsable" not in argv:
            argv.insert(1, "--parsable")
        record = {
            "status": "intent",
            "label": args.label,
            "gpus": args.gpus,
            "time_limit_seconds": args.time_limit,
            "cap": args.cap,
            "charged_before": charged,
            "queue_at_intent": queue,
            "argv": argv,
            "cwd": os.getcwd(),
            "created_at": utc_now(),
        }
        path = ledger / f"intent-{max(map(int, intents), default=0) + 1:04d}.json"
        write_json(path, record, exclusive=True)
        record.update(sbatch_outcome(argv, {job["job_id"] for job in state["jobs"]}))
        write_json(path, record)
    print(json.dumps(record))
    return {"submitted": 0, "not_submitted": 1}.get(record["status"], 3)


def reconcile(args):
    with lock(args.ledger):
        intents = load_intents(args.ledger)
        name = f"{args.intent:04d}"
        record = intents.get(name)
        if record is None or record["status"] not in UNRESOLVED:
            raise GuardError(f"intent {name} is not awaiting reconciliation")
        if not args.evidence.strip():
            raise GuardError("--evidence must not be empty")
        if args.job_id is None:
            record["status"] = "not_submitted"
        elif not re.fullmatch(r"[0-9]+", args.job_id):
            raise GuardError("--job-id must be a numeric Slurm job id")
        elif any(other.get("job_id") == args.job_id for other in intents.values()):
            raise GuardError(f"job {args.job_id} is already recorded")
        else:
            record |= {"status": "submitted", "job_id": args.job_id}
        record |= {"evidence": args.evidence, "reconciled_at": utc_now()}
        write_json(args.ledger / f"intent-{name}.json", record)
    print(json.dumps(record))
    return 0


def status(args):
    print(json.dumps(refresh(args.ledger, args.cap), indent=2))
    return 0


def probe(job):
    try:
        return active_state(job)
    except GuardError:
        return "UNKNOWN"


def final_state(job, created_at):
    try:
        accounted = [row["state"] for row in allocations(job, 0, created_at)]
    except GuardError as error:
        accounted = str(error)
    return {"squeue": probe(job), "sacct": accounted}


def stop_jobs(ledger, reason, charged, args):
    """Stop every active recorded job; only the first stop signals and waits."""
    path = ledger / "cap-stop.json"
    created = {
        record["job_id"]: record.get("created_at")
        for record in load_intents(ledger).values()
        if record["status"] == "submitted"
    }
    jobs = list(created)
    first = not path.exists()
    if first:
        stop = {
            "reason": reason,
            "at": utc_now(),
            "charged_gpu_seconds": charged,
            "cap_gpu_seconds": args.cap,
            "margin_gpu_seconds": args.margin,
            "actions": [],
        }
        write_json(path, stop, exclusive=True)
    else:
        stop = json.loads(path.read_text())

    def scancel(job, *options):
        argv = ["scancel", *options, job]
        try:
            result = run(*argv)
            outcome = {"exit": result.returncode, "stderr": result.stderr[-500:]}
        except GuardError as error:
            outcome = {"exit": None, "stderr": str(error)}
        stop["actions"].append({"at": utc_now(), "argv": argv} | outcome)
        write_json(path, stop)

    states = {job: probe(job) for job in jobs}
    live = [job for job in jobs if states[job]]
    graceful = [job for job in live if first and is_running(states[job])]
    for job in live:
        if job not in graceful:  # pending work is cancelled before any waiting
            scancel(job)
    for job in graceful:
        scancel(job, "--batch", "--signal=USR1")
    deadline = monotonic() + args.grace
    while monotonic() < deadline and any(is_running(probe(job)) for job in graceful):
        sleep(POLL_SECONDS)
    for job in graceful:
        scancel(job)
    if first:
        stop["final"] = {job: final_state(job, created[job]) for job in live}
        write_json(path, stop)


def tick(args, failures=0):
    """Run one watch pass and return the consecutive failed-tick count."""
    stopped = (args.ledger / "cap-stop.json").exists()
    try:
        charged = refresh(args.ledger, args.cap)["charged_gpu_seconds"]
        failures = 0
        reason = "cap_margin"
        due = stopped or charged + args.margin >= args.cap
    except GuardError as error:
        failures += 1
        print(f"allocation_guard: tick failed ({failures}): {error}", file=sys.stderr)
        charged = None
        reason = "accounting_unavailable"
        due = stopped or failures >= MAX_FAILED_TICKS
    if due:
        try:
            stop_jobs(args.ledger, reason, charged, args)
        except (GuardError, OSError, ValueError) as error:
            failures = max(failures, 1)
            print(f"allocation_guard: stop failed: {error}", file=sys.stderr)
    return failures


def watch(args):
    failures = 0
    while True:
        failures = tick(args, failures)
        if args.once:
            return 1 if failures else 0
        sleep(args.interval)


def parser():
    root = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = root.add_subparsers(dest="command", required=True)

    def command(name, cap=True):
        sub = commands.add_parser(name)
        sub.add_argument("--ledger", type=Path, required=True)
        if cap:
            sub.add_argument("--cap", type=int, required=True)
        return sub

    sub = command("submit")
    sub.add_argument("--label", required=True)
    sub.add_argument("--gpus", type=int, required=True)
    sub.add_argument("--time-limit", type=int, required=True)
    sub.add_argument("--allow-active", action="store_true")
    sub.add_argument("--known-job", action="append", default=[])
    sub.add_argument("--max-user-gpus", type=int, default=4)
    sub.add_argument("--user", default=None)
    sub = command("reconcile", cap=False)
    sub.add_argument("--intent", type=int, required=True)
    outcome = sub.add_mutually_exclusive_group(required=True)
    outcome.add_argument("--job-id")
    outcome.add_argument("--not-submitted", action="store_true")
    sub.add_argument("--evidence", required=True)
    command("status")
    sub = command("watch")
    sub.add_argument("--margin", type=int, required=True)
    sub.add_argument("--interval", type=float, default=60)
    sub.add_argument("--grace", type=float, default=1000)
    sub.add_argument("--once", action="store_true")
    return root


COMMANDS = {"submit": submit, "reconcile": reconcile, "status": status, "watch": watch}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cut = argv.index("--") if "--" in argv else len(argv)
    args = parser().parse_args(argv[:cut])
    args.sbatch = argv[cut + 1 :]
    try:
        if args.command != "submit" and not args.ledger.is_dir():
            raise GuardError(f"ledger directory {args.ledger} does not exist")
        return COMMANDS[args.command](args)
    except GuardError as error:
        print(f"allocation_guard: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
