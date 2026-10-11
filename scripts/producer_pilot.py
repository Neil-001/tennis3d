"""Measure a bounded local producer run. Publication and Azure stay separate.

Run with the task Python, which is also the exact child executable. This script
holds the supplied cpu.lock until generation, draining and bounded local reload
finish. It never changes the recipe or execution config to make a pilot pass.
"""

import argparse
import fcntl
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc_now():
    return datetime.now(UTC).isoformat()


@contextmanager
def cpu_lock(path):
    """Refuse concurrent work instead of waiting with an unbounded deadline."""
    path = Path(path).resolve()
    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("cpu.lock is already held") from None
        stat = os.fstat(stream.fileno())
        yield {
            "path": str(path),
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "holder_pid": os.getpid(),
            "mode": "exclusive flock held by pilot wrapper",
        }


def disk_usage(roots):
    """Count a nested external pool once, including writer temporary files."""
    unique = sorted({Path(p).resolve() for p in roots}, key=lambda p: len(p.parts))
    parents = []
    for root in unique:
        if not any(root.is_relative_to(parent) for parent in parents):
            parents.append(root)
    logical, allocated = 0, 0
    for root in parents:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            try:
                if path.is_symlink():
                    raise ValueError("pilot staging/pool roots cannot contain symlinks")
                if path.is_file():
                    stat = path.stat()
                    logical += stat.st_size
                    allocated += stat.st_blocks * 512
            except FileNotFoundError:
                # Atomic replacement/removal can race the sampler.
                continue
    return {"logical_bytes": logical, "allocated_bytes": allocated}


def proc_stat(path):
    raw = (path / "stat").read_text()
    fields = raw[raw.rfind(")") + 2 :].split()
    return {
        "pid": int(path.name),
        "ppid": int(fields[1]),
        "pgrp": int(fields[2]),
        "start_ticks": int(fields[19]),
        "cpu_ticks": int(fields[11]) + int(fields[12]),
        "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
    }


def process_tree(root_pid):
    """Include the session's workers even if they become briefly orphaned."""
    all_processes = {}
    for path in Path("/proc").iterdir():
        if path.name.isdecimal():
            try:
                item = proc_stat(path)
                all_processes[item["pid"]] = item
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
    owned = {root_pid}
    while True:
        additions = {
            pid
            for pid, item in all_processes.items()
            if item["ppid"] in owned or item["pgrp"] == root_pid
        }
        if additions <= owned:
            break
        owned.update(additions)
    return [all_processes[pid] for pid in sorted(owned) if pid in all_processes]


class Sampler:
    def __init__(self, roots):
        self.roots = roots
        self.samples = 0
        self.cpu_ticks = {}
        self.wrapper_cpu_start = proc_stat(Path("/proc") / str(os.getpid()))[
            "cpu_ticks"
        ]
        self.peaks = {}
        self.pss_incomplete_samples = 0

    def sample(self, pid):
        processes = process_tree(pid)
        # The measurement controller also consumes physical RAM during a run.
        if pid != os.getpid():
            processes.append(proc_stat(Path("/proc") / str(os.getpid())))
        rss, pss, missing = 0, 0, 0
        for item in processes:
            key = item["pid"], item["start_ticks"]
            self.cpu_ticks[key] = max(self.cpu_ticks.get(key, 0), item["cpu_ticks"])
            rss += item["rss_bytes"]
            try:
                values = {}
                for line in (
                    (Path("/proc") / str(item["pid"]) / "smaps_rollup")
                    .read_text()
                    .splitlines()
                ):
                    name, _, value = line.partition(":")
                    if name in ("Rss", "Pss"):
                        values[name] = int(value.split()[0]) * 1024
                if "Pss" not in values:
                    missing += 1
                else:
                    pss += values["Pss"]
                    rss += values.get("Rss", item["rss_bytes"]) - item["rss_bytes"]
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                missing += 1
        disk = disk_usage(self.roots)
        current = {
            "rss_bytes": rss,
            "pss_bytes": pss,
            "pss_missing_processes": missing,
            "processes": len(processes),
            "disk_logical_bytes": disk["logical_bytes"],
            "disk_allocated_bytes": disk["allocated_bytes"],
        }
        for name in (
            "rss_bytes",
            "pss_bytes",
            "processes",
            "disk_logical_bytes",
            "disk_allocated_bytes",
        ):
            self.peaks[name] = max(self.peaks.get(name, 0), current[name])
        self.samples += 1
        self.pss_incomplete_samples += bool(missing)
        return current

    def report(self):
        return {
            "sample_count": self.samples,
            "sampled_peaks": self.peaks,
            "pss_incomplete_samples": self.pss_incomplete_samples,
            "sampled_cpu_seconds_lower_bound": max(
                0, sum(self.cpu_ticks.values()) - self.wrapper_cpu_start
            )
            / os.sysconf("SC_CLK_TCK"),
            "notes": [
                "PSS apportions shared pages; aggregate RSS double counts shared pages.",
                "Sampled peaks can miss brief allocations. PSS is partial when a process vanishes or smaps access fails.",
                "CPU sampling can miss processes whose entire lifetime falls between samples.",
                "RAM includes the wrapper, generator and its process tree. CPU sampling includes wrapper overhead; wait4 and producer CPU exclude the wrapper.",
                "Disk covers the union of output and pool roots, including temporary files; wrapper evidence/logs are separate.",
            ],
        }


def signal_group(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def run_bounded(
    argv, *, cwd, env, roots, log_path, samples_path, timeout, drain, interval
):
    """Return wait4 usage before Popen can reap the process without its usage."""
    started = time.monotonic()
    sampler = Sampler(roots)
    stop_at, hard_at = started + timeout - drain, started + timeout
    requested, killed, stop_reason = False, False, None
    interrupted = []
    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(
            sig, lambda number, frame: interrupted.append(number)
        )
    process = None
    usage = None
    try:
        with Path(log_path).open("xb") as log, Path(samples_path).open("x") as samples:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while True:
                now = time.monotonic()
                if not requested and (now >= stop_at or interrupted):
                    requested = True
                    stop_reason = (
                        "caller_signal" if interrupted else "generation_deadline"
                    )
                    hard_at = min(hard_at, now + drain)
                    try:
                        os.kill(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                if now >= hard_at and not killed:
                    killed = True
                    signal_group(process.pid, signal.SIGKILL)
                point = sampler.sample(process.pid)
                samples.write(
                    json.dumps({"elapsed_seconds": time.monotonic() - started, **point})
                    + "\n"
                )
                samples.flush()
                try:
                    waited, status, usage_value = os.wait4(process.pid, os.WNOHANG)
                except ChildProcessError:
                    break
                if waited:
                    process.returncode = os.waitstatus_to_exitcode(status)
                    usage = usage_value
                    # A successful producer reaps its workers. Any escaped drain
                    # tail in the owned group must not outlive the wrapper lock.
                    signal_group(process.pid, signal.SIGKILL)
                    break
                time.sleep(min(interval, max(0.001, hard_at - time.monotonic())))
    finally:
        if process is not None and process.returncode is None:
            signal_group(process.pid, signal.SIGKILL)
            try:
                _, status, usage = os.wait4(process.pid, 0)
                process.returncode = os.waitstatus_to_exitcode(status)
            except ChildProcessError:
                pass
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return {
        "returncode": process.returncode if process is not None else None,
        "wall_seconds": time.monotonic() - started,
        "wait4_cpu_seconds_including_reaped_descendants": (
            usage.ru_utime + usage.ru_stime if usage is not None else None
        ),
        "wait4_ru_maxrss_kib": usage.ru_maxrss if usage is not None else None,
        "stop_requested": requested,
        "stop_reason": stop_reason,
        "hard_kill_sent": killed,
        "resources": sampler.report(),
    }


def python_size(value, seen=None):
    """Approximate retained Python row-group objects, counted once by identity."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(
            python_size(k, seen) + python_size(v, seen) for k, v in value.items()
        )
    elif isinstance(value, (list, tuple)):
        size += sum(python_size(item, seen) for item in value)
    return size


def inspect_parquet(
    output, execution, *, max_groups=4, max_rows=16, max_group_bytes=64 << 20
):
    """Metadata for every shard, numeric reload only for a bounded sample."""
    import pyarrow.parquet as pq

    from tennis3d.datasets.io import ShardReader, read_manifest
    from tennis3d.synthetic.producer_state import load

    files = Path(output) / "files"
    manifest_path = files / execution.prefix / "manifest.json"
    if not manifest_path.exists():
        return {"status": "blocked", "reason": "no completed local manifest"}
    manifest = read_manifest(manifest_path)
    groups, footer_bytes, compressed_bytes, shard_bytes = [], 0, 0, 0
    metadata_started = time.monotonic()
    for shard in manifest.shards:
        with pq.ParquetFile(files / shard.path) as parquet:
            metadata = parquet.metadata
            footer_bytes += metadata.serialized_size + 12
            shard_bytes += shard.size_bytes
            for i in range(metadata.num_row_groups):
                group = metadata.row_group(i)
                compressed = sum(
                    group.column(j).total_compressed_size
                    for j in range(group.num_columns)
                )
                compressed_bytes += compressed
                groups.append(
                    {
                        "shard": shard.path,
                        "group": i,
                        "rows": group.num_rows,
                        "parquet_uncompressed_bytes": group.total_byte_size,
                        "compressed_column_bytes": compressed,
                    }
                )
    metadata_seconds = time.monotonic() - metadata_started
    ordered = sorted(
        groups,
        key=lambda group: (
            -group["parquet_uncompressed_bytes"],
            group["shard"],
            group["group"],
        ),
    )
    by_path = {shard.path: shard for shard in manifest.shards}
    reloaded, rows = [], 0
    reload_sampler = Sampler((Path(output),))
    reload_sampler.sample(os.getpid())
    for group in ordered:
        if len(reloaded) >= max_groups:
            break
        if (
            rows + group["rows"] > max_rows
            or group["rows"] * execution.max_record_bytes > max_group_bytes
        ):
            continue
        shard = by_path[group["shard"]]
        path = files / shard.path
        started, cpu = time.monotonic(), time.process_time()
        with pq.ParquetFile(path) as parquet:
            table = parquet.read_row_group(group["group"], use_threads=False)
            arrow_bytes = table.nbytes
            reload_sampler.sample(os.getpid())
            del table
        ids, numeric_bytes = [], 0
        with ShardReader(path, shard, provenance=manifest.provenance) as reader:
            verified_at = time.monotonic()
            for row in range(
                group["group"] * shard.row_group_rallies,
                group["group"] * shard.row_group_rallies + group["rows"],
            ):
                record = reader.read(row)
                ids.append(record.metadata.rally_id)
                numeric_bytes += sum(a.nbytes for a in record.physical.values()) + sum(
                    a.nbytes for view in record.views for a in view.arrays.values()
                )
                del record
            decoded_at = time.monotonic()
            reload_sampler.sample(os.getpid())
            python_bytes = python_size(reader._rows)
        reloaded.append(
            {
                **group,
                "rally_ids": ids,
                "arrow_buffer_bytes": arrow_bytes,
                "typed_numeric_bytes": numeric_bytes,
                "retained_python_group_bytes_estimate": python_bytes,
                "open_verify_plus_arrow_seconds": verified_at - started,
                "reader_decode_seconds": decoded_at - verified_at,
                "wall_seconds": time.monotonic() - started,
                "cpu_seconds": time.process_time() - cpu,
            }
        )
        rows += group["rows"]
    summary_path = files / execution.prefix / "summary.json"
    summary = load(summary_path) if summary_path.exists() else None
    physical_rallies = sum(len(shard.records) for shard in manifest.shards)
    return {
        "status": "measured" if reloaded else "blocked",
        "metadata_seconds": metadata_seconds,
        "physical_rallies": physical_rallies,
        "shards": len(manifest.shards),
        "row_groups": len(groups),
        "shard_bytes": shard_bytes,
        "compressed_column_bytes": compressed_bytes,
        "footer_and_magic_bytes": footer_bytes,
        "other_encoded_overhead_bytes": shard_bytes - compressed_bytes - footer_bytes,
        "shard_bytes_per_physical_rally": shard_bytes / physical_rallies,
        "footer_and_magic_bytes_per_shard": footer_bytes / len(manifest.shards),
        "max_metadata_row_group": ordered[0] if ordered else None,
        "configured_typed_record_admission_bytes": execution.max_record_bytes,
        "configured_typed_group_admission_bytes": execution.max_record_bytes
        * execution.row_group_rallies,
        "bounded_reload": reloaded,
        "local_reload_resource_snapshots": reload_sampler.report(),
        "reload_limits": {
            "groups": max_groups,
            "rows": max_rows,
            "typed_numeric_group_bytes": max_group_bytes,
        },
        "canonical_summary": summary,
        "notes": [
            "Parquet uncompressed metadata bytes are encoded column sizes, not decoded Python RAM.",
            "The sample ranks actual row groups by uncompressed metadata size. Numeric reload is bounded and is not a second full numeric audit.",
            "Python size estimates count retained row-group objects; allocator/workspace peaks remain part of the separate run resource measurement.",
        ],
    }


def validate_pilot(cfg, execution, drain):
    rallies = sum(
        getattr(cfg, f"{split}_rallies") for split in ("train", "val", "test")
    )
    if rallies > 512:
        raise ValueError(
            "pilot is limited to 512 physical rallies; main generation needs separate release"
        )
    if execution.workers not in (1, 2):
        raise ValueError("pilot permits one or two workers")
    if execution.repo_id or execution.revision or execution.evict:
        raise ValueError("pilot is local only; publication must be measured separately")
    if execution.drain_seconds > drain:
        raise ValueError("producer drain exceeds the wrapper drain allowance")
    return rallies


def actual_controls(output):
    from tennis3d.synthetic.producer_state import load

    result = {}
    for name in ("plan", "result", "performance", "timings"):
        path = Path(output) / "control" / f"{name}.json"
        result[name] = load(path) if path.exists() else None
    return result


def source_evidence(root, config, execution):
    from tennis3d.provenance import code_identity
    from tennis3d.synthetic.producer import (
        pool_source_identity,
        runtime_identity,
        source_identity,
    )

    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        text=True,
        timeout=10,
    )
    return {
        **code_identity(),
        "producer_source_sha256": source_identity(),
        "pool_source_sha256": pool_source_identity(),
        "runtime_versions": runtime_identity(),
        "python_executable": sys.executable,
        "wrapper_sha256": sha256(__file__),
        "uv_lock_sha256": sha256(root / "uv.lock"),
        "config_files": {str(path): sha256(path) for path in (config, execution)},
        "git_status_porcelain": status.splitlines(),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
        },
    }


def write_evidence(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--pool-cache", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=7200)
    parser.add_argument("--drain-seconds", type=float, default=45)
    parser.add_argument(
        "--sample-seconds", type=float, choices=(0.25, 0.5), default=0.5
    )
    parser.add_argument("--native-threads", type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    os.environ.update({name: str(args.native_threads) for name in THREAD_VARIABLES})
    root = Path(__file__).resolve().parent.parent
    # Direct script execution must import the actual checkout, not a different wheel.
    sys.path.insert(0, str(root))
    from tennis3d.config import load_config
    from tennis3d.synthetic.config import SimulationConfig
    from tennis3d.synthetic.producer_config import ProducerConfig

    if not 0 < args.drain_seconds < args.timeout_seconds <= 14400:
        parser.error("require 0 < drain < timeout <= 14400 seconds")
    if not 1 <= args.drain_seconds <= 300:
        parser.error("drain must be between 1 and 300 seconds")
    config, execution_path = args.config.resolve(), args.execution_config.resolve()
    output, evidence_path = args.output.absolute(), args.evidence.resolve()
    if output.is_symlink() or (args.pool_cache and args.pool_cache.is_symlink()):
        parser.error("output/pool paths cannot be symlinks")
    output = output.resolve()
    pool_cache = args.pool_cache.resolve() if args.pool_cache else output / "pools"
    if evidence_path.is_relative_to(output) or evidence_path.is_relative_to(pool_cache):
        parser.error("evidence must be outside the output and pool roots")
    log_path = evidence_path.with_suffix(".stdout.log")
    samples_path = evidence_path.with_suffix(".samples.jsonl")
    if any(path.exists() for path in (evidence_path, log_path, samples_path)):
        parser.error(
            "evidence/log/sample paths already exist; use a new evidence filename"
        )
    cfg = load_config(SimulationConfig, config)
    execution = load_config(ProducerConfig, execution_path)
    rallies = validate_pilot(cfg, execution, args.drain_seconds)
    if execution.workers * args.native_threads > 4:
        parser.error("workers times native threads cannot exceed four")
    env = os.environ.copy()
    env.update({name: str(args.native_threads) for name in THREAD_VARIABLES})
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        }
    )
    env["PYTHONPATH"] = str(root)
    child_argv = [
        sys.executable,
        "-m",
        "tennis3d",
        "simulate",
        "--format",
        "parquet",
        "--config",
        str(config),
        "--execution-config",
        str(execution_path),
        "--output",
        str(output),
        "--pool-cache",
        str(pool_cache),
    ]
    if args.resume:
        child_argv.append("--resume")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, Any] = {
        "schema_version": "producer-pilot-v1",
        "qualification": "local arithmetic/throughput proxy; not Azure qualification",
        "started_utc": utc_now(),
        "argv": child_argv,
        "cwd": str(root),
        "resolved_recipe": cfg.model_dump(),
        "resolved_execution": execution.model_dump(),
        "planned_physical_rallies": rallies,
        "thread_environment": {name: env[name] for name in THREAD_VARIABLES},
        "offline_environment": {
            name: env[name]
            for name in (
                "HF_HUB_OFFLINE",
                "HF_HUB_DISABLE_TELEMETRY",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN",
            )
        },
        "deadlines": {
            "hard_wall_seconds": args.timeout_seconds,
            "generation_seconds": args.timeout_seconds - args.drain_seconds,
            "drain_seconds": args.drain_seconds,
        },
        "sampling_interval_seconds": args.sample_seconds,
        "resource_roots": [str(output), str(pool_cache)],
        "publication": {
            "status": "not_measured",
            "upload": None,
            "anonymous_remote_readback": None,
            "verified_eviction": None,
        },
        "paid_admission": {
            "status": "blocked",
            "reason": "Azure generation and publication throughput and real deallocation need separate actual qualification",
        },
    }
    try:
        with cpu_lock(args.lock) as lock:
            evidence["lock"] = lock
            evidence["source_before"] = source_evidence(root, config, execution_path)
            evidence["run"] = run_bounded(
                child_argv,
                cwd=root,
                env=env,
                roots=(output, pool_cache),
                log_path=log_path,
                samples_path=samples_path,
                timeout=args.timeout_seconds,
                drain=args.drain_seconds,
                interval=args.sample_seconds,
            )
            evidence["controls"] = actual_controls(output)
            evidence["source_reported_state"] = (
                evidence["controls"]["result"] or {}
            ).get("state")
            if (
                evidence["run"]["returncode"] == 0
                and evidence["source_reported_state"] == "complete"
            ):
                evidence["parquet"] = inspect_parquet(output, execution)
            else:
                evidence["parquet"] = {
                    "status": "blocked",
                    "reason": "producer did not report successful complete output",
                }
            evidence["source_after"] = source_evidence(root, config, execution_path)
            identity_keys = (
                "code_revision",
                "source_sha256",
                "producer_source_sha256",
                "pool_source_sha256",
                "runtime_versions",
                "python_executable",
                "wrapper_sha256",
                "uv_lock_sha256",
                "config_files",
            )
            evidence["source_unchanged"] = all(
                evidence["source_before"][key] == evidence["source_after"][key]
                for key in identity_keys
            )
            plan = evidence["controls"]["plan"]
            evidence["source_matches_reported_plan"] = bool(
                plan
                and plan["source_sha256"]
                == evidence["source_before"]["producer_source_sha256"]
                and plan["pool_source_sha256"]
                == evidence["source_before"]["pool_source_sha256"]
                and plan["runtime_versions"]
                == evidence["source_before"]["runtime_versions"]
                and plan["recipe"] == cfg.model_dump()
                and plan["layout"] == execution.layout()
            )
    except Exception as error:  # noqa: BLE001 - retain evidence for any failed qualification stage
        evidence["wrapper_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    evidence["finished_utc"] = utc_now()
    evidence["logs"] = {
        str(path): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in (log_path, samples_path)
        if path.exists()
    }
    write_evidence(evidence_path, evidence)
    print(
        json.dumps(
            {
                "evidence": str(evidence_path),
                "source_reported_state": evidence.get("source_reported_state"),
                "returncode": evidence.get("run", {}).get("returncode"),
                "wrapper_error": evidence.get("wrapper_error"),
            }
        )
    )
    return (
        0
        if (
            "wrapper_error" not in evidence
            and evidence.get("source_unchanged")
            and evidence.get("source_matches_reported_plan")
            and evidence.get("parquet", {}).get("status") == "measured"
            and evidence.get("source_reported_state") == "complete"
            and evidence["run"]["returncode"] == 0
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
