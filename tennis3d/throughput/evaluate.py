"""Frozen independent benchmark driver, cache conditioning and parity gates."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tennis3d.throughput import campaign

RULES_PATH = "configs/throughput-main-v1/benchmark.json"


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    campaign._write(path, campaign._json(value), exclusive=True)


def worker(
    oracle: Path, source: Path, mode: str, request: dict, output: Path, timeout: int
):
    request = {**request, "source": str(source.resolve())}
    if mode in ("measure", "audit"):
        request["process_started"] = time.perf_counter()
    request_path = output.with_suffix(".request.json")
    write(request_path, request)
    env = {
        **os.environ,
        "PYTHONPATH": str(source.resolve()),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    }
    # Score clocks and RAM accounting belong to this oracle process, not a candidate.
    command = [
        sys.executable,
        str(oracle / "tennis3d/throughput/worker.py"),
        mode,
        str(request_path),
        str(output),
    ]
    started = request.get("process_started", time.perf_counter())
    deadline = time.perf_counter() + timeout
    boundaries, buffer, usage, exited = {}, b"", None, None
    child = None
    with output.with_suffix(".log").open("xb") as log:
        try:
            child = subprocess.Popen(
                command,
                cwd=source,
                env=env,
                stdout=subprocess.PIPE,
                stderr=log,
                start_new_session=True,
            )
            assert child.stdout is not None
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                while child.returncode is None or selector.get_map():
                    if time.perf_counter() >= deadline:
                        raise subprocess.TimeoutExpired(command, timeout)
                    for key, _ in selector.select(0.1):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        log.write(chunk)
                        buffer += chunk
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            if line.startswith(b"THROUGHPUT_BOUNDARY "):
                                name = line.removeprefix(
                                    b"THROUGHPUT_BOUNDARY "
                                ).decode("ascii")
                                if name in boundaries or name not in (
                                    "stream",
                                    "sustained",
                                    "complete",
                                ):
                                    raise ValueError(
                                        "invalid worker completion boundary"
                                    )
                                boundaries[name] = time.perf_counter()
                        if len(buffer) > 65536:
                            buffer = b""
                    if child.returncode is None:
                        pid, status, measured_usage = os.wait4(child.pid, os.WNOHANG)
                        if pid:
                            exited = time.perf_counter()
                            child.returncode = os.waitstatus_to_exitcode(status)
                            usage = measured_usage
            if child.returncode:
                raise subprocess.CalledProcessError(child.returncode, command)
        finally:
            if child is not None and child.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                _, status, usage = os.wait4(child.pid, 0)
                child.returncode = os.waitstatus_to_exitcode(status)
            if child is not None and child.stdout is not None:
                child.stdout.close()
    report = campaign._load(output)
    if mode in ("measure", "audit"):
        if (
            list(boundaries) != ["stream", "sustained", "complete"]
            or usage is None
            or exited is None
            or exited < boundaries["complete"]
        ):
            raise ValueError("worker did not complete the frozen stream boundaries")
        report.update(
            worker_elapsed_s=report["elapsed_s"],
            worker_stream_s=report["stream_elapsed_s"],
            worker_sustained_s=report["sustained_elapsed_s"],
            worker_ram_peak_bytes=report["ram_peak_bytes"],
        )
        report.update(
            elapsed_s=exited - started,
            stream_elapsed_s=exited - boundaries["stream"],
            sustained_elapsed_s=exited - boundaries["sustained"],
            completed_work_s=boundaries["complete"] - started,
            completed_stream_s=boundaries["complete"] - boundaries["stream"],
            completed_sustained_s=boundaries["complete"] - boundaries["sustained"],
            post_completion_s=exited - boundaries["complete"],
            ram_peak_bytes=usage.ru_maxrss * 1024,
            evaluator_process_wall_s=time.perf_counter() - started,
        )
        if "windows" in report:
            report["windows_per_second"] = report["windows"] / report["elapsed_s"]
        # Parent-measured report is durable and remains outside candidate imports.
        campaign._write(output, campaign._json(report))

    return report


def source_descriptor(oracle: Path, rules: dict, destination: Path, cache: Path):
    original = oracle / rules["descriptor"]
    if campaign._sha(campaign._read(original)) != rules["descriptor_sha256"]:
        raise ValueError("frozen main descriptor mismatch")
    raw = campaign._load(original)
    if (
        raw["revision"] != rules["revision"]
        or raw["manifest"]["sha256"] != rules["manifest_sha256"]
        or raw["split"] != "train"
        or raw["subset_sha256"] is not None
    ):
        raise ValueError("main training descriptor identity mismatch")
    raw["cache_dir"] = str(cache.resolve())
    # This is the only descriptor rewrite. Content identity is unchanged.
    write(destination, raw)
    return destination


def check_environment(report: dict, rules: dict):
    env = report["environment"]
    if (
        any(env[key] != rules[key] for key in ("gpu", "torch", "cuda"))
        or env["torch_threads"] != rules["threads"]
    ):
        raise ValueError("fixed GPU/software environment changed")
    if (
        report["ram_peak_bytes"] > rules["memory_bytes"]
        or report["cache_peak_bytes"] > rules["cache_bytes"]
    ):
        raise ValueError("fixed RAM/cache bound exceeded")
    if (
        report["pipeline"]["reserved_peak_bytes"] > rules["prefetch_bytes"]
        or report["pipeline"]["outstanding_peak_batches"] > rules["prefetch_depth"]
    ):
        raise ValueError("fixed prefetch bound exceeded")
    expected_steps = 0 if report["phase"] == "audit" else rules["steps"]
    if (
        report["steps"] != expected_steps
        or report["accepted_batches"] != rules["steps"]
        or report["windows"] != rules["steps"] * rules["batch_size"]
        or report["pipeline"]["generated_batches"] != rules["steps"]
    ):
        raise ValueError("wrong fixed amount of accepted optimizer work")
    if report["slurm_cpus"] != str(rules["threads"]) or report[
        "slurm_memory_mb"
    ] != str(rules["memory_bytes"] // 1024**2):
        raise ValueError("fixed Slurm CPU/RAM allocation changed")


def parity(report):
    return {
        k: report[k]
        for k in (
            "batch_sha256",
            "device_batch_sha256",
            "sampler_sha256",
            "rng_sha256",
            "draws",
            "rejected",
            "initial_state_sha256",
            "source_identity",
            "environment",
            "software",
        )
    }


def fixture_gates(oracle: Path, candidate: Path, output: Path, rules: dict):
    policy = {}
    for name, source in (("oracle", oracle), ("candidate", candidate)):
        policy[name] = worker(
            oracle,
            source,
            "policy",
            {"output_dir": str(output / name)},
            output / f"policy-{name}.json",
            rules["worker_timeout_seconds"],
        )
    if any(
        policy[name].get(key) is not True
        for name in policy
        for key in ("owned_cache", "bounded_reader")
    ):
        raise ValueError("independent source ownership/admission policy failed")
    reports = {}
    for name, source, extra in (
        ("maintained", oracle, {"maintained": True, "depth": 0}),
        ("oracle-loop", oracle, {"depth": 0}),
        ("candidate-loop", candidate, {"depth": rules["prefetch_depth"]}),
        (
            "candidate-resume",
            candidate,
            {"depth": rules["prefetch_depth"], "resume": True},
        ),
    ):
        reports[name] = worker(
            oracle,
            source,
            "fixture",
            {**extra, "output_dir": str(output / name)},
            output / f"{name}.json",
            rules["worker_timeout_seconds"],
        )
    expected = {
        k: reports["maintained"][k]
        for k in ("model", "ema", "optimizer", "sampler", "rng")
    }
    if any({k: report[k] for k in expected} != expected for report in reports.values()):
        raise ValueError("independent maintained-trainer update/resume parity failed")
    lifecycle = {}
    for name, source, depth in (
        ("oracle", oracle, 0),
        ("candidate", candidate, rules["prefetch_depth"]),
    ):
        lifecycle[name] = worker(
            oracle,
            source,
            "lifecycle",
            {"depth": depth, "output_dir": str(output / name)},
            output / f"lifecycle-{name}.json",
            rules["worker_timeout_seconds"],
        )
    if any(
        lifecycle["oracle"][key] != lifecycle["candidate"][key]
        for key in ("traces", "rng_sha256")
    ):
        raise ValueError("independent rejection/lifecycle/corruption parity failed")
    env = {
        **os.environ,
        "PYTHONPATH": str(candidate),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }
    # Fixed tests use retained views and faults. No simulator or generated arrays.
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-o",
        "addopts=",
        "-p",
        "no:cacheprovider",
        str(oracle / "tests/test_throughput_pipeline.py"),
        str(oracle / "tests/test_dataset_sources.py"),
        "-k",
        "not reader_rejects_group_before_decode",
    ]
    with (output / "lifecycle-tests.log").open("xb") as log:
        subprocess.run(
            command,
            cwd=candidate,
            env=env,
            stdout=log,
            stderr=log,
            check=True,
            timeout=rules["worker_timeout_seconds"],
        )
    return {
        "maintained_update_state": expected,
        "source_policy": policy,
        "lifecycle": lifecycle,
        "reports": reports,
        "lifecycle_tests_sha256": campaign._sha(
            campaign._read(output / "lifecycle-tests.log")
        ),
    }


def condition_cache(oracle: Path, rules: dict, cache: Path, artifacts: list[dict]):
    """One physical owned4GiB cache; restore declared inventory before a run."""
    from tennis3d.datasets.schema import Artifact
    from tennis3d.datasets.source import SourceSpec, VerifiedCache

    raw = campaign._load(oracle / rules["descriptor"])
    spec = SourceSpec.model_validate({**raw, "cache_dir": str(cache)})
    active = VerifiedCache(cache, spec)
    fetched = 0
    try:
        values = [Artifact.model_validate(v) for v in artifacts]
        desired = {active.path(v).name for v in values}
        for path in cache.iterdir():
            if path.name != "lock" and path.name not in desired:
                path.unlink()
        for value in values:
            path = active.path(value)
            if not path.exists() or campaign._sha(campaign._read(path)) != value.sha256:
                fetched += value.size_bytes
            active.get(value)
        if {p.name for p in cache.iterdir() if p.name != "lock"} != desired:
            raise ValueError("cache conditioning inventory changed")
        return fetched
    finally:
        active.close()


def cache_inventory(cache: Path):
    """Frozen parent observes a quiescent owned cache, excluding its lock."""
    result = []
    for path in sorted(cache.iterdir()):
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise ValueError("cache inventory contains an unowned path")
        if path.name != "lock":
            if len(path.name) != 64 or any(
                c not in "0123456789abcdef" for c in path.name
            ):
                raise ValueError("cache inventory contains an incomplete artifact")
            result.append({"key": path.name, "size_bytes": metadata.st_size})
    return result


def condition_proof(cache, template, workload, report, initial):
    final = cache_inventory(cache)
    initial_keys, final_keys = {v["key"] for v in initial}, {v["key"] for v in final}
    targets = template["target_inventory"]
    proof = {
        "condition": workload["condition"],
        "template_sha256": campaign._sha(campaign._json(template)),
        "initial_inventory": initial,
        "initial_bytes": sum(v["size_bytes"] for v in initial),
        "available_bytes": template["cache_bytes"]
        - sum(v["size_bytes"] for v in initial),
        "target_keys": [v["key"] for v in targets],
        "target_present": [v["key"] in initial_keys for v in targets],
        "final_inventory": final,
        "removed_initial_keys": sorted(initial_keys - final_keys),
        "removed_target_keys": [
            v["key"] for v in targets if v["key"] not in final_keys
        ],
        "read_target_keys": [
            v["key"] for v in targets if v["path"] in report["read_artifact_paths"]
        ],
        "fetched_target_keys": [
            v["key"] for v in targets if v["path"] in report["fetched_artifact_paths"]
        ],
    }
    if proof["read_target_keys"] != proof["target_keys"]:
        raise ValueError("measurement did not cover every declared target shard")
    if workload["condition"] == "evicted" and (
        any(proof["target_present"])
        or proof["fetched_target_keys"] != proof["target_keys"]
        or not (proof["removed_initial_keys"] or proof["removed_target_keys"])
    ):
        raise ValueError(
            "evicted measurement did not exercise cold reads and cache pressure"
        )
    return proof


def measurement(
    oracle,
    source,
    rules,
    prepared,
    workload,
    output,
    condition,
    expected=None,
    depth=None,
    auditing=False,
    audit_report=None,
):
    cache, artifacts = condition
    template = prepared["cache_templates"][workload["condition"]]
    conditioning_started = time.perf_counter()
    conditioning_bytes = condition_cache(oracle, rules, cache, artifacts)
    descriptor = source_descriptor(
        oracle, rules, output.with_suffix(".source.json"), cache
    )
    initial = cache_inventory(cache)
    if initial != [
        {"key": v["key"], "size_bytes": v["size_bytes"]} for v in template["inventory"]
    ]:
        raise ValueError("actual starting cache differs from frozen template")
    conditioning_s = time.perf_counter() - conditioning_started
    request = {
        "rules": rules,
        "workload": workload,
        "descriptor": str(descriptor),
        "data_identity": prepared["data_identity"],
        "initial_state": prepared["states"][workload["id"]],
        "depth": rules["prefetch_depth"] if depth is None else depth,
        "source_sha256": source.parent.name,
        "cache_template_sha256": campaign._sha(campaign._json(template)),
    }
    report = worker(
        oracle,
        source,
        "audit" if auditing else "measure",
        request,
        output,
        rules["worker_timeout_seconds"],
    )
    check_environment(report, rules)
    actual = (
        parity(report)
        if auditing
        else {
            key: report[key]
            for key in expected
            if key not in ("batch_sha256", "device_batch_sha256")
        }
        if expected is not None
        else None
    )
    comparison = (
        expected
        if auditing
        else {
            key: value
            for key, value in expected.items()
            if key not in ("batch_sha256", "device_batch_sha256")
        }
        if expected is not None
        else None
    )
    if expected is not None and actual != comparison:
        raise ValueError("actual main-data batch/RNG/sampler parity failed")
    if not auditing:
        if (
            audit_report is None
            or audit_report["source_sha256"] != report["source_sha256"]
            or audit_report["contract_sha256"] != report["contract_sha256"]
        ):
            raise ValueError("timing is not bound to its independent replay")
        report.update(
            audit=audit_report, audit_sha256=campaign._sha(campaign._json(audit_report))
        )
    proof = condition_proof(cache, template, workload, report, initial)
    enriched = {
        **report,
        "cache_conditioning_s": conditioning_s,
        "cache_conditioning_bytes": conditioning_bytes,
        "cache_condition": proof,
        "cache_condition_sha256": campaign._sha(campaign._json(proof)),
    }
    # Preserve completed checks even when a later process/allocation fails.
    campaign._write(output, campaign._json(enriched))
    return enriched


def prepare_caches(oracle: Path, rules: dict, output: Path):
    cache = output / "source-cache"
    descriptor = source_descriptor(oracle, rules, output / "prepare.source.json", cache)
    prepared = worker(
        oracle,
        oracle,
        "prepare",
        {"rules": rules, "descriptor": str(descriptor)},
        output / "prepared.json",
        1200,
    )
    templates = {
        condition: (cache, template["artifacts"])
        for condition, template in prepared["cache_templates"].items()
    }
    return prepared, templates, 0.0


def qualify(source: Path, output: Path):
    if output.exists():
        raise ValueError("qualification output must be new")
    output.mkdir(parents=True)
    inventory = campaign._inventory(source)
    identity = campaign._snapshot(output, source, inventory)
    oracle = output / "snapshots" / identity / "source"
    rules = campaign._load(oracle / RULES_PATH)
    write(output / "source-manifest.json", inventory)
    prepared, templates, warm_s = prepare_caches(oracle, rules, output)
    gates = fixture_gates(oracle, oracle, output / "gates", rules)
    qualification, expectations, reports, audits, execution_order = {}, {}, [], {}, []
    for workload in reversed(rules["workloads"]):
        # Independent main-data oracle trace for every accepted benchmark batch.
        reference = measurement(
            oracle,
            oracle,
            rules,
            prepared,
            workload,
            output / f"oracle-{workload['id']}.json",
            templates[workload["condition"]],
            depth=0,
            auditing=True,
        )
        if (
            workload["condition"] == "evicted"
            and not reference["cache_condition"]["removed_initial_keys"]
        ):
            raise ValueError("oracle replay did not demonstrate filler eviction")
        expectations[workload["id"]] = parity(reference)
        audits[workload["id"]] = reference
        pairs = []
        for round_index in range(rules["paired_rounds"]):
            pair = {}
            for side in ("a", "b")[:: 1 if round_index % 2 == 0 else -1]:
                report = measurement(
                    oracle,
                    oracle,
                    rules,
                    prepared,
                    workload,
                    output / f"baseline-{workload['id']}-{round_index}-{side}.json",
                    templates[workload["condition"]],
                    expectations[workload["id"]],
                    audit_report=reference,
                )
                if (
                    workload["condition"] == "evicted"
                    and not report["cache_condition"]["removed_initial_keys"]
                ):
                    raise ValueError(
                        "baseline did not demonstrate timed filler eviction"
                    )
                pair[side] = report["elapsed_s"]
                execution_order.append([workload["id"], round_index, side])
                reports.append(
                    {
                        "workload": workload["id"],
                        "round": round_index,
                        "side": side,
                        **report,
                    }
                )
            pairs.append([pair["a"], pair["b"]])
        qualification[workload["id"]] = pairs
    result = {
        "schema_version": 1,
        "source_sha256": identity,
        "benchmark_sha256": campaign._sha(campaign._read(oracle / RULES_PATH)),
        "descriptor_sha256": rules["descriptor_sha256"],
        "rules": rules,
        "prepared": prepared,
        "expectations": expectations,
        "audits": audits,
        "execution_order": execution_order,
        "qualification": qualification,
        "reports": reports,
        "fixture_gates": gates,
        "warm_conditioning_s": warm_s,
        "passed": True,
        "independent": True,
    }
    write(output / "qualification.json", result)
    protocol = {
        "schema_version": 1,
        "trial_budget": None,
        "trial_job_budget_s": rules["job_seconds"],
        "time_budget_s": None,
        "workloads": [w["id"] for w in rules["workloads"]],
        "min_paired_rounds": rules["paired_rounds"],
        "qualification": qualification,
        "noise_floor_fraction": 0.02,
        "benchmark": result,
    }
    write(output / "campaign-protocol.json", protocol)
    # Only metadata survives qualification; the single source cache is disposable.
    shutil.rmtree(output / "source-cache")
    return {
        "passed": True,
        "source_sha256": identity,
        "qualification": qualification,
        "protocol": str(output / "campaign-protocol.json"),
    }


def evaluate(run: Path, trial_id: str, cache_root: Path):
    import tennis3d

    if Path(tennis3d.__file__).resolve().parent != run.resolve() / "oracle/tennis3d":
        raise ValueError("evaluator imports must come from the frozen oracle")
    job_started = time.perf_counter()
    # State validation binds all frozen files and the active reservation first.
    with campaign._lock(run):
        state = campaign._state(run)
        if state["active"] != trial_id:
            raise ValueError("evaluate requires the recorded active trial")
        reserved = state["trials"][trial_id]
        protocol = state["protocol"]
        initial = state["initial"]
    oracle = run / "oracle"
    candidate = run / "snapshots" / reserved["source_sha256"] / "source"
    incumbent = run / "snapshots" / reserved["incumbent_sha256"] / "source"
    output = run / "trials" / trial_id
    identity = {
        k: ({**initial, **reserved})[k]
        for k in (
            "trial_id",
            "source_sha256",
            "incumbent_sha256",
            "oracle_sha256",
            "protocol_sha256",
        )
    }
    benchmark = protocol["benchmark"]
    rules = campaign._load(oracle / RULES_PATH)
    if (
        rules != benchmark["rules"]
        or campaign._sha(campaign._read(oracle / RULES_PATH))
        != benchmark["benchmark_sha256"]
    ):
        raise ValueError("qualification benchmark rules changed")
    if cache_root.exists():
        raise ValueError("trial cache root must be fresh and owned")
    cache_root.mkdir(parents=True)
    result = {**identity, "status": "failed", "measurements": []}
    evidence: dict[str, Any] = {**identity, "passed": False, "independent": True}
    try:
        evidence["fixture_gates"] = fixture_gates(
            oracle, candidate, output / "gates", rules
        )
        prepared, templates, warm_s = prepare_caches(oracle, rules, cache_root)
        if (
            prepared["data_identity"] != benchmark["prepared"]["data_identity"]
            or prepared["states"] != benchmark["prepared"]["states"]
        ):
            raise ValueError("frozen workload initial state changed")
        evidence["conditioning"] = {"eviction": prepared, "warm_s": warm_s}
        audit_reports = {}
        evidence["execution_order"] = []
        for workload in reversed(rules["workloads"]):
            for label, source in (("incumbent", incumbent), ("candidate", candidate)):
                path = output / f"audit-{workload['id']}-{label}.json"
                audit_reports[(workload["id"], label)] = measurement(
                    oracle,
                    source,
                    rules,
                    prepared,
                    workload,
                    path,
                    templates[workload["condition"]],
                    benchmark["expectations"][workload["id"]],
                    auditing=True,
                )
            for r in range(rules["paired_rounds"]):
                pair = {}
                # AB then BA on one node; both are fresh seed/state/cache copies.
                for label, source in (
                    ("incumbent", incumbent),
                    ("candidate", candidate),
                )[:: 1 if r % 2 == 0 else -1]:
                    evidence["execution_order"].append([workload["id"], r, label])
                    path = output / f"{workload['id']}-{r}-{label}.json"
                    pair[label] = measurement(
                        oracle,
                        source,
                        rules,
                        prepared,
                        workload,
                        path,
                        templates[workload["condition"]],
                        benchmark["expectations"][workload["id"]],
                        audit_report=audit_reports[(workload["id"], label)],
                    )
                result["measurements"].append(
                    {
                        "workload": workload["id"],
                        "round": r,
                        "incumbent_elapsed_s": pair["incumbent"]["elapsed_s"],
                        "candidate_elapsed_s": pair["candidate"]["elapsed_s"],
                        "incumbent_stream_s": pair["incumbent"]["stream_elapsed_s"],
                        "candidate_stream_s": pair["candidate"]["stream_elapsed_s"],
                        "incumbent_sustained_s": pair["incumbent"][
                            "sustained_elapsed_s"
                        ],
                        "candidate_sustained_s": pair["candidate"][
                            "sustained_elapsed_s"
                        ],
                    }
                )
                evidence.setdefault("reports", []).append(
                    {"workload": workload["id"], "round": r, **pair}
                )
        evidence["passed"] = True
        result["status"] = "completed"
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        # Avoid exception strings that may contain signed transfer URLs/secrets.
        evidence["failure_type"] = type(error).__name__
    finally:
        result["job_accounting"] = {
            "compute_s": time.perf_counter() - job_started,
            "basis": "driver",
            "job_id": os.environ.get("SLURM_JOB_ID"),
        }
        write(output / "evidence.json", evidence)
        result["gates"] = {
            "passed": evidence["passed"],
            "independent": True,
            "evidence_file": "evidence.json",
            "evidence_sha256": campaign._sha(campaign._read(output / "evidence.json")),
        }
        write(output / "measurement.json", result)
        if cache_root.exists():
            shutil.rmtree(cache_root)
    return {"passed": evidence["passed"], "result": str(output / "measurement.json")}


def main():
    def interrupted(_signal, _frame):
        raise InterruptedError("bounded evaluator stopped for evidence sync")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    qual = subs.add_parser("qualify")
    qual.add_argument("--source", type=Path, required=True)
    qual.add_argument("--output", type=Path, required=True)
    trial = subs.add_parser("trial")
    trial.add_argument("--run", type=Path, required=True)
    trial.add_argument("--trial", required=True)
    trial.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "qualify":
        result = qualify(args.source.resolve(), args.output.resolve())
    else:
        result = evaluate(args.run.resolve(), args.trial, args.cache_root.resolve())
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        parser.exit(1)


if __name__ == "__main__":
    main()
