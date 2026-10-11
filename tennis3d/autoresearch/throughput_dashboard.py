"""Read small throughput envelopes without importing campaign execution code."""

import hashlib
import json
import math
import time
from pathlib import Path

from tennis3d.throughput.campaign import _score


class SnapshotError(ValueError):
    pass


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def public(value):
    """Diagnostics are opt-in; environment paths and arbitrary errors stay private."""
    names = {
        "setup_s",
        "stream_s",
        "stream_elapsed_s",
        "sustained_elapsed_s",
        "accepted_batches",
        "cache_conditioning_s",
        "cache_conditioning_bytes",
        "sustained_s",
        "elapsed_s",
        "windows_per_s",
        "windows_per_second",
        "stream_windows_per_second",
        "sustained_windows_per_second",
        "model_setup_s",
        "data_setup_s",
        "queue_wait_mean_s",
        "queue_wait_p95_s",
        "gpu_allocated_peak_bytes",
        "gpu_reserved_peak_bytes",
        "stream_windows_per_s",
        "sustained_windows_per_s",
        "steps",
        "windows",
        "loader_stall_mean_s",
        "loader_stall_p95_s",
        "stall_mean_s",
        "stall_p95_s",
        "queue_delay_mean_s",
        "ram_peak_bytes",
        "gpu_peak_bytes",
        "cache_peak_bytes",
        "bytes_fetched",
        "node_load",
        "slurm_cpus",
        "slurm_memory_mb",
        "pipeline",
        "reserved_peak_bytes",
        "outstanding_peak_batches",
        "generated_batches",
        "conditioning_s",
        "shutdown_s",
        "drain_s",
        "diagnostics_s",
    }
    values = {
        k: public(v) if isinstance(v, dict) else v
        for k, v in value.items()
        if k in names and isinstance(v, (dict, int, float, bool))
    }
    for key in ("slurm_cpus", "slurm_memory_mb", "slurm_job_id"):
        item = value.get(key)
        if isinstance(item, str) and item.isascii() and item.isdecimal():
            values[key] = item
    load = value.get("loadavg")
    if isinstance(load, list) and all(
        type(v) in (int, float) and math.isfinite(v) for v in load
    ):
        values["loadavg"] = load
    return values


def ratios(measurements):
    result = {}
    for metric in ("elapsed_s", "stream_s", "sustained_s"):
        groups = {}
        for pair in measurements:
            a, b = pair.get(f"candidate_{metric}"), pair.get(f"incumbent_{metric}")
            if (
                isinstance(a, (int, float))
                and isinstance(b, (int, float))
                and a > 0
                and b > 0
            ):
                groups.setdefault(pair["workload"], []).append(math.log(a / b))
        if groups:
            workloads = {k: math.exp(sum(v) / len(v)) for k, v in groups.items()}
            result[metric] = {
                "workloads": workloads,
                "geometric_ratio": math.exp(
                    sum(map(math.log, workloads.values())) / len(workloads)
                ),
            }
    return result


class ThroughputHistory:
    def __init__(self, run: Path, *, archived: bool):
        self.run = run.resolve()
        self.archived = archived
        self.source = digest(str(self.run).encode())
        self.segment = f"throughput:{self.source}"
        self.cache = {}
        self.canonical_hashes = {}
        self.last = None

    def read(self, relative):
        path = self.run / relative
        if not path.resolve().is_relative_to(self.run) or path.is_symlink():
            raise SnapshotError("Unsafe artifact")
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        if self.cache.get(relative, (None,))[0] == key:
            return self.cache[relative][1:]
        raw = path.read_bytes()
        after = path.stat()
        if (after.st_mtime_ns, after.st_size, after.st_ino) != key:
            raise SnapshotError("Artifact changed during read")
        value = json.loads(raw) if relative != "journal.jsonl" else None
        if value is not None:
            self.canonical_hashes[relative] = digest(canonical(value))
        self.cache[relative] = (key, raw, value)
        return raw, value

    def empty(self, status, warning):
        return {
            "segments": [
                {
                    "segment": self.segment,
                    "segment_name": "Throughput search",
                    "kind": "throughput",
                    "profile": "main-data throughput",
                    "archived": self.archived,
                    "status": status,
                    "source_run": self.source,
                }
            ],
            "trials": [],
            "best": {},
            "stopped": False,
            "server_time": time.time(),
            "last_update": None,
            "warnings": [warning],
        }

    def view(self):
        try:
            state = self.load()
            self.last = state
            return state
        except (
            ValueError,
            OSError,
            KeyError,
            TypeError,
            OverflowError,
            AttributeError,
        ):
            if self.last:
                # No newly observed partial decision is ever presented as kept.
                return {
                    **self.last,
                    "server_time": time.time(),
                    "segments": [
                        {**s, "status": "stale/incomplete"}
                        for s in self.last["segments"]
                    ],
                    "warnings": [
                        "Throughput snapshot stale/incomplete: journal or required envelopes unavailable or inconsistent"
                    ],
                }
            return self.empty(
                "incomplete" if self.run.exists() else "not initialized",
                "Throughput snapshot unavailable: waiting for a complete local mirror",
            )

    def load(self):
        raw, _ = self.read("journal.jsonl")
        if not raw.endswith(b"\n"):
            raise SnapshotError("Torn journal")
        entries = []
        for line in raw.splitlines():
            entry = json.loads(line)
            sha = entry.pop("sha256")
            if (
                entry["sequence"] != len(entries)
                or entry["previous"] != (entries[-1]["sha256"] if entries else "0" * 64)
                or digest(canonical(entry)) != sha
            ):
                raise SnapshotError("Journal chain mismatch")
            entries.append({**entry, "sha256": sha})
        if not entries or entries[0]["kind"] != "initialize":
            raise SnapshotError("Missing initialization")
        initial = entries[0]["payload"]
        _, cursor = self.read("cursor.json")
        if cursor != {"sequence": len(entries) - 1, "sha256": entries[-1]["sha256"]}:
            raise SnapshotError("Cursor mismatch")
        _, protocol = self.read("protocol.json")
        if self.canonical_hashes["protocol.json"] != initial["protocol_sha256"]:
            raise SnapshotError("Protocol changed")
        best = {
            "trial_id": None,
            "source_sha256": initial["oracle_sha256"],
            "result_sha256": None,
        }
        trials, active = {}, None
        for entry in entries[1:]:
            kind, payload = entry["kind"], entry["payload"]
            identifier = payload["trial_id"]
            if kind == "trial":
                if (
                    active
                    or identifier in trials
                    or not isinstance(identifier, str)
                    or not identifier.isdecimal()
                    or len(identifier) != 6
                    or payload["incumbent_sha256"] != best["source_sha256"]
                ):
                    raise SnapshotError("Invalid reservation")
                _, reservation = self.read(f"trials/{identifier}/reservation.json")
                if reservation != payload:
                    raise SnapshotError("Reservation mismatch")
                trials[identifier] = {
                    **payload,
                    "trial": identifier,
                    "segment": self.segment,
                    "kind": "throughput",
                    "source_run": self.source,
                    "archived": self.archived,
                    "status": "active",
                    "reserved_at": payload.get("reserved_at"),
                    "hypothesis": payload.get("description")
                    or "Description not recorded",
                    "incumbent_trial": best["trial_id"],
                    "measurements": [],
                    "ratios": {},
                }
                active = identifier
            elif kind == "record":
                if identifier != active or "result_sha256" in trials[identifier]:
                    raise SnapshotError("Invalid record")
                trial = trials[identifier]
                result_raw, result = self.read(f"trials/{identifier}/result.json")
                if digest(result_raw) != payload["result_sha256"] or any(
                    result.get(k) != ({**initial, **trial})[k]
                    for k in (
                        "trial_id",
                        "source_sha256",
                        "incumbent_sha256",
                        "oracle_sha256",
                        "protocol_sha256",
                    )
                ):
                    raise SnapshotError("Result identity mismatch")
                gates = result["gates"]
                if gates["evidence_file"] != "evidence.json":
                    raise SnapshotError("Unexpected evidence file")
                evidence_raw, evidence = self.read(f"trials/{identifier}/evidence.json")
                if (
                    digest(evidence_raw) != gates["evidence_sha256"]
                    or any(
                        evidence.get(k) != result.get(k)
                        for k in (
                            "trial_id",
                            "source_sha256",
                            "incumbent_sha256",
                            "oracle_sha256",
                            "protocol_sha256",
                        )
                    )
                    or any(
                        type(gates[k]) is not bool or evidence.get(k) is not gates[k]
                        for k in ("passed", "independent")
                    )
                ):
                    raise SnapshotError("Evidence mismatch")
                measurements = result.get("measurements", [])
                if result["status"] == "completed":
                    expected = {
                        (w, r)
                        for w in protocol["workloads"]
                        for r in range(protocol["min_paired_rounds"])
                    }
                    if {
                        (p["workload"], p["round"]) for p in measurements
                    } != expected or len(measurements) != len(expected):
                        raise SnapshotError("Incomplete measurements")
                safe_pairs = [
                    {
                        k: v
                        for k, v in p.items()
                        if k in ("workload", "round")
                        or k
                        in {
                            f"{side}_{metric}"
                            for side in ("candidate", "incumbent")
                            for metric in ("elapsed_s", "stream_s", "sustained_s")
                        }
                    }
                    for p in measurements
                ]
                if result["status"] not in ("completed", "failed"):
                    raise SnapshotError("Invalid result status")
                reports = {
                    (p["workload"], p["round"]): p for p in evidence.get("reports", [])
                }
                for pair in measurements:
                    report = reports.get((pair["workload"], pair["round"]), {})
                    for side in ("candidate", "incumbent"):
                        for metric in ("elapsed_s", "stream_s", "sustained_s"):
                            value = pair.get(f"{side}_{metric}")
                            if value is not None and (
                                type(value) not in (int, float)
                                or not math.isfinite(value)
                                or value <= 0
                                or report.get(side, {}).get(
                                    {
                                        "stream_s": "stream_elapsed_s",
                                        "sustained_s": "sustained_elapsed_s",
                                    }.get(metric, metric)
                                )
                                != value
                            ):
                                raise SnapshotError("Paired timing evidence mismatch")
                score = _score(protocol, result)
                if result.get("score") != score:
                    raise SnapshotError("Sealed score mismatch")
                trial["recorded_result"] = result
                trial.update(
                    result_sha256=payload["result_sha256"],
                    status="failed" if result["status"] == "failed" else "completed",
                    recorded_at=result.get("recorded_at"),
                    score={
                        k: v
                        for k, v in result.get("score", {}).items()
                        if k
                        in (
                            "metric",
                            "eligible",
                            "geometric_stream_ratio",
                            "geometric_sustained_ratio",
                            "geometric_elapsed_ratio",
                            "per_workload_elapsed_ratio",
                            "per_round_stream_ratio",
                            "per_round_sustained_ratio",
                            "per_workload_stream_ratio",
                            "per_workload_sustained_ratio",
                            "noise_threshold_fraction",
                            "sustained_noise_threshold_fraction",
                            "per_workload_total_noise_fraction",
                            "per_workload_sustained_noise_fraction",
                        )
                    },
                    job_accounting={
                        k: v
                        for k, v in result.get("job_accounting", {}).items()
                        if k in ("basis", "compute_s", "queue_s", "job_id")
                    },
                    gates={k: gates[k] for k in ("passed", "independent")},
                    measurements=safe_pairs,
                    ratios=ratios(safe_pairs),
                    diagnostics=[
                        {
                            "workload": p["workload"],
                            "round": p["round"],
                            **{
                                side: public(p.get(side, {}))
                                for side in ("candidate", "incumbent")
                            },
                        }
                        for p in evidence.get("reports", [])
                    ],
                )
            elif kind in ("retain", "discard"):
                if identifier != active:
                    raise SnapshotError("Invalid decision")
                trial = trials[identifier]
                if kind == "retain":
                    if (
                        trial["status"] != "completed"
                        or trial["gates"] != {"passed": True, "independent": True}
                        or not _score(protocol, trial["recorded_result"])["eligible"]
                    ):
                        raise SnapshotError("Retained without passing gates")
                    best = {
                        "trial_id": identifier,
                        "source_sha256": trial["source_sha256"],
                        "result_sha256": trial["result_sha256"],
                    }
                trial["decision"] = kind
                trial["status"] = (
                    "kept"
                    if kind == "retain"
                    else "failed"
                    if trial["status"] == "failed"
                    else "discarded"
                )
                active = None
            else:
                raise SnapshotError("Unknown journal event")
        _, pointer = self.read("best.json")
        if pointer != best:
            raise SnapshotError("Best pointer mismatch")
        if self.read("journal.jsonl")[0] != raw:
            raise SnapshotError("Journal changed during snapshot")
        rules = protocol.get("benchmark", {}).get("rules", {})
        updated = (self.run / "journal.jsonl").stat().st_mtime
        segment = {
            "segment": self.segment,
            "segment_name": "Throughput search",
            "kind": "throughput",
            "profile": "main-data throughput",
            "archived": self.archived,
            "source_run": self.source,
            "status": "active reservation" if active else "idle",
            "active_trial": active,
            "last_update": updated,
            "objective": "Minimize paired main-data pipeline seconds while independent parity gates pass",
            "best": best,
            "protocol_sha256": initial["protocol_sha256"],
            "oracle_sha256": initial["oracle_sha256"],
            "data": {
                k: rules[k]
                for k in (
                    "revision",
                    "descriptor_sha256",
                    "steps",
                    "batch_size",
                    "context",
                    "max_frames",
                    "sustained_steps",
                    "sustained_start_step",
                    "steady_skip_steps",
                    "manifest_sha256",
                    "membership_sha256",
                    "paired_rounds",
                    "threads",
                    "memory_bytes",
                    "gpu",
                    "torch",
                    "cuda",
                )
                if k in rules
            },
            "workloads": protocol["workloads"],
            "qualification": protocol.get("qualification", {}),
            "guards": {
                k: v
                for k, v in protocol.items()
                if k
                in (
                    "noise_threshold_fraction",
                    "per_workload_regression_fraction",
                    "per_workload_noise_threshold_fraction",
                    "selection_metric",
                    "sustained_noise_threshold_fraction",
                    "per_workload_total_noise_fraction",
                    "per_workload_sustained_noise_fraction",
                )
            },
        }
        return {
            "segments": [segment],
            "trials": [
                {k: v for k, v in t.items() if k != "recorded_result"}
                for t in trials.values()
            ],
            "best": {self.segment: {**best, "trial": best["trial_id"]}},
            "server_time": time.time(),
            "last_update": updated,
            "stopped": False,
            "warnings": [],
        }

    def details(self, trial):
        return {
            "trial": trial,
            "curve": [],
            "measurements": trial["measurements"],
            "diagnostics": trial.get("diagnostics", []),
        }
