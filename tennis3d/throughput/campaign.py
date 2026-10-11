"""Fixed throughput campaign state and source capture. No training runs occur here.

All writes use a process lock. Journal and cursor disagreements fail closed rather
than refunding reservations or repeating an interrupted trial. Results bind an
independent evaluator's evidence to the reserved candidate and incumbent sources.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EDITABLE = frozenset(
    {
        "tennis3d/lifting/data.py",
        "tennis3d/lifting/features.py",
        "tennis3d/synthetic/corrupt.py",
        "tennis3d/datasets/source.py",
        "tennis3d/datasets/cache.py",
        "tennis3d/throughput/pipeline.py",
    }
)
ZERO = "0" * 64
SCORE_METRIC = "stream_s_with_sustained_gate"


class CampaignError(ValueError):
    """A campaign cannot safely proceed."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _regular(path: Path) -> None:
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise CampaignError(f"Symlink is forbidden: {parent}")
    if not path.is_file():
        raise CampaignError(f"Missing regular file: {path}")


def _read(path: Path) -> bytes:
    _regular(path)
    return path.read_bytes()


def _load(path: Path) -> Any:
    try:
        return json.loads(_read(path))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise CampaignError(f"Invalid JSON: {path}") from exc


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write(path: Path, data: bytes, *, exclusive: bool = False) -> None:
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise CampaignError(f"Symlink is forbidden: {parent}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    _sync_dir(path.parent)


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(p in (".", "..") for p in path.parts)
    ):
        raise CampaignError(f"Unsafe relative path: {value}")
    return path


def _git(source: Path, *args: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source), *args], stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as exc:
        raise CampaignError("Source must be a Git worktree") from exc


def _inventory(source: Path) -> dict[str, dict[str, Any]]:
    if source.is_symlink():
        raise CampaignError("Source cannot be a symlink")
    root = _git(source, "rev-parse", "--show-toplevel").decode().strip()
    if Path(root).resolve() != source.resolve():
        raise CampaignError("Source must be the Git worktree root")
    tracked = set(_git(source, "ls-files", "-z").decode().split("\0")) - {""}
    untracked = set(
        _git(source, "ls-files", "--others", "--exclude-standard", "-z")
        .decode()
        .split("\0")
    ) - {""}
    if untracked - EDITABLE:
        raise CampaignError(
            f"Undeclared untracked files: {sorted(untracked - EDITABLE)}"
        )
    # Ignored Python files can otherwise bypass the tracked-source contract.
    for path in (source / "tennis3d").rglob("*.py"):
        name = path.relative_to(source).as_posix()
        if name not in tracked and name not in EDITABLE:
            raise CampaignError(f"Undeclared Python module: {name}")
    inventory = {}
    for name in sorted(
        tracked | untracked | {p for p in EDITABLE if (source / p).exists()}
    ):
        path = source / _safe_relative(name)
        for parent in (path, *path.parents):
            if parent.is_symlink():
                raise CampaignError(f"Symlink is forbidden: {parent}")
        if not path.exists() and name in EDITABLE:
            continue
        data = _read(path)
        inventory[name] = {
            "sha256": _sha(data),
            "executable": bool(path.stat().st_mode & stat.S_IXUSR),
        }
    return inventory


def _snapshot(run: Path, source: Path, manifest: dict[str, Any]) -> str:
    digest = _sha(_json(manifest))
    destination = run / "snapshots" / digest
    if destination.exists():
        _verify_snapshot(run, digest)
        return digest
    destination.mkdir(parents=True)
    _write(destination / "manifest.json", _json(manifest), exclusive=True)
    for name, identity in manifest.items():
        data = _read(source / name)
        if _sha(data) != identity["sha256"]:
            raise CampaignError("Source changed during capture")
        target = destination / "source" / name
        _write(target, data, exclusive=True)
        target.chmod(0o555 if identity["executable"] else 0o444)
    _sync_dir(destination)
    return digest


def _verify_tree(directory: Path, manifest: dict[str, Any]) -> None:
    actual = {
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.is_file() or p.is_symlink()
    }
    if actual != set(manifest):
        raise CampaignError(f"Snapshot file inventory mismatch: {directory}")
    for name, identity in manifest.items():
        path = directory / _safe_relative(name)
        if (
            _sha(_read(path)) != identity["sha256"]
            or bool(path.stat().st_mode & stat.S_IXUSR) != identity["executable"]
        ):
            raise CampaignError(f"Snapshot content mismatch: {name}")


def _verify_snapshot(run: Path, digest: str) -> dict[str, Any]:
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise CampaignError("Invalid source SHA256")
    directory = run / "snapshots" / digest
    manifest = _load(directory / "manifest.json")
    if _sha(_json(manifest)) != digest:
        raise CampaignError("Source manifest hash mismatch")
    _verify_tree(directory / "source", manifest)
    return manifest


def _positive(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise CampaignError("Expected a positive finite number")
    return float(value)


BENCHMARK_PATH = "configs/throughput-main-v1/benchmark.json"
MAIN_REVISION = "c2e782053694cfde305362e670ebbb4f4e70884c"
MAIN_DESCRIPTOR_SHA256 = (
    "a6aa8ba11adda8b6a5f6bf39277de1f73dd24338db80907a4dce8a3d293e27a1"
)
PARITY_FIELDS = (
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


def _digest_string(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise CampaignError("Expected SHA256 identity")
    return value


def _state_digest(value: Any) -> str:
    """Match the frozen worker's digest for JSON sampler state, without imports."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
        elif isinstance(item, list):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
            digest.update(b"\0")

    visit(value)
    return digest.hexdigest()


def _fixture_metadata(gates: dict[str, Any]) -> None:
    policy = gates.get("source_policy", {})
    if any(
        policy.get(side, {}).get(key) is not True
        for side in ("oracle", "candidate")
        for key in ("owned_cache", "bounded_reader")
    ):
        raise CampaignError("Independent owned-cache/bounded-reader policy gate failed")
    expected = gates.get("maintained_update_state")
    if not isinstance(expected, dict) or set(expected) != {
        "model",
        "ema",
        "optimizer",
        "sampler",
        "rng",
    }:
        raise CampaignError("Missing independent fixture update-state evidence")
    for value in expected.values():
        _digest_string(value)
    reports = gates.get("reports", {})
    for name in ("maintained", "oracle-loop", "candidate-loop", "candidate-resume"):
        report = reports.get(name, {})
        if {key: report.get(key) for key in expected} != expected:
            raise CampaignError("Independent fixture update-state parity failed")
    _digest_string(gates.get("lifecycle_tests_sha256"))
    lifecycle = gates.get("lifecycle", {})
    oracle, candidate = lifecycle.get("oracle", {}), lifecycle.get("candidate", {})
    expected_counts = {
        "ordinary_epochs": 8,
        "rejections": 8,
        "checkpoint_stop_resume": 6,
        "failure_resume": 6,
        "inflight_stop_resume": 6,
        "complete_view_corruption": 3,
    }
    traces = oracle.get("traces", {})
    if (
        set(traces) != set(expected_counts)
        or candidate.get("traces") != traces
        or candidate.get("rng_sha256") != oracle.get("rng_sha256")
    ):
        raise CampaignError("Independent lifecycle trace parity failed")
    _digest_string(oracle.get("rng_sha256"))
    for name, count in expected_counts.items():
        if not isinstance(traces[name], list) or len(traces[name]) != count:
            raise CampaignError("Independent lifecycle trace count changed")
        for pair in traces[name]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise CampaignError("Independent lifecycle trace is invalid")
            for value in pair:
                _digest_string(value)


def _artifact_rows(
    prepared: dict[str, Any], descriptor: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    def row(value):
        artifact = {key: value[key] for key in ("path", "sha256", "size_bytes")}
        _safe_relative(artifact["path"])
        _digest_string(artifact["sha256"])
        if type(artifact["size_bytes"]) is not int or artifact["size_bytes"] <= 0:
            raise CampaignError("Invalid source artifact size")
        key = _sha(
            _json(
                {
                    "repo": descriptor["repo_id"],
                    "commit": descriptor["revision"],
                    **artifact,
                }
            )[:-1]
        )
        return {"key": key, **artifact}

    manifest = row(descriptor["manifest"])
    shards = [row(value) for value in prepared.get("shards", [])]
    if (
        not shards
        or len({value["key"] for value in [manifest, *shards]}) != len(shards) + 1
        or len({value["path"] for value in [manifest, *shards]}) != len(shards) + 1
    ):
        raise CampaignError("Missing or duplicated source artifact identity")
    return manifest, shards


def _cache_templates(
    prepared: dict[str, Any], rules: dict[str, Any], descriptor: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, shards = _artifact_rows(prepared, descriptor)
    by_key = {value["key"]: value for value in [manifest, *shards]}
    cap = rules["cache_bytes"]
    if type(cap) is not int or cap != descriptor["cache_bytes"]:
        raise CampaignError("Source and benchmark cache capacities differ")
    definitions = {value["condition"]: value for value in rules["workloads"]}
    if len(definitions) != 2 or set(definitions) != {"warm", "evicted"}:
        raise CampaignError("Benchmark needs one warm and one evicted workload")
    indices = prepared.get("shard_indices", {})
    if set(indices) != {value["id"] for value in definitions.values()}:
        raise CampaignError("Missing qualified workload shard indices")
    targets = {}
    for condition, workload in definitions.items():
        values = indices[workload["id"]]
        if (
            not isinstance(values, list)
            or not values
            or any(
                type(index) is not int or not 0 <= index < len(shards)
                for index in values
            )
            or len(set(values)) != len(values)
        ):
            raise CampaignError("Invalid or duplicated workload target shard")
        targets[condition] = [shards[index] for index in values]
    stress = prepared.get("stress_shards", [])
    if (
        not isinstance(stress, list)
        or len(stress) != rules["eviction_shards"]
        or any(
            type(index) is not int or not 0 <= index < len(shards) for index in stress
        )
        or len(set(stress)) != len(stress)
    ):
        raise CampaignError("Invalid real-shard eviction stress identity")
    if manifest["size_bytes"] + sum(
        shards[index]["size_bytes"] for index in stress
    ) <= cap or not prepared.get("evicted_keys"):
        raise CampaignError(
            "Qualification did not establish real cache eviction pressure"
        )
    keys = prepared.get("cache_keys", [])
    stress_keys = {shards[index]["key"] for index in stress}
    if (
        not isinstance(keys, list)
        or len(set(keys)) != len(keys)
        or manifest["key"] not in keys
        or not set(keys) <= stress_keys | {manifest["key"]}
    ):
        raise CampaignError("Invalid verified stress-survivor cache inventory")
    if sum(by_key[key]["size_bytes"] for key in keys) > cap:
        raise CampaignError("Stress-survivor cache exceeded its byte capacity")
    target_keys = {value["key"] for value in targets["evicted"]}
    if target_keys & set(keys):
        raise CampaignError("Evicted workload targets survived cache stress")
    survivors = sorted(
        (
            by_key[key]
            for key in keys
            if key != manifest["key"] and key not in target_keys
        ),
        key=lambda value: (-value["size_bytes"], value["path"]),
    )
    selected, initial = [manifest], manifest["size_bytes"]
    target_bytes = sum(value["size_bytes"] for value in targets["evicted"])
    for value in survivors:
        if initial + target_bytes > cap:
            break
        selected.append(value)
        initial += value["size_bytes"]
    if initial > cap or initial + target_bytes <= cap:
        raise CampaignError(
            "Stress survivors cannot form the frozen minimal-pressure template"
        )
    templates = {}
    for condition, initial_rows in (
        ("warm", [manifest, *targets["warm"]]),
        ("evicted", selected),
    ):
        initial_bytes = sum(value["size_bytes"] for value in initial_rows)
        if initial_bytes > cap:
            raise CampaignError("Initial cache template exceeds capacity")
        targets_bytes = sum(value["size_bytes"] for value in targets[condition])
        templates[condition] = {
            "artifacts": [
                {key: value[key] for key in ("path", "sha256", "size_bytes")}
                for value in initial_rows
            ],
            "inventory": sorted(initial_rows, key=lambda value: value["key"]),
            "cache_bytes": cap,
            "initial_bytes": initial_bytes,
            "available_bytes": cap - initial_bytes,
            "target_inventory": targets[condition],
            "target_bytes": targets_bytes,
            "targets_present": condition == "warm",
            "overflow_bytes": max(0, initial_bytes + targets_bytes - cap)
            if condition == "evicted"
            else 0,
        }
    if prepared.get("cache_templates") != templates:
        raise CampaignError(
            "Qualified cache template differs from source-derived minimal pressure"
        )
    return templates, by_key


def _cache_condition(
    report: dict[str, Any],
    template: dict[str, Any],
    condition: str,
    artifacts: dict[str, Any],
    *,
    baseline: bool,
) -> None:
    template_sha = _sha(_json(template))
    proof = report.get("cache_condition", {})
    initial = [
        {key: value[key] for key in ("key", "size_bytes")}
        for value in template["inventory"]
    ]
    target_keys = [value["key"] for value in template["target_inventory"]]
    if report.get("cache_template_sha256") != template_sha or report.get(
        "cache_condition_sha256"
    ) != _sha(_json(proof)):
        raise CampaignError("Cache condition/template proof hash mismatch")
    if (
        proof.get("condition") != condition
        or proof.get("template_sha256") != template_sha
        or proof.get("initial_inventory") != initial
        or proof.get("initial_bytes") != template["initial_bytes"]
        or proof.get("available_bytes") != template["available_bytes"]
        or proof.get("target_keys") != target_keys
        or proof.get("target_present")
        != [template["targets_present"]] * len(target_keys)
    ):
        raise CampaignError(
            "Run started from a different cache inventory or target presence"
        )
    known_paths = {value["path"] for value in artifacts.values()}
    for field in ("read_artifact_paths", "fetched_artifact_paths"):
        observed = report.get(field)
        if not isinstance(observed, list) or any(
            not isinstance(path, str) or path not in known_paths for path in observed
        ):
            raise CampaignError("Missing or foreign source artifact observation")
    read = [
        value["key"]
        for value in template["target_inventory"]
        if value["path"] in report["read_artifact_paths"]
    ]
    fetched = [
        value["key"]
        for value in template["target_inventory"]
        if value["path"] in report["fetched_artifact_paths"]
    ]
    if (
        proof.get("read_target_keys") != read
        or proof.get("fetched_target_keys") != fetched
        or read != target_keys
        or (condition == "evicted" and fetched != target_keys)
    ):
        raise CampaignError("Run did not read/fetch every frozen target shard")
    final = proof.get("final_inventory")
    if not isinstance(final, list) or any(
        not isinstance(row, dict)
        or set(row) != {"key", "size_bytes"}
        or not isinstance(row.get("key"), str)
        or row.get("key") not in artifacts
        or type(row.get("size_bytes")) is not int
        or row.get("size_bytes") != artifacts[row["key"]]["size_bytes"]
        for row in final
    ):
        raise CampaignError("Invalid final cache artifact inventory")
    final_keys = [row["key"] for row in final]
    if (
        final_keys != sorted(set(final_keys))
        or sum(row["size_bytes"] for row in final) > template["cache_bytes"]
    ):
        raise CampaignError(
            "Final cache inventory is duplicated, unordered or over capacity"
        )
    manifest_key = next(
        value["key"]
        for value in template["inventory"]
        if value["path"] == template["artifacts"][0]["path"]
    )
    if manifest_key not in final_keys:
        raise CampaignError("Pinned manifest disappeared from cache")
    removed_initial = sorted({row["key"] for row in initial} - set(final_keys))
    removed_targets = [key for key in target_keys if key not in final_keys]
    if (
        proof.get("removed_initial_keys") != removed_initial
        or proof.get("removed_target_keys") != removed_targets
    ):
        raise CampaignError("Cache removal proof differs from final inventory")
    if condition == "evicted" and (
        not removed_initial and not removed_targets or baseline and not removed_initial
    ):
        raise CampaignError("Evicted run did not prove actual cache removal")


def _execution_order(rules: dict[str, Any], *, qualification: bool) -> list[list[Any]]:
    labels = ("a", "b") if qualification else ("incumbent", "candidate")
    return [
        [workload["id"], index, side]
        for workload in reversed(rules["workloads"])
        for index in range(rules["paired_rounds"])
        for side in labels[:: 1 if index % 2 == 0 else -1]
    ]


def _audit_metadata(
    audit: dict[str, Any],
    rules: dict[str, Any],
    expected: dict[str, Any],
    source_sha256: str,
    state: dict[str, Any],
    workload: dict[str, Any],
    template: dict[str, Any],
    artifacts: dict[str, Any],
    *,
    baseline: bool,
) -> str:
    _cache_condition(
        audit, template, workload["condition"], artifacts, baseline=baseline
    )
    if {key: audit.get(key) for key in PARITY_FIELDS} != expected:
        raise CampaignError("Independent replay batch/device/state parity changed")
    if (
        audit.get("phase") != "audit"
        or type(audit.get("steps")) is not int
        or audit["steps"] != 0
        or audit.get("accepted_batches") != rules["steps"]
        or audit.get("windows") != rules["steps"] * rules["batch_size"]
        or audit.get("pipeline", {}).get("generated_batches") != rules["steps"]
    ):
        raise CampaignError(
            "Independent replay must accept the fixed batches without optimizer steps"
        )
    contract = _sha(
        _json(
            {
                "source_sha256": source_sha256,
                "rules": rules,
                "workload": workload,
                "initial_state": state,
                "data_identity": expected["source_identity"],
                "software": expected["software"],
                "cache_template_sha256": _sha(_json(template)),
            }
        )[:-1]
    )
    if (
        audit.get("source_sha256") != source_sha256
        or audit.get("contract_sha256") != contract
    ):
        raise CampaignError("Independent replay source or contract identity changed")
    return contract


def _report_metadata(
    report: dict[str, Any],
    rules: dict[str, Any],
    expected: dict[str, Any],
    source_sha256: str,
    state: dict[str, Any],
    workload: dict[str, Any],
    template: dict[str, Any],
    artifacts: dict[str, Any],
    *,
    baseline: bool,
) -> None:
    _cache_condition(
        report, template, workload["condition"], artifacts, baseline=baseline
    )
    audit = report.get("audit", {})
    contract = _audit_metadata(
        audit,
        rules,
        expected,
        source_sha256,
        state,
        workload,
        template,
        artifacts,
        baseline=baseline,
    )
    if (
        report.get("audit_sha256") != _sha(_json(audit))
        or report.get("source_sha256") != source_sha256
        or report.get("contract_sha256") != contract
    ):
        raise CampaignError("Timed report independent replay binding changed")
    final_fields = set(PARITY_FIELDS) - {"batch_sha256", "device_batch_sha256"}
    if (
        report.get("phase") != "measure"
        or any(key in report for key in ("batch_sha256", "device_batch_sha256"))
        or {key: report.get(key) for key in final_fields}
        != {key: expected[key] for key in final_fields}
    ):
        raise CampaignError("Actual main-data trace or sampler/RNG parity changed")
    _timings(report)
    environment = report.get("environment", {})
    if (
        any(environment.get(key) != rules[key] for key in ("gpu", "torch", "cuda"))
        or environment.get("torch_threads") != rules["threads"]
    ):
        raise CampaignError("Fixed GPU/software environment changed")
    software = report.get("software", {})
    packages = software.get("packages", {})
    if (
        any(
            not isinstance(environment.get(key), str) or not environment[key]
            for key in ("python", "numpy", "device")
        )
        or environment["device"] != "cuda"
        or software.get("python") != environment["python"]
        or set(packages)
        != {"torch", "numpy", "pyarrow", "pydantic", "PyYAML", "huggingface-hub"}
        or any(not isinstance(value, str) or not value for value in packages.values())
        or software.get("float32_matmul_precision") not in ("highest", "high", "medium")
        or any(
            type(software.get(key)) is not bool
            for key in (
                "cuda_matmul_allow_tf32",
                "cudnn_allow_tf32",
                "cudnn_benchmark",
                "cudnn_deterministic",
            )
        )
    ):
        raise CampaignError("Incomplete software or numerical precision identity")
    pipeline = report.get("pipeline", {})
    if (
        report.get("steps") != rules["steps"]
        or report.get("accepted_batches") != rules["steps"]
        or report.get("windows") != rules["steps"] * rules["batch_size"]
        or pipeline.get("generated_batches") != rules["steps"]
    ):
        raise CampaignError("Wrong fixed amount of accepted optimizer work")
    for value, bound in (
        (report.get("ram_peak_bytes"), rules["memory_bytes"]),
        (report.get("cache_peak_bytes"), rules["cache_bytes"]),
        (pipeline.get("reserved_peak_bytes"), rules["prefetch_bytes"]),
        (pipeline.get("outstanding_peak_batches"), rules["prefetch_depth"]),
    ):
        if type(value) is not int or value < 0 or value > bound:
            raise CampaignError("Fixed memory/cache/prefetch resource bound changed")
    if report.get("slurm_cpus") != str(rules["threads"]) or report.get(
        "slurm_memory_mb"
    ) != str(rules["memory_bytes"] // 1024**2):
        raise CampaignError("Fixed Slurm allocation changed")


def _validate_benchmark(
    source: Path, manifest: dict[str, Any], protocol: dict[str, Any]
) -> None:
    benchmark = protocol.get("benchmark")
    if (
        not isinstance(benchmark, dict)
        or benchmark.get("passed") is not True
        or benchmark.get("independent") is not True
    ):
        raise CampaignError("Initialization requires passing independent qualification")
    if benchmark.get("source_sha256") != _sha(_json(manifest)):
        raise CampaignError("Qualified source identity does not match captured source")
    if _sha(_read(source / BENCHMARK_PATH)) != benchmark.get("benchmark_sha256"):
        raise CampaignError("Qualification benchmark rules hash mismatch")
    rules = _load(source / BENCHMARK_PATH)
    if rules != benchmark.get("rules"):
        raise CampaignError("Qualification benchmark rules changed")
    descriptor_path = source / _safe_relative(rules["descriptor"])
    descriptor = _load(descriptor_path)
    if (
        _sha(_read(descriptor_path)) != MAIN_DESCRIPTOR_SHA256
        or rules.get("descriptor_sha256") != MAIN_DESCRIPTOR_SHA256
        or benchmark.get("descriptor_sha256") != MAIN_DESCRIPTOR_SHA256
    ):
        raise CampaignError("Pinned main descriptor hash mismatch")
    if (
        descriptor.get("revision") != MAIN_REVISION
        or rules.get("revision") != MAIN_REVISION
        or descriptor.get("split") != "train"
        or descriptor.get("subset_sha256") is not None
        or descriptor.get("manifest", {}).get("sha256") != rules.get("manifest_sha256")
    ):
        raise CampaignError("Qualification must use pinned main training data")
    if any(
        rules.get(key) != value
        for key, value in {
            "pair_order": "workload-grouped-ab-ba",
            "evicted_cache": "minimal-survivor-pressure-v1",
            "audit_mode": "independent-replay",
            "inline_audit_ceiling_fraction": 0.01,
        }.items()
    ):
        raise CampaignError("Frozen ordering/cache/replay policy changed")
    workloads = [w["id"] for w in rules["workloads"]]
    if (
        type(rules.get("steady_skip_steps")) is not int
        or not 0 < rules["steady_skip_steps"] < rules["steps"]
    ):
        raise CampaignError("Benchmark needs a bounded sustained-work interval")
    if (
        workloads != protocol["workloads"]
        or rules["paired_rounds"] != protocol["min_paired_rounds"]
        or benchmark.get("qualification") != protocol["qualification"]
    ):
        raise CampaignError("Qualification workload, round or timing values changed")
    _fixture_metadata(benchmark.get("fixture_gates", {}))
    prepared = benchmark.get("prepared", {})
    identity = prepared.get("data_identity", {})
    if any(
        identity.get(key) != value
        for key, value in {
            "revision": MAIN_REVISION,
            "split": "train",
            "subset_sha256": None,
            "repo_id": descriptor["repo_id"],
            "manifest_sha256": rules["manifest_sha256"],
            "membership_sha256": rules["membership_sha256"],
        }.items()
    ):
        raise CampaignError("Qualified main data identity mismatch")
    states, expectations = prepared.get("states", {}), benchmark.get("expectations", {})
    if set(states) != set(workloads) or set(expectations) != set(workloads):
        raise CampaignError("Missing qualified workload sampler state or main trace")
    for name in workloads:
        state, expected = states[name], expectations[name]
        if (
            not isinstance(state, dict)
            or state.get("identity") != identity
            or set(expected) != set(PARITY_FIELDS)
            or expected["source_identity"] != identity
            or expected["initial_state_sha256"] != _state_digest(state)
        ):
            raise CampaignError("Qualified main trace initial state identity mismatch")
        for field in ("batch_sha256", "device_batch_sha256"):
            batches = expected[field]
            if not isinstance(batches, list) or len(batches) != rules["steps"]:
                raise CampaignError("Qualified main trace has wrong batch count")
            for value in batches:
                _digest_string(value)
        for field in ("sampler_sha256", "rng_sha256"):
            _digest_string(expected[field])
        if any(
            type(expected[key]) is not int or expected[key] < 0
            for key in ("draws", "rejected")
        ):
            raise CampaignError("Qualified main trace counters are invalid")
    audits = benchmark.get("audits", {})
    if set(audits) != set(workloads):
        raise CampaignError("Missing independent workload replay")
    templates, artifacts = _cache_templates(prepared, rules, descriptor)
    execution = _execution_order(rules, qualification=True)
    if (
        benchmark.get("execution_order") != execution
        or [
            [report.get("workload"), report.get("round"), report.get("side")]
            for report in benchmark.get("reports", [])
        ]
        != execution
    ):
        raise CampaignError(
            "Qualification did not use frozen workload-grouped AB/BA order"
        )
    definitions = {workload["id"]: workload for workload in rules["workloads"]}
    for name, audit in audits.items():
        _audit_metadata(
            audit,
            rules,
            expectations[name],
            benchmark["source_sha256"],
            states[name],
            definitions[name],
            templates[definitions[name]["condition"]],
            artifacts,
            baseline=True,
        )
    seen = set()
    for report in benchmark.get("reports", []):
        name, index, side = (
            report.get("workload"),
            report.get("round"),
            report.get("side"),
        )
        key = (name, index, side)
        if (
            name not in workloads
            or type(index) is not int
            or index not in range(rules["paired_rounds"])
            or side not in ("a", "b")
            or key in seen
        ):
            raise CampaignError("Invalid or duplicated qualification report")
        seen.add(key)
        _report_metadata(
            report,
            rules,
            expectations[name],
            benchmark["source_sha256"],
            states[name],
            definitions[name],
            templates[definitions[name]["condition"]],
            artifacts,
            baseline=True,
        )
        if report["audit"] != audits[name]:
            raise CampaignError(
                "Qualification timed report references a different replay"
            )
        if report["elapsed_s"] != protocol["qualification"][name][index][side == "b"]:
            raise CampaignError("Qualification timing differs from independent report")
    if seen != {
        (name, index, side)
        for name in workloads
        for index in range(rules["paired_rounds"])
        for side in ("a", "b")
    }:
        raise CampaignError("Missing independent qualification report")


def _validate_measurement_evidence(
    run: Path,
    protocol: dict[str, Any],
    result: dict[str, Any],
    evidence: dict[str, Any],
) -> None:
    if result["status"] != "completed" or not result["gates"]["passed"]:
        return
    benchmark = protocol["benchmark"]
    _fixture_metadata(evidence.get("fixture_gates", {}))
    prepared = evidence.get("conditioning", {}).get("eviction", {})
    if (
        prepared.get("states") != benchmark["prepared"]["states"]
        or prepared.get("data_identity") != benchmark["prepared"]["data_identity"]
    ):
        raise CampaignError(
            "Trial conditioning differs from qualified state/data identity"
        )
    for field in ("shards", "shard_indices", "stress_shards", "cache_templates"):
        if prepared.get(field) != benchmark["prepared"].get(field):
            raise CampaignError(
                "Trial cache/source preparation differs from qualification"
            )
    rules = benchmark["rules"]
    templates, artifacts = _cache_templates(
        prepared, rules, _load(run / "oracle" / _safe_relative(rules["descriptor"]))
    )
    execution = _execution_order(rules, qualification=False)
    expected_pairs = [[value[0], value[1]] for value in execution[::2]]
    if (
        evidence.get("execution_order") != execution
        or [
            [report.get("workload"), report.get("round")]
            for report in evidence.get("reports", [])
        ]
        != expected_pairs
    ):
        raise CampaignError("Trial did not use frozen workload-grouped AB/BA order")
    reports = {}
    for report in evidence.get("reports", []):
        name, index = report.get("workload"), report.get("round")
        key = (name, index)
        if (
            name not in protocol["workloads"]
            or type(index) is not int
            or index not in range(protocol["min_paired_rounds"])
            or key in reports
        ):
            raise CampaignError("Invalid or duplicated independent measurement report")
        for side in ("incumbent", "candidate"):
            _report_metadata(
                report.get(side, {}),
                benchmark["rules"],
                benchmark["expectations"][name],
                result["source_sha256" if side == "candidate" else "incumbent_sha256"],
                benchmark["prepared"]["states"][name],
                next(
                    workload
                    for workload in benchmark["rules"]["workloads"]
                    if workload["id"] == name
                ),
                templates[
                    next(
                        workload["condition"]
                        for workload in rules["workloads"]
                        if workload["id"] == name
                    )
                ],
                artifacts,
                baseline=False,
            )
        reports[key] = report
    grid = {
        (name, index)
        for name in protocol["workloads"]
        for index in range(protocol["min_paired_rounds"])
    }
    if set(reports) != grid:
        raise CampaignError(
            "Independent reports must cover every frozen workload and round"
        )
    seen = set()
    for measurement in result.get("measurements", []):
        key = (measurement.get("workload"), measurement.get("round"))
        if type(key[1]) is not int or key not in reports or key in seen:
            raise CampaignError("Invalid or duplicated paired measurement")
        seen.add(key)
        if any(
            _positive(measurement.get(f"{side}_{result_field}"))
            != reports[key][side][report_field]
            for side in ("incumbent", "candidate")
            for result_field, report_field in (
                ("elapsed_s", "elapsed_s"),
                ("stream_s", "stream_elapsed_s"),
                ("sustained_s", "sustained_elapsed_s"),
            )
        ):
            raise CampaignError(
                "Result timing differs from independent measurement report"
            )
    if seen != grid:
        raise CampaignError(
            "Measurements must cover exactly every frozen workload and round"
        )


def _timings(report: dict[str, Any]) -> dict[str, float]:
    values = {
        key: _positive(report.get(key))
        for key in ("elapsed_s", "stream_elapsed_s", "sustained_elapsed_s")
    }
    if (
        not values["sustained_elapsed_s"]
        <= values["stream_elapsed_s"]
        <= values["elapsed_s"]
    ):
        raise CampaignError("Invalid total/stream/sustained timer order")
    return values


def _noise(protocol: dict[str, Any], floor: float) -> dict[str, Any]:
    reports = protocol.get("benchmark", {}).get("reports", [])
    grid = {}
    expected = {
        (name, index, side)
        for name in protocol["workloads"]
        for index in range(protocol["min_paired_rounds"])
        for side in ("a", "b")
    }
    for report in reports:
        key = (report.get("workload"), report.get("round"), report.get("side"))
        if type(key[1]) is not int or key not in expected or key in grid:
            raise CampaignError("Invalid or duplicated qualification report")
        grid[key] = _timings(report)
    if set(grid) != expected:
        raise CampaignError("Missing independent qualification report")
    pair_logs = {
        metric: {name: [] for name in protocol["workloads"]}
        for metric in ("elapsed_s", "stream_elapsed_s", "sustained_elapsed_s")
    }
    for metric, by_workload in pair_logs.items():
        for name, logs in by_workload.items():
            for index in range(protocol["min_paired_rounds"]):
                logs.append(
                    math.log(
                        grid[name, index, "b"][metric] / grid[name, index, "a"][metric]
                    )
                )

    def aggregate(metric):
        values = pair_logs[metric]
        return max(
            floor,
            math.expm1(
                max(
                    abs(
                        sum(values[name][index] for name in protocol["workloads"])
                        / len(values)
                    )
                    for index in range(protocol["min_paired_rounds"])
                )
            ),
        )

    def individual(metric):
        return {
            name: max(floor, math.expm1(max(abs(value) for value in logs)))
            for name, logs in pair_logs[metric].items()
        }

    return {
        "noise_threshold_fraction": aggregate("stream_elapsed_s"),
        "sustained_noise_threshold_fraction": aggregate("sustained_elapsed_s"),
        "per_workload_total_noise_fraction": individual("elapsed_s"),
        "per_workload_sustained_noise_fraction": individual("sustained_elapsed_s"),
    }


def _validate_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    if protocol.get("schema_version") != 1:
        raise CampaignError("Unsupported protocol schema")
    protocol = dict(protocol)
    trials = protocol.get("trial_budget")
    if trials is not None and (type(trials) is not int or trials < 1):
        raise CampaignError("Invalid trial_budget")
    protocol["trial_budget"] = trials
    rounds = protocol.get("min_paired_rounds")
    if type(rounds) is not int or rounds < 2:
        raise CampaignError("Invalid min_paired_rounds")
    workloads = protocol.get("workloads")
    if (
        not isinstance(workloads, list)
        or len(workloads) < 2
        or any(not isinstance(w, str) or not w for w in workloads)
        or len(set(workloads)) != len(workloads)
    ):
        raise CampaignError("At least two unique workload IDs are required")
    qualification = protocol.get("qualification")
    if not isinstance(qualification, dict) or set(qualification) != set(workloads):
        raise CampaignError("Qualification must cover each workload")
    for values in qualification.values():
        if not isinstance(values, list) or len(values) != rounds:
            raise CampaignError(
                "Qualification needs exactly the frozen paired rounds per workload"
            )
        for pair in values:
            if not isinstance(pair, list) or len(pair) != 2:
                raise CampaignError(
                    "Qualification entries must be [baseline_a_s, baseline_b_s]"
                )
            for value in pair:
                _positive(value)
    floor = _positive(protocol.get("noise_floor_fraction", 0.02))
    if floor >= 1:
        raise CampaignError("Noise floor fraction must be less than one")
    protocol["noise_floor_fraction"] = floor
    # All thresholds come from qualification; caller-supplied derived values cannot
    # replace the frozen aggregation or make one noisy workload reject all wins.
    protocol.pop("per_workload_regression_fraction", None)
    protocol.update(_noise(protocol, floor))
    protocol["trial_job_budget_s"] = _positive(protocol.get("trial_job_budget_s", 3600))
    if protocol["trial_job_budget_s"] > 3600:
        raise CampaignError("One trial may reserve at most 3600 seconds")
    protocol["time_budget_s"] = (
        None
        if protocol.get("time_budget_s") is None
        else _positive(protocol["time_budget_s"])
    )
    return protocol


@contextlib.contextmanager
def _lock(run: Path):
    if not run.is_dir() or any(parent.is_symlink() for parent in (run, *run.parents)):
        raise CampaignError("Campaign directory does not exist or is unsafe")
    path = run / ".lock"
    if path.is_symlink():
        raise CampaignError("Unsafe campaign lock")
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _append(
    run: Path, entries: list[dict[str, Any]], kind: str, payload: dict[str, Any]
) -> None:
    entry = {
        "sequence": len(entries),
        "previous": entries[-1]["sha256"] if entries else ZERO,
        "kind": kind,
        "payload": payload,
    }
    entry["sha256"] = _sha(_json(entry))
    path = run / "journal.jsonl"
    if path.exists():
        _regular(path)
    with path.open("ab") as stream:
        stream.write(_json(entry))
        stream.flush()
        os.fsync(stream.fileno())
    _sync_dir(run)
    _write(
        run / "cursor.json",
        _json({"sequence": entry["sequence"], "sha256": entry["sha256"]}),
    )


def _state(run: Path) -> dict[str, Any]:
    raw = _read(run / "journal.jsonl")
    if not raw.endswith(b"\n"):
        raise CampaignError("Torn journal")
    entries = []
    for line in raw.splitlines():
        try:
            entry = json.loads(line)
            digest = entry.pop("sha256")
        except (ValueError, KeyError, TypeError) as exc:
            raise CampaignError("Invalid journal entry") from exc
        if (
            entry.get("sequence") != len(entries)
            or entry.get("previous") != (entries[-1]["sha256"] if entries else ZERO)
            or _sha(_json(entry)) != digest
        ):
            raise CampaignError("Journal hash chain mismatch")
        entry["sha256"] = digest
        entries.append(entry)
    if not entries or entries[0]["kind"] != "initialize":
        raise CampaignError("Missing initialization")
    if _load(run / "cursor.json") != {
        "sequence": len(entries) - 1,
        "sha256": entries[-1]["sha256"],
    }:
        raise CampaignError("Journal and cursor diverged")
    initial = entries[0]["payload"]
    protocol = _load(run / "protocol.json")
    if _sha(_json(protocol)) != initial["protocol_sha256"]:
        raise CampaignError("Frozen protocol changed")
    manifest = _verify_snapshot(run, initial["oracle_sha256"])
    _verify_tree(run / "oracle", manifest)
    _validate_benchmark(run / "oracle", manifest, protocol)
    best = {
        "source_sha256": initial["oracle_sha256"],
        "trial_id": None,
        "result_sha256": None,
    }
    active = None
    trials = {}
    reserved = 0.0
    recorded = 0
    accounting_totals = {"compute_s": 0.0, "queue_s": 0.0}
    accounting_counts = {"compute_s": 0, "queue_s": 0}
    for entry in entries[1:]:
        kind, payload = entry["kind"], entry["payload"]
        if kind == "trial":
            identifier = payload["trial_id"]
            if (
                active is not None
                or identifier != str(len(trials) + 1).zfill(6)
                or payload["incumbent_sha256"] != best["source_sha256"]
                or payload["job_budget_s"] != protocol["trial_job_budget_s"]
            ):
                raise CampaignError("Invalid trial reservation")
            _verify_snapshot(run, payload["source_sha256"])
            if _load(run / "trials" / identifier / "reservation.json") != payload:
                raise CampaignError("Reservation artifact diverged from journal")
            trials[identifier] = dict(payload)
            active = identifier
            reserved += payload["job_budget_s"]
        elif kind == "record":
            if payload["trial_id"] != active or "result_sha256" in trials[active]:
                raise CampaignError("Invalid result transition")
            trials[active]["result_sha256"] = payload["result_sha256"]
            result = _verify_result(run, trials[active], initial)
            recorded += 1
            for key, value in _accounting(result.get("job_accounting")).items():
                accounting_totals[key] += value
                accounting_counts[key] += 1
        elif kind in ("retain", "discard"):
            if payload["trial_id"] != active:
                raise CampaignError("Invalid decision transition")
            if kind == "retain":
                trial = trials[active]
                result = _verify_result(run, trial, initial)
                if not _score(protocol, result)["eligible"]:
                    raise CampaignError("Journal retained an ineligible result")
                best = {
                    "source_sha256": trial["source_sha256"],
                    "trial_id": active,
                    "result_sha256": trial["result_sha256"],
                }
            active = None
        else:
            raise CampaignError(f"Unknown journal event: {kind}")
    if (
        protocol["trial_budget"] is not None and len(trials) > protocol["trial_budget"]
    ) or (
        protocol["time_budget_s"] is not None and reserved > protocol["time_budget_s"]
    ):
        raise CampaignError("Journal exceeds frozen budget")
    if _load(run / "best.json") != best:
        raise CampaignError("Best pointer diverged from journal")
    _verify_snapshot(run, best["source_sha256"])
    return {
        "entries": entries,
        "initial": initial,
        "protocol": protocol,
        "manifest": manifest,
        "best": best,
        "active": active,
        "trials": trials,
        "reserved_s": reserved,
        "recorded_jobs": recorded,
        "accounting_totals": accounting_totals,
        "accounting_counts": accounting_counts,
    }


def _identities(trial: dict[str, Any], initial: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (trial if key in trial else initial)[key]
        for key in (
            "trial_id",
            "source_sha256",
            "incumbent_sha256",
            "oracle_sha256",
            "protocol_sha256",
        )
    }


def _accounting(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise CampaignError("Invalid job accounting")
    result = {}
    for key in ("compute_s", "queue_s"):
        number = value.get(key)
        if number is None:
            continue
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number < 0
        ):
            raise CampaignError("Job accounting requires finite nonnegative durations")
        result[key] = float(number)
    return result


def _prefer_slurm_accounting(run: Path, trial_id: str, result: dict[str, Any]) -> None:
    path = run / "trials" / trial_id / "slurm-accounting.json"
    if not path.exists() and not path.is_symlink():
        return
    official = _load(path)
    _accounting(official)
    if (
        official.get("basis") != "slurm"
        or not isinstance(official.get("job_id"), str)
        or not official["job_id"]
    ):
        raise CampaignError("Official accounting needs Slurm job identity")
    previous = result.get("job_accounting") or {}
    expected_job = previous.get("job_id")
    if expected_job is None:
        gates = result["gates"]
        evidence = _load(
            run / "trials" / trial_id / _safe_relative(gates["evidence_file"])
        )
        jobs = {
            report[side].get("slurm_job_id")
            for report in evidence.get("reports", [])
            for side in ("incumbent", "candidate")
            if report.get(side, {}).get("slurm_job_id") is not None
        }
        if len(jobs) == 1:
            expected_job = jobs.pop()
    if expected_job != official["job_id"]:
        raise CampaignError("Slurm accounting belongs to a different or unbound job")
    result["job_accounting"] = {
        **previous,
        **official,
        "driver": previous,
        "sidecar_sha256": _sha(_read(path)),
    }


def _validate_result(
    run: Path, trial: dict[str, Any], initial: dict[str, Any], result: dict[str, Any]
) -> None:
    accounting = result.get("job_accounting")
    _accounting(accounting)
    if accounting is not None and accounting.get("basis") == "slurm":
        sidecar = run / "trials" / trial["trial_id"] / "slurm-accounting.json"
        if accounting.get("sidecar_sha256") != _sha(_read(sidecar)):
            raise CampaignError("Sealed Slurm accounting sidecar changed")
        if any(accounting.get(key) != value for key, value in _load(sidecar).items()):
            raise CampaignError("Sealed Slurm accounting differs from sidecar")
    identities = _identities(trial, initial)
    if any(result.get(key) != value for key, value in identities.items()):
        raise CampaignError("Result identity does not match reserved trial")
    if result.get("status") not in ("completed", "failed"):
        raise CampaignError("Result must declare completed or failed status")
    gates = result.get("gates", {})
    if (
        type(gates.get("passed")) is not bool
        or type(gates.get("independent")) is not bool
    ):
        raise CampaignError("Result must declare independent gate status")
    evidence_path = (
        run
        / "trials"
        / trial["trial_id"]
        / _safe_relative(gates.get("evidence_file", ""))
    )
    evidence_raw = _read(evidence_path)
    if _sha(evidence_raw) != gates.get("evidence_sha256"):
        raise CampaignError("Gate evidence hash mismatch")
    evidence = _load(evidence_path)
    if (
        any(evidence.get(key) != value for key, value in identities.items())
        or evidence.get("passed") is not gates["passed"]
        or evidence.get("independent") is not gates["independent"]
    ):
        raise CampaignError("Gate evidence identity or status mismatch")
    _validate_measurement_evidence(run, _load(run / "protocol.json"), result, evidence)


def _verify_result(
    run: Path, trial: dict[str, Any], initial: dict[str, Any]
) -> dict[str, Any]:
    raw = _read(run / "trials" / trial["trial_id"] / "result.json")
    if _sha(raw) != trial.get("result_sha256"):
        raise CampaignError("Recorded result hash mismatch")
    result = json.loads(raw)
    _validate_result(run, trial, initial, result)
    if result.get("score") != _score(_load(run / "protocol.json"), result):
        raise CampaignError("Sealed score differs from frozen score aggregation")
    return result


def _score(protocol: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    if (
        result["status"] != "completed"
        or not result["gates"]["passed"]
        or not result["gates"]["independent"]
    ):
        return {
            "metric": SCORE_METRIC,
            "eligible": False,
            "reason": "independent gates failed",
        }
    rounds = {
        metric: {name: {} for name in protocol["workloads"]}
        for metric in ("elapsed_s", "stream_s", "sustained_s")
    }
    for row in result.get("measurements", []):
        name, index = row.get("workload"), row.get("round")
        if (
            name not in protocol["workloads"]
            or type(index) is not int
            or index < 0
            or index in rounds["elapsed_s"][name]
        ):
            raise CampaignError("Invalid or duplicated paired measurement")
        for metric, values in rounds.items():
            values[name][index] = math.log(
                _positive(row.get(f"candidate_{metric}"))
                / _positive(row.get(f"incumbent_{metric}"))
            )
    if any(
        set(values) != set(range(protocol["min_paired_rounds"]))
        for values in rounds["elapsed_s"].values()
    ):
        raise CampaignError(
            "Measurements must cover exactly every frozen workload and round"
        )
    per_workload = {
        metric: {
            name: math.exp(sum(values.values()) / len(values))
            for name, values in workloads.items()
        }
        for metric, workloads in rounds.items()
    }
    aggregate = {
        metric: math.exp(
            sum(math.log(value) for value in values.values()) / len(values)
        )
        for metric, values in per_workload.items()
    }
    per_round = {
        metric: {
            str(index): math.exp(
                sum(values[index] for values in workloads.values()) / len(workloads)
            )
            for index in range(protocol["min_paired_rounds"])
        }
        for metric, workloads in rounds.items()
        if metric in ("stream_s", "sustained_s")
    }
    stream_threshold = protocol["noise_threshold_fraction"]
    sustained_threshold = protocol["sustained_noise_threshold_fraction"]
    eligible = (
        aggregate["stream_s"] < 1 / (1 + stream_threshold)
        and aggregate["sustained_s"] < 1 / (1 + sustained_threshold)
        and all(
            value < 1 / (1 + stream_threshold)
            for value in per_round["stream_s"].values()
        )
        and all(
            value < 1 / (1 + sustained_threshold)
            for value in per_round["sustained_s"].values()
        )
        and all(
            value <= 1 + protocol["per_workload_total_noise_fraction"][name]
            for name, value in per_workload["elapsed_s"].items()
        )
        and all(
            value <= 1 + protocol["per_workload_sustained_noise_fraction"][name]
            for name, value in per_workload["sustained_s"].items()
        )
    )
    return {
        "metric": SCORE_METRIC,
        "eligible": eligible,
        "geometric_stream_ratio": aggregate["stream_s"],
        "geometric_sustained_ratio": aggregate["sustained_s"],
        "geometric_elapsed_ratio": aggregate["elapsed_s"],
        "per_round_stream_ratio": per_round["stream_s"],
        "per_round_sustained_ratio": per_round["sustained_s"],
        "per_workload_elapsed_ratio": per_workload["elapsed_s"],
        "per_workload_stream_ratio": per_workload["stream_s"],
        "per_workload_sustained_ratio": per_workload["sustained_s"],
        "noise_threshold_fraction": stream_threshold,
        "sustained_noise_threshold_fraction": sustained_threshold,
        "per_workload_total_noise_fraction": protocol[
            "per_workload_total_noise_fraction"
        ],
        "per_workload_sustained_noise_fraction": protocol[
            "per_workload_sustained_noise_fraction"
        ],
    }


def initialize(source: Path, run: Path, protocol_path: Path) -> dict[str, Any]:
    source, run = source.absolute(), run.absolute()
    if run.resolve().is_relative_to(
        source.resolve()
    ) or source.resolve().is_relative_to(run.resolve()):
        raise CampaignError("Campaign artifacts must be outside the source worktree")
    if any(parent.is_symlink() for parent in (run, *run.parents)):
        raise CampaignError("Campaign directory has a symlink parent")
    protocol = _validate_protocol(_load(protocol_path))
    manifest = _inventory(source)
    _validate_benchmark(source, manifest, protocol)
    run.mkdir(parents=True, exist_ok=False)
    with _lock(run):
        digest = _snapshot(run, source, manifest)
        shutil.copytree(run / "snapshots" / digest / "source", run / "oracle")
        _write(run / "protocol.json", _json(protocol), exclusive=True)
        initial = {
            "oracle_sha256": digest,
            "protocol_sha256": _sha(_json(protocol)),
            "git_revision": _git(source, "rev-parse", "HEAD").decode().strip(),
            "editable_paths": sorted(EDITABLE),
            "initialized_at": _utc_now(),
        }
        _append(run, [], "initialize", initial)
        _write(
            run / "best.json",
            _json({"source_sha256": digest, "trial_id": None, "result_sha256": None}),
            exclusive=True,
        )
    return readiness(run)


def _capture(run: Path, source: Path, state: dict[str, Any]) -> str:
    manifest = _inventory(source)
    frozen = {
        name: identity
        for name, identity in state["manifest"].items()
        if name not in EDITABLE
    }
    if {
        name: identity for name, identity in manifest.items() if name not in EDITABLE
    } != frozen:
        raise CampaignError("Candidate changed frozen source or dependencies")
    return _snapshot(run, source, manifest)


def capture(run: Path, source: Path) -> dict[str, Any]:
    with _lock(run):
        state = _state(run)
        digest = _capture(run, source, state)
        return {
            "source_sha256": digest,
            "source_path": str(run / "snapshots" / digest / "source"),
        }


def trial(run: Path, source: Path, description: str | None = None) -> dict[str, Any]:
    if description is not None and (
        not isinstance(description, str)
        or not description.strip()
        or len(description) > 1000
        or "\0" in description
    ):
        raise CampaignError(
            "Trial description must be nonempty, at most 1000 characters and contain no NUL"
        )
    with _lock(run):
        state = _state(run)
        if state["active"] is not None:
            raise CampaignError(
                f"Trial {state['active']} remains active; reconcile it before reserving another"
            )
        protocol = state["protocol"]
        if (
            protocol["trial_budget"] is not None
            and len(state["trials"]) >= protocol["trial_budget"]
        ) or (
            protocol["time_budget_s"] is not None
            and state["reserved_s"] + protocol["trial_job_budget_s"]
            > protocol["time_budget_s"]
        ):
            raise CampaignError("Campaign reservation budget exhausted")
        digest = _capture(run, source, state)
        identifier = str(len(state["trials"]) + 1).zfill(6)
        directory = run / "trials" / identifier
        directory.mkdir(parents=True, exist_ok=False)
        payload = {
            "trial_id": identifier,
            "source_sha256": digest,
            "incumbent_sha256": state["best"]["source_sha256"],
            "job_budget_s": protocol["trial_job_budget_s"],
            "description": description,
            "reserved_at": _utc_now(),
        }
        _write(directory / "reservation.json", _json(payload), exclusive=True)
        _append(run, state["entries"], "trial", payload)
        return {
            **payload,
            **state["initial"],
            "trial_path": str(directory),
            "candidate_path": str(run / "snapshots" / digest / "source"),
            "incumbent_path": str(
                run / "snapshots" / payload["incumbent_sha256"] / "source"
            ),
            "oracle_path": str(run / "oracle"),
        }


def record(run: Path, trial_id: str, result_path: Path) -> dict[str, Any]:
    with _lock(run):
        state = _state(run)
        if state["active"] != trial_id or "result_sha256" in state["trials"][trial_id]:
            raise CampaignError("Trial is not active or already has a result")
        result = _load(result_path)
        _validate_result(run, state["trials"][trial_id], state["initial"], result)
        _prefer_slurm_accounting(run, trial_id, result)
        _validate_result(run, state["trials"][trial_id], state["initial"], result)
        score = _score(state["protocol"], result)
        result["score"] = score
        result["recorded_at"] = _utc_now()
        raw = _json(result)
        _write(run / "trials" / trial_id / "result.json", raw, exclusive=True)
        _append(
            run,
            state["entries"],
            "record",
            {"trial_id": trial_id, "result_sha256": _sha(raw)},
        )
        return score


def decide(run: Path, trial_id: str, source: Path, *, retain: bool) -> dict[str, Any]:
    with _lock(run):
        state = _state(run)
        if state["active"] != trial_id:
            raise CampaignError("Decision requires the active trial")
        reserved = state["trials"][trial_id]
        best = state["best"]
        if retain:
            result = _verify_result(run, reserved, state["initial"])
            if not _score(state["protocol"], result)["eligible"]:
                raise CampaignError(
                    "Candidate did not pass independent gates and retention rule"
                )
            best = {
                "source_sha256": reserved["source_sha256"],
                "trial_id": trial_id,
                "result_sha256": reserved["result_sha256"],
            }
        # Validate the current worktree before restoring, including directory symlinks.
        _capture(run, source, state)
        manifest = _verify_snapshot(run, best["source_sha256"])
        for name in EDITABLE:
            target = source / name
            if name in manifest:
                _write(
                    target,
                    _read(run / "snapshots" / best["source_sha256"] / "source" / name),
                )
                target.chmod(0o755 if manifest[name]["executable"] else 0o644)
            elif target.exists():
                target.unlink()
        _append(
            run,
            state["entries"],
            "retain" if retain else "discard",
            {"trial_id": trial_id},
        )
        if retain:
            _write(run / "best.json", _json(best))
        return {"decision": "retain" if retain else "discard", "best": best}


def readiness(run: Path) -> dict[str, Any]:
    with _lock(run):
        state = _state(run)
        protocol = state["protocol"]
        remaining = (
            None
            if protocol["trial_budget"] is None
            else protocol["trial_budget"] - len(state["trials"])
        )
        seconds = (
            None
            if protocol["time_budget_s"] is None
            else protocol["time_budget_s"] - state["reserved_s"]
        )
        return {
            "ready": state["active"] is None
            and (remaining is None or remaining > 0)
            and (seconds is None or seconds >= protocol["trial_job_budget_s"]),
            "active_trial": state["trials"].get(state["active"]),
            "recorded_compute_seconds": state["accounting_totals"]["compute_s"]
            if state["accounting_counts"]["compute_s"]
            else None,
            "recorded_queue_seconds": state["accounting_totals"]["queue_s"]
            if state["accounting_counts"]["queue_s"]
            else None,
            "compute_accounted_jobs": state["accounting_counts"]["compute_s"],
            "queue_accounted_jobs": state["accounting_counts"]["queue_s"],
            "recorded_jobs": state["recorded_jobs"],
            "reserved_jobs": len(state["trials"]),
            "reserved_seconds": state["reserved_s"],
            "remaining_trials": remaining,
            "remaining_reserved_seconds": seconds,
            "best": state["best"],
            "oracle_sha256": state["initial"]["oracle_sha256"],
            "protocol_sha256": state["initial"]["protocol_sha256"],
            "noise_threshold_fraction": protocol["noise_threshold_fraction"],
            "sustained_noise_threshold_fraction": protocol[
                "sustained_noise_threshold_fraction"
            ],
            "per_workload_total_noise_fraction": protocol[
                "per_workload_total_noise_fraction"
            ],
            "per_workload_sustained_noise_fraction": protocol[
                "per_workload_sustained_noise_fraction"
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command in (
        "initialize",
        "capture",
        "trial",
        "record",
        "retain",
        "discard",
        "readiness",
    ):
        subparser = subcommands.add_parser(command)
        subparser.add_argument("--run", type=Path, required=True)
        if command in ("initialize", "capture", "trial", "retain", "discard"):
            subparser.add_argument("--source", type=Path, required=True)
        if command == "initialize":
            subparser.add_argument("--protocol", type=Path, required=True)
        if command in ("record", "retain", "discard"):
            subparser.add_argument("--trial", required=True)
        if command == "trial":
            subparser.add_argument("--description")
        if command == "record":
            subparser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "initialize":
            result = initialize(args.source, args.run, args.protocol)
        elif args.command == "capture":
            result = capture(args.run, args.source)
        elif args.command == "trial":
            result = trial(args.run, args.source, args.description)
        elif args.command == "record":
            result = record(args.run, args.trial, args.result)
        elif args.command in ("retain", "discard"):
            result = decide(
                args.run, args.trial, args.source, retain=args.command == "retain"
            )
        else:
            result = readiness(args.run)
    except (CampaignError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"campaign: {exc}\n")
    print(_json(result).decode(), end="")


if __name__ == "__main__":
    main()
