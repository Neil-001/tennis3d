"""Campaign decisions never use loss or an unchecked candidate measurement."""

import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from tennis3d.throughput import campaign

ROOT = Path(__file__).resolve().parents[1]


def fixture_metadata():
    expected = {
        key: campaign._sha(key.encode())
        for key in ("model", "ema", "optimizer", "sampler", "rng")
    }
    lifecycle = {
        "traces": {
            name: [["a" * 64, "b" * 64] for _ in range(count)]
            for name, count in {
                "ordinary_epochs": 8,
                "rejections": 8,
                "checkpoint_stop_resume": 6,
                "failure_resume": 6,
                "inflight_stop_resume": 6,
                "complete_view_corruption": 3,
            }.items()
        },
        "rng_sha256": "c" * 64,
    }
    return {
        "maintained_update_state": expected,
        "source_policy": {
            side: {"owned_cache": True, "bounded_reader": True}
            for side in ("oracle", "candidate")
        },
        "reports": {
            name: dict(expected)
            for name in (
                "maintained",
                "oracle-loop",
                "candidate-loop",
                "candidate-resume",
            )
        },
        "lifecycle_tests_sha256": "d" * 64,
        "lifecycle": {
            "oracle": copy.deepcopy(lifecycle),
            "candidate": copy.deepcopy(lifecycle),
        },
    }


def cache_preparation(rules, descriptor, identity, states):
    def entry(artifact):
        return {
            "key": campaign._sha(
                campaign._json(
                    {
                        "repo": descriptor["repo_id"],
                        "commit": descriptor["revision"],
                        **artifact,
                    }
                )[:-1]
            ),
            **artifact,
        }

    manifest = entry(descriptor["manifest"])
    shards = [
        {
            "path": f"dataset/part-{index:03d}.parquet",
            "sha256": campaign._sha(f"shard-{index}".encode()),
            "size_bytes": 300_000_000,
        }
        for index in range(20)
    ]
    rows = [entry(value) for value in shards]
    definitions = {workload["condition"]: workload for workload in rules["workloads"]}
    indices = {
        definitions["warm"]["id"]: [0, 1, 2, 3],
        definitions["evicted"]["id"]: [16, 17, 18, 19],
    }
    templates = {}
    # With these sizes, exactly ten of the thirteen retained stress shards are
    # the shortest descending-size/path prefix that forces subsequent eviction.
    for condition, selected in (
        ("warm", [manifest, *rows[:4]]),
        ("evicted", [manifest, *rows[2:12]]),
    ):
        targets = rows[:4] if condition == "warm" else rows[16:20]
        initial_bytes = sum(value["size_bytes"] for value in selected)
        templates[condition] = {
            "artifacts": [
                {key: value[key] for key in ("path", "sha256", "size_bytes")}
                for value in selected
            ],
            "inventory": sorted(selected, key=lambda value: value["key"]),
            "cache_bytes": rules["cache_bytes"],
            "initial_bytes": initial_bytes,
            "available_bytes": rules["cache_bytes"] - initial_bytes,
            "target_inventory": targets,
            "target_bytes": 1_200_000_000,
            "targets_present": condition == "warm",
            "overflow_bytes": max(
                0, initial_bytes + 1_200_000_000 - rules["cache_bytes"]
            )
            if condition == "evicted"
            else 0,
        }
    return {
        "data_identity": identity,
        "states": states,
        "shards": shards,
        "shard_indices": indices,
        "stress_shards": [16, *range(15)],
        "cache_keys": sorted(
            [manifest["key"], *(value["key"] for value in rows[2:15])]
        ),
        "evicted_keys": [rows[index]["key"] for index in (16, 0, 1)],
        "cache_templates": templates,
    }


def cache_report(benchmark, workload, *, remove_target=False):
    condition = next(
        value["condition"]
        for value in benchmark["rules"]["workloads"]
        if value["id"] == workload
    )
    template = benchmark["prepared"]["cache_templates"][condition]
    initial = [
        {key: value[key] for key in ("key", "size_bytes")}
        for value in template["inventory"]
    ]
    target_keys = [value["key"] for value in template["target_inventory"]]
    final = {value["key"]: value["size_bytes"] for value in initial}
    if condition == "evicted":
        final.update(
            {
                value["key"]: value["size_bytes"]
                for value in template["target_inventory"]
            }
        )
        if remove_target:
            del final[target_keys[0]]
        else:
            filler = template["artifacts"][1]["path"]
            del final[
                next(
                    value["key"]
                    for value in template["inventory"]
                    if value["path"] == filler
                )
            ]
    final_inventory = [
        {"key": key, "size_bytes": size} for key, size in sorted(final.items())
    ]
    proof = {
        "condition": condition,
        "template_sha256": campaign._sha(campaign._json(template)),
        "initial_inventory": initial,
        "initial_bytes": template["initial_bytes"],
        "available_bytes": template["available_bytes"],
        "target_keys": target_keys,
        "target_present": [condition == "warm"] * len(target_keys),
        "final_inventory": final_inventory,
        "removed_initial_keys": sorted(
            {value["key"] for value in initial} - set(final)
        ),
        "removed_target_keys": [key for key in target_keys if key not in final],
        "read_target_keys": target_keys,
        "fetched_target_keys": target_keys if condition == "evicted" else [],
    }
    return {
        "cache_template_sha256": proof["template_sha256"],
        "cache_condition": proof,
        "cache_condition_sha256": campaign._sha(campaign._json(proof)),
        "read_artifact_paths": [
            value["path"] for value in template["target_inventory"]
        ],
        "fetched_artifact_paths": [
            value["path"] for value in template["target_inventory"]
        ]
        if condition == "evicted"
        else [],
    }


def execution_order(rules, *, qualification):
    labels = ("a", "b") if qualification else ("incumbent", "candidate")
    order = []
    for workload in reversed(rules["workloads"]):
        for index in range(rules["paired_rounds"]):
            for side in labels if index % 2 == 0 else reversed(labels):
                order.append([workload["id"], index, side])
    return order


def audit_metadata(benchmark, workload, source_sha256=None):
    rules = benchmark["rules"]
    identity = source_sha256 or benchmark["source_sha256"]
    expected = benchmark["expectations"][workload]
    definition = next(value for value in rules["workloads"] if value["id"] == workload)
    contract = campaign._sha(
        campaign._json(
            {
                "source_sha256": identity,
                "rules": rules,
                "workload": definition,
                "initial_state": benchmark["prepared"]["states"][workload],
                "data_identity": expected["source_identity"],
                "software": expected["software"],
                "cache_template_sha256": cache_report(benchmark, workload)[
                    "cache_template_sha256"
                ],
            }
        )[:-1]
    )
    return {
        **copy.deepcopy(expected),
        **cache_report(benchmark, workload),
        "phase": "audit",
        "steps": 0,
        "accepted_batches": rules["steps"],
        "windows": rules["steps"] * rules["batch_size"],
        "pipeline": {"generated_batches": rules["steps"]},
        "source_sha256": identity,
        "contract_sha256": contract,
    }


def report_metadata(
    benchmark, workload, elapsed, *, stream=None, sustained=None, source_sha256=None
):
    rules = benchmark["rules"]
    audit = audit_metadata(benchmark, workload, source_sha256)
    return {
        **{
            key: copy.deepcopy(value)
            for key, value in benchmark["expectations"][workload].items()
            if key not in ("batch_sha256", "device_batch_sha256")
        },
        **cache_report(benchmark, workload),
        "phase": "measure",
        "source_sha256": audit["source_sha256"],
        "contract_sha256": audit["contract_sha256"],
        "audit": audit,
        "audit_sha256": campaign._sha(campaign._json(audit)),
        "elapsed_s": elapsed,
        "stream_elapsed_s": elapsed * 0.8 if stream is None else stream,
        "sustained_elapsed_s": elapsed * 0.7 if sustained is None else sustained,
        "steps": rules["steps"],
        "accepted_batches": rules["steps"],
        "windows": rules["steps"] * rules["batch_size"],
        "pipeline": {
            "generated_batches": rules["steps"],
            "reserved_peak_bytes": 1000,
            "outstanding_peak_batches": 2,
        },
        "ram_peak_bytes": 1000,
        "cache_peak_bytes": 1000,
        "slurm_cpus": str(rules["threads"]),
        "slurm_memory_mb": str(rules["memory_bytes"] // 1024**2),
    }


def qualify_metadata(source, protocol):
    rules_path = source / campaign.BENCHMARK_PATH
    rules = json.loads(rules_path.read_text())
    descriptor = json.loads((source / rules["descriptor"]).read_text())
    identity = {
        "revision": rules["revision"],
        "split": "train",
        "subset_sha256": None,
        "repo_id": descriptor["repo_id"],
        "manifest_sha256": rules["manifest_sha256"],
        "membership_sha256": rules["membership_sha256"],
    }
    states = {
        name: {"identity": identity, "cursor": 0, "version": 2}
        for name in protocol["workloads"]
    }
    environment = {key: rules[key] for key in ("gpu", "torch", "cuda")} | {
        "torch_threads": rules["threads"],
        "python": "3.11.1",
        "numpy": "2.1.0",
        "device": "cuda",
    }
    software = {
        "python": environment["python"],
        "packages": {
            "torch": rules["torch"],
            "numpy": environment["numpy"],
            "pyarrow": "25.0.0",
            "pydantic": "2.11.0",
            "PyYAML": "6.0.2",
            "huggingface-hub": "1.33.0",
        },
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": False,
    }
    expectations = {
        name: {
            "batch_sha256": ["a" * 64] * rules["steps"],
            "device_batch_sha256": ["e" * 64] * rules["steps"],
            "sampler_sha256": "b" * 64,
            "rng_sha256": "c" * 64,
            "draws": 384,
            "rejected": 0,
            "initial_state_sha256": campaign._state_digest(states[name]),
            "source_identity": identity,
            "environment": environment,
            "software": software,
        }
        for name in protocol["workloads"]
    }
    benchmark = {
        "source_sha256": campaign._sha(campaign._json(campaign._inventory(source))),
        "benchmark_sha256": campaign._sha(rules_path.read_bytes()),
        "descriptor_sha256": rules["descriptor_sha256"],
        "rules": rules,
        "qualification": copy.deepcopy(protocol["qualification"]),
        "prepared": cache_preparation(rules, descriptor, identity, states),
        "expectations": expectations,
        "fixture_gates": fixture_metadata(),
        "passed": True,
        "independent": True,
    }
    benchmark["audits"] = {
        name: audit_metadata(benchmark, name) for name in protocol["workloads"]
    }
    benchmark["execution_order"] = execution_order(rules, qualification=True)
    benchmark["reports"] = [
        {
            "workload": name,
            "round": index,
            "side": side,
            **report_metadata(
                benchmark, name, protocol["qualification"][name][index][side == "b"]
            ),
        }
        for name, index, side in benchmark["execution_order"]
    ]
    return benchmark


def sync_result_evidence(run, result):
    evidence_path = run / "trials" / result["trial_id"] / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    benchmark = json.loads((run / "protocol.json").read_text())["benchmark"]
    evidence["execution_order"] = execution_order(
        benchmark["rules"], qualification=False
    )
    pair_positions = {
        (value[0], value[1]): index
        for index, value in enumerate(evidence["execution_order"][::2])
    }
    evidence["reports"] = [
        {
            "workload": row["workload"],
            "round": row["round"],
            **{
                side: report_metadata(
                    benchmark,
                    row["workload"],
                    row[f"{side}_elapsed_s"],
                    stream=row[f"{side}_stream_s"],
                    sustained=row[f"{side}_sustained_s"],
                    source_sha256=result[
                        "source_sha256" if side == "candidate" else "incumbent_sha256"
                    ],
                )
                for side in ("incumbent", "candidate")
            },
        }
        for row in sorted(
            result["measurements"],
            key=lambda value: pair_positions[value["workload"], value["round"]],
        )
    ]
    write_json(evidence_path, evidence)
    result["gates"]["evidence_sha256"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    files = {
        "tennis3d/lifting/data.py": "baseline = 1\n",
        "tennis3d/lifting/model.py": "math = 42\n",
        "pyproject.toml": "dependencies = []\n",
        "uv.lock": "version = 1\n",
    }
    rules = json.loads((ROOT / campaign.BENCHMARK_PATH).read_text())
    rules.update(
        steps=128,
        steady_skip_steps=8,
        job_seconds=3600,
        pair_order="workload-grouped-ab-ba",
        evicted_cache="minimal-survivor-pressure-v1",
        audit_mode="independent-replay",
        inline_audit_ceiling_fraction=0.01,
    )
    rules["workloads"] = [
        {"id": "warm", "condition": "warm"},
        {"id": "eviction", "condition": "evicted"},
    ]
    files[campaign.BENCHMARK_PATH] = json.dumps(rules)
    files[rules["descriptor"]] = (ROOT / rules["descriptor"]).read_text()
    for name, contents in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "-qm",
            "baseline",
        ],
        check=True,
    )
    # Initialization freezes actual file contents, including uncommitted edits.
    (source / "tennis3d/lifting/data.py").write_text("baseline = 2\n")
    protocol = tmp_path / "protocol.json"
    write_json(
        protocol,
        {
            "schema_version": 1,
            "trial_budget": 3,
            "min_paired_rounds": 2,
            "trial_job_budget_s": 100,
            "time_budget_s": 200,
            "workloads": ["warm", "eviction"],
            "qualification": {
                "warm": [[10, 10.1], [11, 11.2]],
                "eviction": [[20, 20.2], [21, 21.1]],
            },
        },
    )
    values = json.loads(protocol.read_text())
    values["benchmark"] = qualify_metadata(source, values)
    write_json(protocol, values)
    run = tmp_path / "campaign"
    campaign.initialize(source, run, protocol)
    return source, run


def reserve(prepared):
    source, run = prepared
    (source / "tennis3d/lifting/data.py").write_text("candidate = 3\n")
    return campaign.trial(run, source)


def result_file(run, reservation, *, ratio=0.8, passed=True):
    identities = {
        key: reservation[key]
        for key in (
            "trial_id",
            "source_sha256",
            "incumbent_sha256",
            "oracle_sha256",
            "protocol_sha256",
        )
    }
    evidence = run / "trials" / reservation["trial_id"] / "evidence.json"
    write_json(
        evidence,
        {
            **identities,
            "passed": passed,
            "independent": True,
            "batch_parity": "exact",
            "fixture_gates": fixture_metadata(),
            "conditioning": {
                "eviction": json.loads((run / "protocol.json").read_text())[
                    "benchmark"
                ]["prepared"]
            },
        },
    )
    result = {
        **identities,
        "status": "completed",
        "gates": {
            "passed": passed,
            "independent": True,
            "evidence_file": "evidence.json",
            "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        },
        "measurements": [
            {
                "workload": workload,
                "round": i,
                "incumbent_elapsed_s": 10,
                "candidate_elapsed_s": ratio * 10,
                "incumbent_stream_s": 8,
                "candidate_stream_s": ratio * 8,
                "incumbent_sustained_s": 7,
                "candidate_sustained_s": ratio * 7,
            }
            for workload in ("warm", "eviction")
            for i in range(2)
        ],
        "loss": -999999,  # Neither failure nor an apparent win can use this.
    }
    sync_result_evidence(run, result)
    path = run.parent / "result.json"
    write_json(path, result)
    return path


def test_initialize_hashes_actual_source_and_keeps_external_history(prepared):
    source, run = prepared
    status = campaign.readiness(run)
    assert status["ready"]
    assert (run / "oracle/tennis3d/lifting/data.py").read_text() == "baseline = 2\n"
    assert status["noise_threshold_fraction"] == 0.02
    assert campaign.capture(run, source)["source_sha256"] == status["oracle_sha256"]
    assert not (source / "journal.jsonl").exists()


def test_reservation_is_not_reissued_and_consumes_failed_budget(prepared):
    source, run = prepared
    first = reserve(prepared)
    assert campaign.readiness(run)["active_trial"]["trial_id"] == "000001"
    with pytest.raises(campaign.CampaignError, match="remains active"):
        campaign.trial(run, source)
    campaign.decide(run, first["trial_id"], source, retain=False)
    assert (source / "tennis3d/lifting/data.py").read_text() == "baseline = 2\n"
    second = campaign.trial(run, source)
    campaign.decide(run, second["trial_id"], source, retain=False)
    assert campaign.readiness(run)["remaining_reserved_seconds"] == 0
    with pytest.raises(campaign.CampaignError, match="budget exhausted"):
        campaign.trial(run, source)


def test_retain_then_discard_restores_incumbent_and_preserves_results(prepared):
    source, run = prepared
    first = reserve(prepared)
    result = result_file(run, first)
    assert campaign.record(run, first["trial_id"], result)["eligible"]
    campaign.decide(run, first["trial_id"], source, retain=True)
    assert campaign.readiness(run)["best"]["source_sha256"] == first["source_sha256"]
    (source / "tennis3d/lifting/data.py").write_text("candidate = 999\n")
    second = campaign.trial(run, source)
    assert second["incumbent_sha256"] == first["source_sha256"]
    campaign.record(
        run, second["trial_id"], result_file(run, second, ratio=0.1, passed=False)
    )
    with pytest.raises(campaign.CampaignError, match="did not pass"):
        campaign.decide(run, second["trial_id"], source, retain=True)
    campaign.decide(run, second["trial_id"], source, retain=False)
    assert (source / "tennis3d/lifting/data.py").read_text() == "candidate = 3\n"
    assert (run / "trials/000002/result.json").is_file()


@pytest.mark.parametrize(
    "path", ["pyproject.toml", "uv.lock", "tennis3d/lifting/model.py"]
)
def test_frozen_math_dependencies_cannot_change(prepared, path):
    source, run = prepared
    (source / path).write_text("changed\n")
    with pytest.raises(campaign.CampaignError, match="frozen source"):
        campaign.capture(run, source)


def test_only_declared_new_helper_allowed_and_discard_removes_it(prepared):
    source, run = prepared
    helper = source / "tennis3d/throughput/pipeline.py"
    helper.parent.mkdir()
    helper.write_text("helper = True\n")
    reservation = campaign.trial(run, source)
    campaign.decide(run, reservation["trial_id"], source, retain=False)
    assert not helper.exists()
    helper.with_name("secret.py").write_text("import os\n")
    with pytest.raises(campaign.CampaignError, match="Undeclared"):
        campaign.capture(run, source)


@pytest.mark.parametrize(
    "kind",
    [
        "torn",
        "chain",
        "cursor",
        "best",
        "oracle",
        "snapshot",
        "protocol",
        "reservation",
    ],
)
def test_readiness_fails_closed_on_tampering(prepared, kind):
    _, run = prepared
    reservation = reserve(prepared)
    if kind == "torn":
        with (run / "journal.jsonl").open("ab") as stream:
            stream.write(b'{"sequence":2')
    elif kind == "chain":
        path = run / "journal.jsonl"
        path.write_bytes(path.read_bytes().replace(b'"sequence":1', b'"sequence":9'))
    elif kind == "cursor":
        write_json(run / "cursor.json", {"sequence": 0, "sha256": "0" * 64})
    elif kind == "best":
        write_json(run / "best.json", {"source_sha256": reservation["source_sha256"]})
    elif kind in ("oracle", "snapshot"):
        path = (
            run / "oracle" if kind == "oracle" else Path(reservation["candidate_path"])
        ) / "tennis3d/lifting/data.py"
        path.chmod(0o644)
        path.write_text("tampered\n")
    elif kind == "protocol":
        write_json(run / "protocol.json", {})
    else:
        write_json(run / "trials/000001/reservation.json", {})
    with pytest.raises(campaign.CampaignError):
        campaign.readiness(run)


@pytest.mark.parametrize("kind", ["hash", "identity", "symlink", "escape", "bare"])
def test_gate_evidence_is_bound_and_must_be_immutable_file(prepared, kind):
    _, run = prepared
    reservation = reserve(prepared)
    path = result_file(run, reservation)
    result = json.loads(path.read_text())
    evidence = run / "trials/000001/evidence.json"
    if kind == "hash":
        evidence.write_text("{}")
    elif kind == "identity":
        result["incumbent_sha256"] = "a" * 64
    elif kind == "symlink":
        raw = evidence.read_bytes()
        evidence.unlink()
        outside = run.parent / "outside.json"
        outside.write_bytes(raw)
        evidence.symlink_to(outside)
    elif kind == "escape":
        result["gates"]["evidence_file"] = "../../outside.json"
    else:
        result["gates"].pop("evidence_file")
    write_json(path, result)
    with pytest.raises(campaign.CampaignError):
        campaign.record(run, reservation["trial_id"], path)
    assert not (run / "trials/000001/result.json").exists()


def test_sealed_evidence_tampering_blocks_resume(prepared):
    _, run = prepared
    reservation = reserve(prepared)
    campaign.record(run, reservation["trial_id"], result_file(run, reservation))
    (run / "trials/000001/evidence.json").write_text("{}")
    with pytest.raises(campaign.CampaignError, match="evidence hash"):
        campaign.readiness(run)


@pytest.mark.parametrize(
    "ratios, eligible", [([0.99, 0.99], False), ([0.5, 1.1], False), ([0.9, 0.9], True)]
)
def test_noise_and_each_workload_control_retention(prepared, ratios, eligible):
    _, run = prepared
    reservation = reserve(prepared)
    path = result_file(run, reservation)
    result = json.loads(path.read_text())
    for row in result["measurements"]:
        ratio = ratios[row["workload"] == "eviction"]
        row["candidate_elapsed_s"] = 10 * ratio
        row["candidate_stream_s"] = 8 * ratio
        row["candidate_sustained_s"] = 7 * ratio
    sync_result_evidence(run, result)
    write_json(path, result)
    assert campaign.record(run, reservation["trial_id"], path)["eligible"] is eligible


def test_incomplete_paired_grid_and_duplicate_results_are_rejected(prepared):
    _, run = prepared
    reservation = reserve(prepared)
    path = result_file(run, reservation)
    result = json.loads(path.read_text())
    result["measurements"].pop()
    write_json(path, result)
    with pytest.raises(campaign.CampaignError, match="every frozen workload"):
        campaign.record(run, reservation["trial_id"], path)
    campaign.record(run, reservation["trial_id"], result_file(run, reservation))
    with pytest.raises(campaign.CampaignError, match="already has"):
        campaign.record(run, reservation["trial_id"], path)


def test_tracked_and_broken_symlinks_are_rejected(prepared):
    source, run = prepared
    path = source / "tennis3d/lifting/data.py"
    path.unlink()
    path.symlink_to(source / "missing")
    with pytest.raises(campaign.CampaignError, match="Symlink"):
        campaign.capture(run, source)


def test_cli_readiness(prepared):
    _, run = prepared
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tennis3d.throughput.campaign",
            "readiness",
            "--run",
            str(run),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["ready"]


def test_qualification_noise_is_frozen_and_controls_retention(prepared):
    source, old_run = prepared
    protocol_path = old_run.parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["qualification"]["warm"] = [[10, 12], [12, 10]]
    protocol["benchmark"] = qualify_metadata(source, protocol)
    write_json(protocol_path, protocol)
    run = old_run.parent / "noisy-campaign"
    status = campaign.initialize(source, run, protocol_path)
    assert status["noise_threshold_fraction"] == pytest.approx(
        (1.2 * 20.2 / 20) ** 0.5 - 1
    )
    reservation = campaign.trial(run, source)
    score = campaign.record(
        run, reservation["trial_id"], result_file(run, reservation, ratio=0.95)
    )
    assert not score["eligible"]


def test_evidence_with_wrong_identity_cannot_be_sealed(prepared):
    _, run = prepared
    reservation = reserve(prepared)
    path = result_file(run, reservation)
    result = json.loads(path.read_text())
    evidence_path = run / "trials/000001/evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["source_sha256"] = reservation["oracle_sha256"]
    write_json(evidence_path, evidence)
    result["gates"]["evidence_sha256"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    write_json(path, result)
    with pytest.raises(campaign.CampaignError, match="evidence identity"):
        campaign.record(run, reservation["trial_id"], path)


def test_restoration_validates_source_before_overwriting(prepared):
    source, run = prepared
    reservation = reserve(prepared)
    outside = run.parent / "outside.py"
    outside.write_text("keep = True\n")
    target = source / "tennis3d/lifting/data.py"
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(campaign.CampaignError, match="Symlink"):
        campaign.decide(run, reservation["trial_id"], source, retain=False)
    assert outside.read_text() == "keep = True\n"
    assert (
        campaign.readiness(run)["active_trial"]["trial_id"] == reservation["trial_id"]
    )


@pytest.mark.parametrize(
    "kind",
    [
        "source",
        "unqualified_model",
        "rules_hash",
        "rules",
        "descriptor",
        "passed",
        "independent",
        "qualification",
        "fixture",
        "lifecycle",
        "state",
        "main_trace",
        "device_trace",
        "missing_report",
        "duplicate_report",
        "report_timing",
        "environment",
    ],
)
def test_initialize_requires_source_bound_independent_qualification(prepared, kind):
    source, old_run = prepared
    protocol_path = old_run.parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    benchmark = protocol["benchmark"]
    if kind == "source":
        benchmark["source_sha256"] = "f" * 64
    elif kind == "unqualified_model":
        (source / "tennis3d/lifting/model.py").write_text("unqualified_math = True\n")
    elif kind == "rules_hash":
        benchmark["benchmark_sha256"] = "f" * 64
    elif kind == "rules":
        benchmark["rules"]["steps"] += 1
    elif kind == "descriptor":
        benchmark["descriptor_sha256"] = "f" * 64
    elif kind in ("passed", "independent"):
        benchmark[kind] = False
    elif kind == "qualification":
        benchmark["qualification"]["warm"][0][0] += 1
    elif kind == "fixture":
        benchmark["fixture_gates"]["reports"]["candidate-resume"]["sampler"] = "f" * 64
    elif kind == "lifecycle":
        benchmark["fixture_gates"]["lifecycle"]["candidate"]["traces"]["rejections"][0][
            1
        ] = "f" * 64
    elif kind == "state":
        benchmark["prepared"]["states"]["warm"]["cursor"] = 99
    elif kind == "main_trace":
        benchmark["expectations"]["warm"]["batch_sha256"].pop()
    elif kind == "device_trace":
        benchmark["expectations"]["warm"]["device_batch_sha256"].pop()
    elif kind == "missing_report":
        benchmark["reports"].pop()
    elif kind == "duplicate_report":
        benchmark["reports"].append(copy.deepcopy(benchmark["reports"][0]))
    elif kind == "report_timing":
        benchmark["reports"][0]["elapsed_s"] = 1
    else:
        benchmark["reports"][0]["environment"]["cuda"] = "changed"
    write_json(protocol_path, protocol)
    run = old_run.parent / "invalid-campaign"
    with pytest.raises(campaign.CampaignError):
        campaign.initialize(source, run, protocol_path)
    assert not run.exists()


def rewrite_evidence(run, result_path, mutate):
    result = json.loads(result_path.read_text())
    evidence_path = run / "trials" / result["trial_id"] / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    mutate(evidence)
    write_json(evidence_path, evidence)
    result["gates"]["evidence_sha256"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    write_json(result_path, result)


def test_one_second_result_cannot_override_ten_second_evidence(prepared):
    _, run = prepared
    reservation = reserve(prepared)
    path = result_file(run, reservation, ratio=1)
    result = json.loads(path.read_text())
    for row in result["measurements"]:
        row["candidate_elapsed_s"] = 1
    write_json(path, result)
    with pytest.raises(campaign.CampaignError, match="timing differs"):
        campaign.record(run, reservation["trial_id"], path)
    assert not (run / "trials/000001/result.json").exists()


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "duplicate",
        "parity",
        "device_parity",
        "software",
        "state",
        "data",
        "fixture",
        "lifecycle",
        "environment",
        "resources",
        "steps",
    ],
)
def test_successful_result_requires_full_independent_reports(prepared, kind):
    _, run = prepared
    reservation = reserve(prepared)
    path = result_file(run, reservation)

    def mutate(evidence):
        if kind == "missing":
            evidence["reports"].pop()
        elif kind == "duplicate":
            evidence["reports"].append(copy.deepcopy(evidence["reports"][0]))
        elif kind == "parity":
            evidence["reports"][0]["candidate"]["sampler_sha256"] = "f" * 64
        elif kind == "device_parity":
            evidence["reports"][0]["candidate"]["audit"]["device_batch_sha256"][0] = (
                "f" * 64
            )
        elif kind == "software":
            evidence["reports"][0]["candidate"]["software"][
                "cuda_matmul_allow_tf32"
            ] = True
        elif kind == "state":
            evidence["conditioning"]["eviction"]["states"]["warm"]["cursor"] = 99
        elif kind == "data":
            evidence["conditioning"]["eviction"]["data_identity"]["revision"] = "other"
        elif kind == "fixture":
            evidence["fixture_gates"]["reports"]["candidate-resume"]["optimizer"] = (
                "f" * 64
            )
        elif kind == "lifecycle":
            evidence["fixture_gates"]["lifecycle"]["candidate"]["traces"].pop(
                "rejections"
            )
        elif kind == "environment":
            evidence["reports"][0]["candidate"]["environment"]["torch"] = "other"
        elif kind == "resources":
            evidence["reports"][0]["candidate"]["ram_peak_bytes"] = 100 * 1024**3
        else:
            evidence["reports"][0]["candidate"]["steps"] = 1

    rewrite_evidence(run, path, mutate)
    with pytest.raises(campaign.CampaignError):
        campaign.record(run, reservation["trial_id"], path)
    assert not (run / "trials/000001/result.json").exists()


def configured_campaign(prepared, mutate):
    source, old_run = prepared
    protocol_path = old_run.parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    mutate(protocol)
    write_json(protocol_path, protocol)
    run = old_run.parent / "configured-campaign"
    campaign.initialize(source, run, protocol_path)
    return source, run


@pytest.mark.parametrize("missing", [False, True])
def test_optional_global_budgets_do_not_stop_campaign(prepared, missing):
    def configure(protocol):
        for key in ("trial_budget", "time_budget_s"):
            if missing:
                protocol.pop(key)
            else:
                protocol[key] = None
        protocol.pop("trial_job_budget_s")

    source, run = configured_campaign(prepared, configure)
    status = campaign.readiness(run)
    assert status["remaining_trials"] is None
    assert status["remaining_reserved_seconds"] is None
    assert status["recorded_compute_seconds"] is None
    assert status["recorded_queue_seconds"] is None
    for index in range(4):
        reservation = campaign.trial(run, source)
        assert reservation["job_budget_s"] == 3600
        assert campaign.readiness(run)["reserved_jobs"] == index + 1
        with pytest.raises(campaign.CampaignError, match="remains active"):
            campaign.trial(run, source)
        campaign.decide(run, reservation["trial_id"], source, retain=False)
    status = campaign.readiness(run)
    assert status["ready"]
    assert status["reserved_jobs"] == 4
    assert status["reserved_seconds"] == 14400
    assert status["recorded_jobs"] == 0


def test_explicit_trial_cap_still_applies_without_time_cap(prepared):
    source, run = configured_campaign(
        prepared, lambda protocol: protocol.update(trial_budget=1, time_budget_s=None)
    )
    reservation = campaign.trial(run, source)
    campaign.decide(run, reservation["trial_id"], source, retain=False)
    assert not campaign.readiness(run)["ready"]
    with pytest.raises(campaign.CampaignError, match="budget exhausted"):
        campaign.trial(run, source)


@pytest.mark.parametrize(
    "field,value",
    [
        ("trial_budget", 0),
        ("trial_budget", 2.5),
        ("trial_budget", True),
        ("time_budget_s", 0),
        ("time_budget_s", -1),
        ("trial_job_budget_s", 3601),
    ],
)
def test_invalid_explicit_budgets_are_rejected(prepared, field, value):
    with pytest.raises(campaign.CampaignError):
        configured_campaign(prepared, lambda protocol: protocol.update({field: value}))


def test_accounting_totals_distinguish_reserved_and_recorded_work(prepared):
    source, run = prepared
    reservation = reserve(prepared)
    path = result_file(run, reservation, passed=False)
    result = json.loads(path.read_text())
    result["status"] = "failed"
    result["job_accounting"] = {"compute_s": 25, "basis": "driver"}
    write_json(path, result)
    campaign.record(run, reservation["trial_id"], path)
    campaign.decide(run, reservation["trial_id"], source, retain=False)
    second = campaign.trial(run, source)
    campaign.decide(run, second["trial_id"], source, retain=False)
    status = campaign.readiness(run)
    assert status["reserved_seconds"] == 200
    assert status["recorded_compute_seconds"] == 25
    assert status["recorded_queue_seconds"] is None
    assert status["compute_accounted_jobs"] == 1
    assert status["queue_accounted_jobs"] == 0
    assert status["recorded_jobs"] == 1


def test_official_slurm_accounting_overrides_driver_and_is_sealed(prepared):
    _, run = prepared
    reservation = reserve(prepared)
    path = result_file(run, reservation)
    result = json.loads(path.read_text())
    result["job_accounting"] = {"compute_s": 25, "basis": "driver", "job_id": "123"}
    write_json(path, result)
    sidecar = run / "trials/000001/slurm-accounting.json"
    write_json(
        sidecar, {"compute_s": 30, "queue_s": 40, "basis": "slurm", "job_id": "123"}
    )
    campaign.record(run, reservation["trial_id"], path)
    sealed = json.loads((run / "trials/000001/result.json").read_text())
    assert sealed["job_accounting"]["driver"]["compute_s"] == 25
    assert sealed["job_accounting"]["basis"] == "slurm"
    status = campaign.readiness(run)
    assert status["recorded_compute_seconds"] == 30
    assert status["recorded_queue_seconds"] == 40
    sidecar.write_text("{}")
    with pytest.raises(campaign.CampaignError, match="sidecar changed"):
        campaign.readiness(run)


def test_unrelated_slurm_accounting_cannot_be_sealed(prepared):
    _, run = prepared
    reservation = reserve(prepared)
    path = result_file(run, reservation)
    result = json.loads(path.read_text())
    result["job_accounting"] = {"compute_s": 25, "basis": "driver", "job_id": "123"}
    write_json(path, result)
    write_json(
        run / "trials/000001/slurm-accounting.json",
        {"compute_s": 30, "queue_s": 40, "basis": "slurm", "job_id": "456"},
    )
    with pytest.raises(campaign.CampaignError, match="different or unbound job"):
        campaign.record(run, reservation["trial_id"], path)


def test_startup_improvement_without_sustained_gain_is_discarded(prepared):
    _, run = prepared
    reservation = reserve(prepared)
    path = result_file(run, reservation, ratio=0.8)
    result = json.loads(path.read_text())
    for row in result["measurements"]:
        row["candidate_sustained_s"] = row["incumbent_sustained_s"]
        row["candidate_stream_s"] = 7.5
    sync_result_evidence(run, result)
    write_json(path, result)
    score = campaign.record(run, reservation["trial_id"], path)
    assert score["geometric_stream_ratio"] < 1
    assert score["geometric_sustained_ratio"] == 1
    assert not score["eligible"]


def test_noise_guard_uses_noisy_total_reads_without_hiding_stream_gain(prepared):
    def configure(protocol):
        protocol["qualification"]["warm"][0][1] = 15
        protocol["benchmark"] = qualify_metadata(prepared[0], protocol)
        for report in protocol["benchmark"]["reports"]:
            report["stream_elapsed_s"] = 8
            report["sustained_elapsed_s"] = 7

    _, run = configured_campaign(prepared, configure)
    status = campaign.readiness(run)
    assert status["noise_threshold_fraction"] == 0.02
    assert status["per_workload_total_noise_fraction"]["warm"] == pytest.approx(0.5)
    reservation = campaign.trial(run, prepared[0])
    path = result_file(run, reservation, ratio=0.8)
    result = json.loads(path.read_text())
    for row in result["measurements"]:
        if row["workload"] == "warm":
            row["candidate_elapsed_s"] = 13
    sync_result_evidence(run, result)
    write_json(path, result)
    assert campaign.record(run, reservation["trial_id"], path)["eligible"]


@pytest.mark.parametrize("field", ["candidate_stream_s", "candidate_sustained_s"])
def test_secondary_timer_must_match_independent_evidence(prepared, field):
    _, run = prepared
    reservation = reserve(prepared)
    path = result_file(run, reservation)
    result = json.loads(path.read_text())
    result["measurements"][0][field] = 1
    write_json(path, result)
    with pytest.raises(campaign.CampaignError, match="timing differs"):
        campaign.record(run, reservation["trial_id"], path)


@pytest.mark.parametrize(
    "kind",
    [
        "source",
        "contract",
        "audit_hash",
        "optimizer_work",
        "missing_audit",
        "timed_batch_hashes",
        "policy",
    ],
)
def test_replay_binds_timed_run_to_exact_source_and_contract(prepared, kind):
    _, run = prepared
    reservation = reserve(prepared)
    path = result_file(run, reservation)

    def mutate(evidence):
        report = evidence["reports"][0]["candidate"]
        if kind == "source":
            report["audit"]["source_sha256"] = reservation["oracle_sha256"]
        elif kind == "contract":
            report["contract_sha256"] = "f" * 64
        elif kind == "audit_hash":
            report["audit_sha256"] = "f" * 64
        elif kind == "optimizer_work":
            report["audit"]["steps"] = 128
        elif kind == "missing_audit":
            report.pop("audit")
        elif kind == "timed_batch_hashes":
            report["batch_sha256"] = report["audit"]["batch_sha256"]
        else:
            evidence["fixture_gates"]["source_policy"]["candidate"][
                "bounded_reader"
            ] = False

    rewrite_evidence(run, path, mutate)
    with pytest.raises(campaign.CampaignError):
        campaign.record(run, reservation["trial_id"], path)


@pytest.mark.parametrize("value", [-1, True, float("inf"), float("nan")])
def test_invalid_job_accounting_is_rejected_without_sealing(prepared, value):
    _, run = prepared
    reservation = reserve(prepared)
    path = result_file(run, reservation)
    result = json.loads(path.read_text())
    result["job_accounting"] = {"compute_s": value, "basis": "driver"}
    write_json(path, result)
    with pytest.raises(campaign.CampaignError, match="finite nonnegative"):
        campaign.record(run, reservation["trial_id"], path)
    assert not (run / "trials/000001/result.json").exists()


def test_caller_cannot_replace_measured_noise_thresholds(prepared):
    def configure(protocol):
        protocol.update(
            noise_threshold_fraction=0,
            sustained_noise_threshold_fraction=0,
            per_workload_total_noise_fraction={"warm": 0},
            per_workload_sustained_noise_fraction={"warm": 0},
            per_workload_regression_fraction=100,
        )

    _, run = configured_campaign(prepared, configure)
    protocol = json.loads((run / "protocol.json").read_text())
    assert protocol["noise_threshold_fraction"] == 0.02
    assert protocol["sustained_noise_threshold_fraction"] == 0.02
    assert protocol["per_workload_total_noise_fraction"] == {
        "warm": 0.02,
        "eviction": 0.02,
    }
    assert "per_workload_regression_fraction" not in protocol


def test_aggregate_noise_uses_paired_equal_workload_log_ratios(prepared):
    def configure(protocol):
        for report in protocol["benchmark"]["reports"]:
            ratio = 1.2 if report["workload"] == "warm" else 1 / 1.2
            if report["side"] == "b":
                report["stream_elapsed_s"] = 6 * ratio
                report["sustained_elapsed_s"] = 5 * ratio
            else:
                report["stream_elapsed_s"] = 6
                report["sustained_elapsed_s"] = 5

    _, run = configured_campaign(prepared, configure)
    status = campaign.readiness(run)
    # Baseline pairs had zero aggregate change, but each workload was noisy.
    assert status["noise_threshold_fraction"] == 0.02
    assert status["sustained_noise_threshold_fraction"] == 0.02
    assert status["per_workload_sustained_noise_fraction"]["warm"] == pytest.approx(0.2)
    assert status["per_workload_sustained_noise_fraction"]["eviction"] == pytest.approx(
        0.2
    )


def test_trial_description_and_timestamps_are_persisted(prepared):
    source, run = prepared
    description = "Prefetch decoded batches to reduce stalls during shard transitions."
    reservation = campaign.trial(run, source, description=description)
    assert reservation["description"] == description
    assert reservation["reserved_at"].endswith("Z")
    offset = datetime.fromisoformat(reservation["reserved_at"]).utcoffset()
    assert offset is not None and offset.total_seconds() == 0
    persisted = json.loads((run / "trials/000001/reservation.json").read_text())
    assert persisted["description"] == description
    assert persisted["reserved_at"] == reservation["reserved_at"]
    assert campaign.readiness(run)["active_trial"]["description"] == description
    assert reservation["initialized_at"].endswith("Z")


@pytest.mark.parametrize(
    "description", ["", "  \n", "x" * 1001, "bad\0description", 42]
)
def test_invalid_description_never_reserves_trial(prepared, description):
    source, run = prepared
    with pytest.raises(campaign.CampaignError, match="description"):
        campaign.trial(run, source, description=description)
    assert campaign.readiness(run)["reserved_jobs"] == 0


def test_cli_trial_description_uses_current_interpreter(prepared):
    source, run = prepared
    description = "Keep source reads bounded while filling the prefetch queue."
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tennis3d.throughput.campaign",
            "trial",
            "--source",
            str(source),
            "--run",
            str(run),
            "--description",
            description,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["description"] == description


@pytest.mark.parametrize("passed", [True, False])
def test_record_seals_score_and_timestamp_for_read_only_progress(prepared, passed):
    _, run = prepared
    reservation = reserve(prepared)
    path = result_file(run, reservation, passed=passed)
    result = json.loads(path.read_text())
    result["score"] = {"eligible": True, "geometric_stream_ratio": 0.00001}
    result["recorded_at"] = "1900-01-01T00:00:00Z"
    write_json(path, result)
    score = campaign.record(run, reservation["trial_id"], path)
    sealed = json.loads((run / "trials/000001/result.json").read_text())
    assert sealed["score"] == score
    assert score["metric"] == "stream_s_with_sustained_gate"
    assert score["eligible"] is passed
    assert sealed["recorded_at"].endswith("Z")
    assert sealed["recorded_at"] != result["recorded_at"]
    assert campaign.readiness(run)["recorded_jobs"] == 1


def test_verification_recomputes_sealed_score(prepared):
    _, run = prepared
    reservation = reserve(prepared)
    campaign.record(run, reservation["trial_id"], result_file(run, reservation))
    with campaign._lock(run):
        state = campaign._state(run)
    path = run / "trials/000001/result.json"
    sealed = json.loads(path.read_text())
    sealed["score"]["geometric_stream_ratio"] = 0.00001
    path.write_bytes(campaign._json(sealed))
    # A matching content hash alone cannot validate a falsely computed score.
    trial = {
        **state["trials"][reservation["trial_id"]],
        "result_sha256": campaign._sha(path.read_bytes()),
    }
    with pytest.raises(campaign.CampaignError, match="Sealed score differs"):
        campaign._verify_result(run, trial, state["initial"])


@pytest.mark.parametrize("metric", ["stream", "sustained"])
def test_apparent_average_win_must_repeat_in_each_paired_round(prepared, metric):
    _, run = prepared
    reservation = reserve(prepared)
    path = result_file(run, reservation)
    result = json.loads(path.read_text())
    for row in result["measurements"]:
        lucky_or_regression = 0.5 if row["round"] == 0 else 1.01
        row["candidate_elapsed_s"] = 10
        row["candidate_stream_s"] = (
            8 * lucky_or_regression if metric == "stream" else 7.6
        )
        row["candidate_sustained_s"] = (
            7 * lucky_or_regression if metric == "sustained" else 3.5
        )
    sync_result_evidence(run, result)
    write_json(path, result)
    score = campaign.record(run, reservation["trial_id"], path)
    assert score["geometric_stream_ratio"] < 1 / (1 + score["noise_threshold_fraction"])
    assert score["geometric_sustained_ratio"] < 1 / (
        1 + score["sustained_noise_threshold_fraction"]
    )
    assert score[f"per_round_{metric}_ratio"]["1"] == pytest.approx(1.01)
    assert not score["eligible"]
    sealed = json.loads((run / "trials/000001/result.json").read_text())
    assert sealed["score"] == score
    assert set(score["per_round_stream_ratio"]) == {"0", "1"}
    assert set(score["per_round_sustained_ratio"]) == {"0", "1"}


def test_all_paired_rounds_repeat_stream_and_sustained_win(prepared):
    _, run = prepared
    reservation = reserve(prepared)
    path = result_file(run, reservation)
    result = json.loads(path.read_text())
    for row in result["measurements"]:
        ratio = (0.85, 0.9)[row["round"]]
        for field in ("elapsed_s", "stream_s", "sustained_s"):
            row[f"candidate_{field}"] = ratio * row[f"incumbent_{field}"]
    sync_result_evidence(run, result)
    write_json(path, result)
    score = campaign.record(run, reservation["trial_id"], path)
    assert score["eligible"]
    assert score["per_round_stream_ratio"] == pytest.approx({"0": 0.85, "1": 0.9})
    assert score["per_round_sustained_ratio"] == pytest.approx({"0": 0.85, "1": 0.9})


def test_source_derived_eviction_template_is_minimal_and_under_capacity(prepared):
    _, run = prepared
    benchmark = json.loads((run / "protocol.json").read_text())["benchmark"]
    template = benchmark["prepared"]["cache_templates"]["evicted"]
    assert len(template["artifacts"]) == 11  # Pinned manifest and ten fillers.
    assert [value["path"] for value in template["artifacts"][1:]] == [
        f"dataset/part-{index:03d}.parquet" for index in range(2, 12)
    ]
    assert template["initial_bytes"] <= template["cache_bytes"]
    assert (
        template["initial_bytes"] + template["target_bytes"] > template["cache_bytes"]
    )
    assert (
        template["initial_bytes"]
        - template["artifacts"][-1]["size_bytes"]
        + template["target_bytes"]
        <= template["cache_bytes"]
    )
    assert template["overflow_bytes"] > 0


@pytest.mark.parametrize(
    "kind",
    [
        "extra_filler",
        "missing_filler",
        "different_prefix",
        "capacity",
        "availability",
        "target_presence",
        "target_order",
        "target_size",
        "survivor_target",
        "foreign_survivor",
        "stress_count",
        "missing_pin",
        "missing_template",
        "duplicate_shard",
        "ordering_policy",
    ],
)
def test_qualification_rejects_changed_cache_template_or_source_proof(prepared, kind):
    def mutate(protocol):
        benchmark = protocol["benchmark"]
        prepared_data = benchmark["prepared"]
        template = prepared_data["cache_templates"]["evicted"]
        if kind == "extra_filler":
            template["artifacts"].append(copy.deepcopy(prepared_data["shards"][12]))
        elif kind == "missing_filler":
            template["artifacts"].pop()
        elif kind == "different_prefix":
            template["artifacts"][1] = copy.deepcopy(prepared_data["shards"][12])
        elif kind == "capacity":
            template["cache_bytes"] += 1
        elif kind == "availability":
            template["available_bytes"] += 1
        elif kind == "target_presence":
            template["targets_present"] = True
        elif kind == "target_order":
            template["target_inventory"].reverse()
        elif kind == "target_size":
            template["target_inventory"][0]["size_bytes"] += 1
        elif kind == "survivor_target":
            prepared_data["cache_keys"].append(template["target_inventory"][0]["key"])
        elif kind == "foreign_survivor":
            prepared_data["cache_keys"].append("f" * 64)
        elif kind == "stress_count":
            prepared_data["stress_shards"].pop()
        elif kind == "missing_pin":
            manifest_path = template["artifacts"][0]["path"]
            manifest_key = next(
                value["key"]
                for value in template["inventory"]
                if value["path"] == manifest_path
            )
            prepared_data["cache_keys"].remove(manifest_key)
        elif kind == "missing_template":
            prepared_data["cache_templates"].pop("warm")
        elif kind == "duplicate_shard":
            prepared_data["shards"].append(copy.deepcopy(prepared_data["shards"][0]))
        else:
            benchmark["rules"]["pair_order"] = "round-major"

    with pytest.raises(campaign.CampaignError):
        configured_campaign(prepared, mutate)


def refresh_cache_hashes(report):
    report["cache_condition_sha256"] = campaign._sha(
        campaign._json(report["cache_condition"])
    )
    if "audit" in report:
        report["audit"]["cache_condition_sha256"] = campaign._sha(
            campaign._json(report["audit"]["cache_condition"])
        )
        report["audit_sha256"] = campaign._sha(campaign._json(report["audit"]))


@pytest.mark.parametrize(
    "kind",
    [
        "template_hash",
        "initial_inventory",
        "initial_bytes",
        "available_bytes",
        "target_presence",
        "missing_read",
        "missing_fetch",
        "forged_reads",
        "final_size",
        "foreign_final",
        "duplicate_final",
        "unordered_final",
        "over_capacity",
        "missing_manifest",
        "false_removal",
        "audit_initial",
    ],
)
def test_rehashed_trial_proofs_cannot_change_cache_contract(prepared, kind):
    _, run = prepared
    reservation = reserve(prepared)
    path = result_file(run, reservation)

    def mutate(evidence):
        report = evidence["reports"][0]["candidate"]
        proof = report["cache_condition"]
        assert proof["condition"] == "evicted"
        if kind == "template_hash":
            report["cache_template_sha256"] = "f" * 64
        elif kind == "initial_inventory":
            proof["initial_inventory"].pop()
        elif kind == "initial_bytes":
            proof["initial_bytes"] += 1
        elif kind == "available_bytes":
            proof["available_bytes"] += 1
        elif kind == "target_presence":
            proof["target_present"][0] = True
        elif kind == "missing_read":
            report["read_artifact_paths"].pop()
        elif kind == "missing_fetch":
            report["fetched_artifact_paths"].pop()
        elif kind == "forged_reads":
            proof["read_target_keys"] = []
        elif kind == "final_size":
            proof["final_inventory"][0]["size_bytes"] += 1
        elif kind == "foreign_final":
            proof["final_inventory"].append({"key": "f" * 64, "size_bytes": 1})
        elif kind == "duplicate_final":
            proof["final_inventory"].append(copy.deepcopy(proof["final_inventory"][0]))
        elif kind == "unordered_final":
            proof["final_inventory"].reverse()
        elif kind == "over_capacity":
            benchmark = json.loads((run / "protocol.json").read_text())["benchmark"]
            template = benchmark["prepared"]["cache_templates"]["evicted"]
            all_rows = [*template["inventory"], *template["target_inventory"]]
            proof["final_inventory"] = sorted(
                (
                    {key: value[key] for key in ("key", "size_bytes")}
                    for value in all_rows
                ),
                key=lambda value: value["key"],
            )
            proof["removed_initial_keys"] = proof["removed_target_keys"] = []
        elif kind == "missing_manifest":
            benchmark = json.loads((run / "protocol.json").read_text())["benchmark"]
            template = benchmark["prepared"]["cache_templates"]["evicted"]
            manifest_key = next(
                value["key"]
                for value in template["inventory"]
                if value["path"] == template["artifacts"][0]["path"]
            )
            proof["final_inventory"] = [
                value
                for value in proof["final_inventory"]
                if value["key"] != manifest_key
            ]
        elif kind == "false_removal":
            proof["removed_initial_keys"] = []
        else:
            report["audit"]["cache_condition"]["initial_bytes"] += 1
        refresh_cache_hashes(report)

    rewrite_evidence(run, path, mutate)
    with pytest.raises(campaign.CampaignError):
        campaign.record(run, reservation["trial_id"], path)
    assert not (run / "trials/000001/result.json").exists()


def test_candidate_may_evict_consumed_target_instead_of_filler(prepared):
    _, run = prepared
    reservation = reserve(prepared)
    path = result_file(run, reservation)
    benchmark = json.loads((run / "protocol.json").read_text())["benchmark"]

    def mutate(evidence):
        for pair in evidence["reports"]:
            if pair["workload"] != "eviction":
                continue
            candidate = pair["candidate"]
            changed = cache_report(benchmark, "eviction", remove_target=True)
            candidate.update(changed)
            candidate["audit"].update(copy.deepcopy(changed))
            candidate["audit_sha256"] = campaign._sha(
                campaign._json(candidate["audit"])
            )
            assert not candidate["cache_condition"]["removed_initial_keys"]
            assert candidate["cache_condition"]["removed_target_keys"]

    rewrite_evidence(run, path, mutate)
    assert campaign.record(run, reservation["trial_id"], path)["eligible"]


def test_baseline_qualification_must_demonstrate_initial_filler_removal(prepared):
    def mutate(protocol):
        benchmark = protocol["benchmark"]
        changed = cache_report(benchmark, "eviction", remove_target=True)
        benchmark["audits"]["eviction"].update(changed)
        for report in benchmark["reports"]:
            if report["workload"] == "eviction":
                report.update(copy.deepcopy(changed))
                report["audit"] = copy.deepcopy(benchmark["audits"]["eviction"])
                report["audit_sha256"] = campaign._sha(campaign._json(report["audit"]))

    with pytest.raises(campaign.CampaignError, match="actual cache removal"):
        configured_campaign(prepared, mutate)


@pytest.mark.parametrize(
    "kind",
    [
        "qualification_labels",
        "qualification_chronology",
        "trial_labels",
        "trial_chronology",
        "round_major",
    ],
)
def test_frozen_grouped_ab_ba_order_cannot_be_relabelled(prepared, kind):
    if kind.startswith("qualification"):

        def mutate(protocol):
            benchmark = protocol["benchmark"]
            if kind == "qualification_labels":
                benchmark["execution_order"][0], benchmark["execution_order"][1] = (
                    benchmark["execution_order"][1],
                    benchmark["execution_order"][0],
                )
            else:
                benchmark["reports"].reverse()

        with pytest.raises(campaign.CampaignError, match="AB/BA order"):
            configured_campaign(prepared, mutate)
        return
    _, run = prepared
    reservation = reserve(prepared)
    path = result_file(run, reservation)

    def mutate(evidence):
        if kind == "trial_labels":
            evidence["execution_order"][0], evidence["execution_order"][1] = (
                evidence["execution_order"][1],
                evidence["execution_order"][0],
            )
        elif kind == "trial_chronology":
            evidence["reports"].reverse()
        else:
            evidence["reports"] = sorted(
                evidence["reports"], key=lambda pair: (pair["round"], pair["workload"])
            )

    rewrite_evidence(run, path, mutate)
    with pytest.raises(campaign.CampaignError, match="AB/BA order"):
        campaign.record(run, reservation["trial_id"], path)
