"""Frozen oracle subprocesses detect changed math and actual resume state."""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch")

from tennis3d.throughput.evaluate import fixture_gates, worker

ROOT = Path(__file__).resolve().parents[1]
RULES = {"prefetch_depth": 2, "worker_timeout_seconds": 120}


def test_frozen_cpu_update_and_resume_match_maintained_trainer(tmp_path):
    result = fixture_gates(ROOT, ROOT, tmp_path / "gates", RULES)
    assert set(result["maintained_update_state"]) == {
        "model",
        "ema",
        "optimizer",
        "sampler",
        "rng",
    }
    assert (
        result["reports"]["candidate-resume"]["model"]
        == result["reports"]["maintained"]["model"]
    )


def test_independent_oracle_rejects_candidate_features(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    for directory in ("tennis3d", "configs", "tests/fixtures"):
        shutil.copytree(
            ROOT / directory,
            candidate / directory,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    features = candidate / "tennis3d/lifting/features.py"
    text = features.read_text().replace(
        "ball[observed] / size", "(ball[observed] / size) + .001"
    )
    features.write_text(text)
    with pytest.raises(ValueError, match="independent maintained-trainer"):
        fixture_gates(ROOT, candidate, tmp_path / "gates", RULES)


def test_worker_rejects_environment_importing_another_checkout(tmp_path):
    # Worker itself proves its selected package root, independent of cwd.
    import subprocess
    import sys

    request = tmp_path / "request.json"
    request.write_text('{"source":"/wrong/source"}')
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tennis3d/throughput/worker.py"),
            "fixture",
            str(request),
            str(tmp_path / "report.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "different source checkout" in completed.stderr


def test_fixture_actual_checkpoint_resume_has_identical_rng(tmp_path):
    baseline = worker(
        ROOT,
        ROOT,
        "fixture",
        {"depth": 0, "output_dir": str(tmp_path / "baseline")},
        tmp_path / "baseline.json",
        120,
    )
    resumed = worker(
        ROOT,
        ROOT,
        "fixture",
        {"depth": 2, "resume": True, "output_dir": str(tmp_path / "resumed")},
        tmp_path / "resumed.json",
        120,
    )
    for field in ("model", "ema", "optimizer", "sampler", "rng"):
        assert baseline[field] == resumed[field]


def test_independent_oracle_rejects_changed_rejection_counter(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    for directory in ("tennis3d", "configs", "tests/fixtures"):
        shutil.copytree(
            ROOT / directory,
            candidate / directory,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    path = candidate / "tennis3d/lifting/data.py"
    path.write_text(
        path.read_text().replace("self.rejected += 1", "self.rejected += 2")
    )
    with pytest.raises(ValueError, match="independent rejection/lifecycle"):
        fixture_gates(ROOT, candidate, tmp_path / "gates", RULES)


@pytest.mark.parametrize("mutation", ["redirect", "unbounded"])
def test_independent_source_policy_rejects_actual_resource_changes(tmp_path, mutation):
    import subprocess

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    for directory in ("tennis3d", "configs", "tests/fixtures"):
        shutil.copytree(
            ROOT / directory,
            candidate / directory,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    path = candidate / "tennis3d/datasets/source.py"
    text = path.read_text()
    if mutation == "redirect":
        text = text.replace(
            "cache_path = Path(self.spec.cache_dir)",
            'cache_path = Path(self.spec.cache_dir).with_name("unowned-cache")',
        )
    else:
        text = text.replace(
            "max_group_bytes=self.spec.max_group_bytes", "max_group_bytes=None"
        )
    path.write_text(text)
    with pytest.raises(subprocess.CalledProcessError):
        fixture_gates(ROOT, candidate, tmp_path / "gates", RULES)


def test_parent_clock_and_os_memory_replace_candidate_diagnostics(tmp_path):
    oracle = tmp_path / "oracle"
    script = oracle / "tennis3d/throughput/worker.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import json,sys,time\n"
        "from pathlib import Path\n"
        "time.sleep(.08)\n"
        "print('THROUGHPUT_BOUNDARY stream',flush=True)\n"
        "time.sleep(.08)\n"
        "print('THROUGHPUT_BOUNDARY sustained',flush=True)\n"
        "time.sleep(.08)\n"
        "print('THROUGHPUT_BOUNDARY complete',flush=True)\n"
        "time.sleep(.25)\n"
        "Path(sys.argv[3]).write_text(json.dumps({'elapsed_s':.001,"
        "'stream_elapsed_s':.001,'sustained_elapsed_s':.001,"
        "'ram_peak_bytes':0,'windows':48,'windows_per_second':48000}))\n"
    )
    result = worker(oracle, oracle, "measure", {}, tmp_path / "timed.json", 5)
    assert result["worker_elapsed_s"] == 0.001
    assert result["elapsed_s"] >= 0.48
    assert result["stream_elapsed_s"] >= 0.40
    assert result["sustained_elapsed_s"] >= 0.32
    assert result["post_completion_s"] >= 0.24
    assert result["completed_work_s"] < result["elapsed_s"] - 0.24
    assert result["worker_ram_peak_bytes"] == 0
    assert result["ram_peak_bytes"] > 0
    assert result["windows_per_second"] == 48 / result["elapsed_s"]


def test_worker_timeout_reaps_owned_child(tmp_path):
    import os
    import subprocess

    oracle = tmp_path / "oracle"
    script = oracle / "tennis3d/throughput/worker.py"
    script.parent.mkdir(parents=True)
    pid = tmp_path / "pid"
    script.write_text(
        "import os,time\n"
        "from pathlib import Path\n"
        f"Path({str(pid)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        worker(oracle, oracle, "policy", {}, tmp_path / "timeout.json", 1)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid.read_text()), 0)


@pytest.mark.parametrize("mutation", ["path", "admission"])
def test_reader_policy_is_checked_on_producer_read(tmp_path, mutation):
    import json
    from types import SimpleNamespace

    from tennis3d.datasets.source import Artifact, json_sha256
    from tennis3d.throughput.worker import install_source_guards

    raw = json.loads(
        (ROOT / "configs/stage4-preparation-main-v1/train.source.json").read_text()
    )
    raw["cache_dir"] = str(tmp_path / "source-cache")
    artifact = Artifact.model_validate(raw["manifest"])
    path = Path(raw["cache_dir"]) / json_sha256(
        {
            "repo": raw["repo_id"],
            "commit": raw["revision"],
            **artifact.model_dump(include={"path", "sha256", "size_bytes"}),
        }
    )
    decoded = []

    class Reader:
        def __init__(self, path, shard, *, provenance, max_group_bytes):
            self.path = path
            self.max_group_bytes = max_group_bytes

        def read(self, index):
            decoded.append(index)
            return index

        def close(self):
            pass

    module = SimpleNamespace(VerifiedCache=object, ShardReader=Reader)
    install_source_guards(module, raw, {"cache_bytes": raw["cache_bytes"]})
    reader = module.ShardReader(
        path, artifact, provenance=None, max_group_bytes=raw["max_group_bytes"]
    )
    assert reader.read(0) == 0
    if mutation == "path":
        reader.path = tmp_path / "unowned"
    else:
        reader.max_group_bytes = None
    with pytest.raises(ValueError, match="owned admission"):
        reader.read(1)
    assert decoded == [0]


def test_candidate_cannot_replace_frozen_boundary_writes(tmp_path):
    import subprocess

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    for directory in ("tennis3d", "configs", "tests/fixtures"):
        shutil.copytree(
            ROOT / directory,
            candidate / directory,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    path = candidate / "tennis3d/throughput/pipeline.py"
    path.write_text(path.read_text() + "\nimport os\nos.write = lambda *args: 0\n")
    with pytest.raises(subprocess.CalledProcessError):
        worker(ROOT, candidate, "policy", {}, tmp_path / "candidate.json", 120)
    assert (
        "changed frozen timing notifications"
        in (tmp_path / "candidate.log").read_text()
    )


def test_parent_cache_proof_distinguishes_cold_reads_and_eviction(tmp_path):
    from tennis3d.throughput.evaluate import condition_proof

    cache = tmp_path / "cache"
    cache.mkdir()
    filler, first, second = "a" * 64, "b" * 64, "c" * 64
    (cache / first).write_bytes(b"first")
    (cache / second).write_bytes(b"second")
    initial = [{"key": filler, "size_bytes": 20}]
    template = {
        "cache_bytes": 30,
        "target_inventory": [
            {"key": first, "path": "first"},
            {"key": second, "path": "second"},
        ],
    }
    report = {
        "read_artifact_paths": ["first", "second"],
        "fetched_artifact_paths": ["first", "second"],
    }
    proof = condition_proof(cache, template, {"condition": "evicted"}, report, initial)
    assert proof["removed_initial_keys"] == [filler]
    assert proof["target_present"] == [False, False]
    assert proof["available_bytes"] == 10
    # An optimized cache can instead retain the filler and evict a consumed target.
    (cache / filler).write_bytes(b"x" * 20)
    (cache / first).unlink()
    proof = condition_proof(cache, template, {"condition": "evicted"}, report, initial)
    assert proof["removed_initial_keys"] == []
    assert proof["removed_target_keys"] == [first]
    report["fetched_artifact_paths"] = ["first"]
    with pytest.raises(ValueError, match="cold reads and cache pressure"):
        condition_proof(cache, template, {"condition": "evicted"}, report, initial)


def test_warm_transition_proof_requires_every_declared_shard(tmp_path):
    from tennis3d.throughput.evaluate import condition_proof

    key = "d" * 64
    (tmp_path / key).write_bytes(b"fixture")
    initial = [{"key": key, "size_bytes": 7}]
    template = {
        "cache_bytes": 30,
        "target_inventory": [{"key": key, "path": "transition"}],
    }
    with pytest.raises(ValueError, match="every declared target shard"):
        condition_proof(
            tmp_path,
            template,
            {"condition": "warm"},
            {"read_artifact_paths": [], "fetched_artifact_paths": []},
            initial,
        )
    proof = condition_proof(
        tmp_path,
        template,
        {"condition": "warm"},
        {"read_artifact_paths": ["transition"], "fetched_artifact_paths": []},
        initial,
    )
    assert proof["target_present"] == [True]
    assert proof["removed_initial_keys"] == []


def test_parent_inventory_rejects_unowned_or_partial_artifacts(tmp_path):
    from tennis3d.throughput.evaluate import cache_inventory

    target = tmp_path / ("a" * 64)
    target.write_bytes(b"fixture")
    link = tmp_path / ("b" * 64)
    link.symlink_to(target)
    with pytest.raises(ValueError, match="unowned path"):
        cache_inventory(tmp_path)
    link.unlink()
    partial = tmp_path / ("c" * 64 + ".partial")
    partial.write_bytes(b"partial")
    with pytest.raises(ValueError, match="incomplete artifact"):
        cache_inventory(tmp_path)


def test_qualification_rejects_replay_without_filler_eviction_before_baselines(
    tmp_path, monkeypatch
):
    from tennis3d.throughput import evaluate as e

    rules = {"workloads": [{"id": "evicted", "condition": "evicted"}]}
    monkeypatch.setattr(e.campaign, "_inventory", lambda _: {})
    monkeypatch.setattr(e.campaign, "_snapshot", lambda *args: "a" * 64)
    monkeypatch.setattr(e.campaign, "_load", lambda _: rules)
    monkeypatch.setattr(e, "prepare_caches", lambda *args: ({}, {"evicted": None}, 0.0))
    monkeypatch.setattr(e, "fixture_gates", lambda *args: {})
    calls = []

    def replay(*args, **kwargs):
        calls.append(kwargs)
        return {
            "cache_condition": {
                "removed_initial_keys": [],
                "removed_target_keys": ["b" * 64],
            }
        }

    monkeypatch.setattr(e, "measurement", replay)
    with pytest.raises(
        ValueError, match="oracle replay did not demonstrate filler eviction"
    ):
        e.qualify(ROOT, tmp_path / "qualification")
    assert calls == [{"depth": 0, "auditing": True}]
    assert not (tmp_path / "qualification/qualification.json").exists()
