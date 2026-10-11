"""Model-free pilot refusal, sampler, deadline and bounded reload checks."""

import copy
import fcntl
import os
import sys
from inspect import unwrap

import pytest

from scripts.producer_pilot import (
    cpu_lock,
    disk_usage,
    inspect_parquet,
    run_bounded,
    validate_pilot,
)
from tennis3d.synthetic.config import SimulationConfig
from tennis3d.synthetic.producer_config import ProducerConfig


def test_lock_refuses_other_open_description_and_releases(tmp_path):
    path = tmp_path / "cpu.lock"
    with cpu_lock(path) as evidence:
        assert evidence["holder_pid"] == os.getpid()
        with path.open("a+b") as other, pytest.raises(BlockingIOError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="already held"), cpu_lock(path):
            pass
    with cpu_lock(path):
        pass


def test_disk_union_counts_temporary_files_once_and_rejects_symlinks(tmp_path):
    output = tmp_path / "output"
    pool = output / "pools"
    pool.mkdir(parents=True)
    (pool / "pool.npz").write_bytes(b"x" * 100)
    (output / ".partial.parquet").write_bytes(b"x" * 37)
    external = tmp_path / "external"
    external.mkdir()
    (external / "pool.npz").write_bytes(b"x" * 23)
    measured = disk_usage((output, pool, external))
    assert measured["logical_bytes"] == 160
    assert measured["allocated_bytes"] >= 160
    (output / "symlink").symlink_to(external)
    with pytest.raises(ValueError, match="symlinks"):
        disk_usage((output, external))


@pytest.mark.parametrize(
    "cfg, execution, drain, reason",
    [
        (SimulationConfig(train_rallies=513), ProducerConfig(), 45, "512"),
        (SimulationConfig(), ProducerConfig(workers=3), 45, "one or two"),
        (
            SimulationConfig(),
            ProducerConfig(repo_id="someone/repo", revision="main"),
            45,
            "local only",
        ),
        (SimulationConfig(), ProducerConfig(drain_seconds=50), 45, "drain exceeds"),
    ],
)
def test_pilot_refuses_main_parallel_publication_and_unbounded_drain(
    cfg, execution, drain, reason
):
    with pytest.raises(ValueError, match=reason):
        validate_pilot(cfg, execution, drain)


def run_fixture(tmp_path, code, *, timeout=2, drain=0.2):
    return run_bounded(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=os.environ.copy(),
        roots=(tmp_path / "output",),
        log_path=tmp_path / "stdout.log",
        samples_path=tmp_path / "samples.jsonl",
        timeout=timeout,
        drain=drain,
        interval=0.02,
    )


def test_already_reaped_child_keeps_missing_measurements_null(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import producer_pilot

    monkeypatch.setattr(
        producer_pilot.subprocess,
        "Popen",
        lambda *a, **kw: SimpleNamespace(pid=123456789, returncode=None),
    )
    monkeypatch.setattr(producer_pilot, "signal_group", lambda *a: None)

    def reaped(*args):
        raise ChildProcessError("already reaped")

    monkeypatch.setattr(producer_pilot.os, "wait4", reaped)
    result = run_fixture(tmp_path, "pass")
    assert result["returncode"] is None
    assert result["wait4_cpu_seconds_including_reaped_descendants"] is None
    assert result["wait4_ru_maxrss_kib"] is None


def test_launch_failure_restores_handlers_without_masking_cause(tmp_path, monkeypatch):
    import signal

    from scripts import producer_pilot

    previous = signal.getsignal(signal.SIGTERM)

    def fail(*args, **kwargs):
        raise OSError("original launch failed")

    monkeypatch.setattr(producer_pilot.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="original launch failed"):
        run_fixture(tmp_path, "pass")
    assert signal.getsignal(signal.SIGTERM) == previous


def test_sampler_includes_child_memory_and_waited_cpu(tmp_path):
    child = "import time; blob=bytearray(4*1024*1024); end=time.monotonic()+0.18\nwhile time.monotonic()<end: sum(range(300))"
    code = f"import subprocess,sys; subprocess.run([sys.executable,'-c',{child!r}],check=True)"
    measured = run_fixture(tmp_path, code)
    assert measured["returncode"] == 0
    assert measured["wait4_cpu_seconds_including_reaped_descendants"] > 0.05
    assert measured["resources"]["sampled_peaks"]["processes"] >= 3
    assert measured["resources"]["sampled_peaks"]["pss_bytes"] > 4 << 20
    assert (
        measured["resources"]["sampled_peaks"]["rss_bytes"]
        >= measured["resources"]["sampled_peaks"]["pss_bytes"]
    )
    assert not measured["hard_kill_sent"]


def test_wall_deadline_kills_uncooperative_process_group(tmp_path):
    child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(10)"
    code = f"import subprocess,sys,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(10)"
    measured = run_fixture(tmp_path, code, timeout=0.35, drain=0.12)
    assert measured["returncode"] == -9
    assert measured["stop_reason"] == "generation_deadline"
    assert measured["hard_kill_sent"]
    assert 0.3 <= measured["wall_seconds"] < 1.5


def test_term_drains_with_incomplete_exit_instead_of_claiming_success(tmp_path):
    code = "import signal,time,sys; signal.signal(signal.SIGTERM,lambda *_: sys.exit(4)); time.sleep(10)"
    measured = run_fixture(tmp_path, code, timeout=0.5, drain=0.2)
    assert measured["returncode"] == 4
    assert measured["stop_requested"]
    assert not measured["hard_kill_sent"]


def test_bounded_reload_reads_only_selected_groups(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    # Reuse the existing numeric fixture, with no physics or model invocation.
    from test_dataset_parquet import sample

    from tennis3d.datasets.io import ShardReader, ShardWriter, write_manifest
    from tennis3d.datasets.schema import Manifest, count_records

    record, provenance = unwrap(sample)()
    files = tmp_path / "files"
    path = files / "dataset/shards/000000.parquet"
    with ShardWriter(
        path,
        provenance,
        row_group_rallies=2,
        relative_path="dataset/shards/000000.parquet",
    ) as writer:
        for i in range(5):
            other = copy.deepcopy(record)
            other.metadata = other.metadata.model_copy(
                update={"rally_id": f"rally-{i}"}
            )
            writer.append(other)
        shard = writer.close()
    manifest = Manifest(
        state="complete",
        provenance=provenance,
        shards=[shard],
        expected=count_records([shard]),
    )
    write_manifest(files / "dataset/manifest.json", manifest)
    reads = []
    original = ShardReader.read

    def tracked_read(self, row):
        reads.append(row)
        return original(self, row)

    monkeypatch.setattr(ShardReader, "read", tracked_read)
    execution = ProducerConfig(row_group_rallies=2)
    measured = inspect_parquet(tmp_path, execution, max_groups=1, max_rows=2)
    assert measured["status"] == "measured"
    assert measured["physical_rallies"] == 5
    assert measured["row_groups"] == 3
    assert measured["shard_bytes"] == path.stat().st_size
    assert len(measured["bounded_reload"]) == 1
    assert len(reads) <= 2
    assert measured["bounded_reload"][0]["typed_numeric_bytes"] > 0
    assert measured["bounded_reload"][0]["arrow_buffer_bytes"] > 0
    assert measured["bounded_reload"][0]["retained_python_group_bytes_estimate"] > 0
    assert measured["local_reload_resource_snapshots"]["sample_count"] >= 3


def test_metadata_remains_actual_when_reload_admission_is_too_small(tmp_path):
    # No manifest means no successful reload evidence, even if output exists.
    measured = inspect_parquet(tmp_path, ProducerConfig(), max_group_bytes=1)
    assert measured == {"status": "blocked", "reason": "no completed local manifest"}
