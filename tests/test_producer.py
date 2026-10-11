"""Bounded coordinator checks reuse a small, genuinely simulated dataset."""

import hashlib
import threading
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("pyarrow")

from tennis3d.artifacts import file_sha256, read_json
from tennis3d.datasets.io import iter_rallies, read_manifest
from tennis3d.datasets.publication import RemoteMissing
from tennis3d.synthetic import KINDS, producer, producer_records
from tennis3d.synthetic.config import SimulationConfig
from tennis3d.synthetic.producer_config import ProducerConfig
from tennis3d.synthetic.producer_parallel import OrderedWorkers, StopController, Stopped
from tennis3d.synthetic.producer_records import (
    AdmissionError,
    close_worker_cache,
    plan_rallies,
    produce_record,
)
from tennis3d.synthetic.producer_recovery import create_bundle, restore_bundle
from tennis3d.synthetic.producer_state import load
from tennis3d.synthetic.rally import Rally


class SerialWorkers:
    """Use the exact task functions in the coordinator process for parity."""

    def __init__(self, workers, inflight, stop):
        self.stop = stop

    def map(self, function, tasks):
        for task in tasks:
            if self.stop.requested:
                raise Stopped("serial test stopped")
            yield function(task)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        close_worker_cache()


def records(output):
    manifest = read_manifest(output / "files/dataset/manifest.json")
    return [
        record
        for shard in manifest.shards
        for record in iter_rallies(
            output / "files" / shard.path, shard, provenance=manifest.provenance
        )
    ]


def content_hashes(output):
    """Compare complete bytes, including membership, provenance and diagnostics."""
    files = output / "files/dataset"
    return {
        p.relative_to(files).as_posix(): file_sha256(p)
        for p in files.rglob("*")
        if p.is_file()
    }


@pytest.fixture(scope="module")
def simulated(tmp_path_factory):
    root = tmp_path_factory.mktemp("producer-small")
    cfg = SimulationConfig(
        train_rallies=2,
        val_rallies=1,
        test_rallies=1,
        pool_size=4,
        min_returns=1,
        max_returns=1,
        views_per_rally=2,
    )
    execution = ProducerConfig(workers=1, inflight=2, shard_rallies=2)
    patch = pytest.MonkeyPatch()
    # Product files may change while these checks run. Capture the
    # actual source once, solely for this fixture's fixed implementation lifetime.
    source = producer.source_identity()
    patch.setattr(producer, "source_identity", lambda: source)
    original_workers = producer.OrderedWorkers
    patch.setattr(producer, "OrderedWorkers", SerialWorkers)
    serial = root / "serial"
    producer.generate_main(serial, cfg, execution)
    patch.setattr(producer, "OrderedWorkers", original_workers)
    parallel = root / "parallel"
    producer.generate_main(
        parallel,
        cfg,
        execution.model_copy(update={"workers": 2}),
        pool_cache=serial / "pools",
    )
    saved_records = records(serial)
    diagnostics = {
        row["rally_id"]: row
        for path in sorted((serial / "files/dataset/admission").glob("*.json"))
        for row in load(path)["records"]
    }
    pools = {
        (p.split, 0): p
        for p in read_manifest(serial / "files/dataset/manifest.json").provenance.pools
    }
    yield {
        "cfg": cfg,
        "execution": execution,
        "serial": serial,
        "parallel": parallel,
        "records": {r.metadata.rally_id: r for r in saved_records},
        "diagnostics": diagnostics,
        "pools": pools,
    }
    patch.undo()


@pytest.fixture
def replay(simulated, monkeypatch):
    """Replay admitted arrays, retaining real writing, auditing and publication."""
    monkeypatch.setattr(producer, "OrderedWorkers", SerialWorkers)

    def prepare(task):
        _, split, index, *_ = task
        return simulated["pools"][(split, index)], {"replay": True}

    def produce(task):
        args, _kwargs = task
        _, split, index, *_ = args
        rally_id = f"{split}-rally-{index:06d}"
        return simulated["records"][rally_id], simulated["diagnostics"][rally_id]

    monkeypatch.setattr(producer, "_prepare", prepare)
    monkeypatch.setattr(producer, "_produce", produce)
    return simulated


def test_actual_serial_and_two_workers_have_identical_arrays_and_artifacts(simulated):
    serial, parallel = simulated["serial"], simulated["parallel"]
    assert content_hashes(serial) == content_hashes(parallel)
    expected = [p.rally_id for p in plan_rallies(simulated["cfg"])]
    left, right = records(serial), records(parallel)
    assert [r.metadata.rally_id for r in left] == expected
    assert [r.metadata.rally_id for r in right] == expected
    for a, b in zip(left, right, strict=True):
        assert a.metadata == b.metadata
        assert a.index() == b.index()
        assert len(a.views) == len(b.views) == 2
        for av, bv in zip(a.views, b.views, strict=True):
            assert av.metadata == bv.metadata
        for arrays_a, arrays_b in [
            (a.physical, b.physical),
            *[(av.arrays, bv.arrays) for av, bv in zip(a.views, b.views, strict=True)],
        ]:
            assert arrays_a.keys() == arrays_b.keys()
            for name in arrays_a:
                assert arrays_a[name].dtype == arrays_b[name].dtype
                np.testing.assert_array_equal(arrays_a[name], arrays_b[name])


def delayed_task(task):
    """Spawn-safe task that forces task one to finish before task zero."""
    index, directory = task
    directory = Path(directory)
    if index == 0:
        deadline = time.monotonic() + 15
        while not (directory / "release").exists():
            if time.monotonic() > deadline:
                raise RuntimeError("ordered delivery test was not released")
            time.sleep(0.01)
    (directory / f"finished-{index}").touch()
    return index


def test_completed_results_remain_in_the_inflight_limit_and_delivery_is_ordered(
    tmp_path,
):
    admitted, delivered = [], []
    checked = []

    def tasks():
        for index in range(5):
            admitted.append(index)
            assert len(admitted) - len(delivered) <= 2
            yield index, str(tmp_path)

    def release_first():
        deadline = time.monotonic() + 10
        while not (tmp_path / "finished-1").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        checked.append(
            (tmp_path / "finished-1").exists()
            and admitted == [0, 1]
            and not (tmp_path / "finished-0").exists()
            and not (tmp_path / "finished-2").exists()
        )
        (tmp_path / "release").touch()

    release = threading.Thread(target=release_first)
    release.start()
    try:
        with OrderedWorkers(2, 2, StopController(0.2)) as workers:
            for result in workers.map(delayed_task, tasks()):
                # Advancing delivery is part of the generator's admission bound.
                assert result == len(delivered)
                delivered.append(result)
    finally:
        release.join(timeout=12)
    assert not release.is_alive()
    assert checked == [True]
    assert delivered == list(range(5))


def unresponsive_task(directory):
    Path(directory, "started").touch()
    time.sleep(30)
    return "too late"


def timed_out_task(_):
    raise TimeoutError("task itself timed out")


def fail_while_peer_runs(directory):
    deadline = time.monotonic() + 10
    while not Path(directory, "started").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    raise RuntimeError("worker failed while peer runs")


def run_task(task):
    function, argument = task
    return function(argument)


@pytest.mark.parametrize("attempt", range(3))
def test_worker_failure_bounds_drain_without_external_stop(tmp_path, attempt):
    stop = StopController(0.15)
    workers = OrderedWorkers(2, 2, stop)
    processes = []

    # Both functions run in separate spawned processes through the same map.
    tasks = [(fail_while_peer_runs, str(tmp_path)), (unresponsive_task, str(tmp_path))]
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="worker failed while peer runs"), workers:
        try:
            list(workers.map(run_task, tasks))
        finally:
            processes.extend(workers.executor._processes.values())
    elapsed = time.monotonic() - started
    assert stop.requested
    assert (tmp_path / "started").exists()
    assert elapsed < 10
    assert processes and all(not p.is_alive() for p in processes)
    assert workers.executor._processes is None


@pytest.mark.parametrize("boundary", ["finalized", "main-publication"])
def test_stop_after_audit_or_publication_keeps_run_incomplete_and_resumable(
    tmp_path, replay, boundary
):
    output = tmp_path / boundary
    stop = StopController(0.1)

    def request(phase):
        if phase == boundary:
            stop.request()

    with pytest.raises(Stopped):
        producer.generate_main(
            output, replay["cfg"], replay["execution"], stop=stop, fault=request
        )
    assert not (output / "control/result.json").exists()
    assert (output / "control/finalized.json").exists()
    result = producer.generate_main(
        output, replay["cfg"], replay["execution"], resume=True
    )
    assert result["state"] == "complete"
    assert content_hashes(output) == content_hashes(replay["serial"])


def test_task_timeout_error_is_propagated_instead_of_retried_as_a_wait_timeout():
    stop = StopController(0.1)
    # A regression must fail promptly rather than wait forever on a done future.
    watchdog = threading.Timer(3, stop.request)
    watchdog.start()
    try:
        with (
            pytest.raises(TimeoutError, match="task itself timed out"),
            OrderedWorkers(1, 1, stop) as workers,
        ):
            list(workers.map(timed_out_task, [None]))
    finally:
        watchdog.cancel()
        watchdog.join(timeout=4)


def test_stop_drains_only_to_deadline_and_reaps_running_worker(tmp_path):
    stop = StopController(0.15)
    requested = []
    processes = []

    def request_when_running():
        deadline = time.monotonic() + 10
        while not (tmp_path / "started").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        processes.extend(workers.executor._processes.values())
        requested.append(time.monotonic())
        stop.request()

    request = threading.Thread(target=request_when_running)
    request.start()
    workers = OrderedWorkers(1, 1, stop)
    try:
        with pytest.raises(Stopped), workers:
            list(workers.map(unresponsive_task, [str(tmp_path)]))
        returned = time.monotonic()
    finally:
        request.join(timeout=12)
    assert not request.is_alive()
    assert (tmp_path / "started").exists()
    assert returned - requested[0] < 2
    assert processes
    assert all(not p.is_alive() and p.exitcode is not None for p in processes)
    assert workers.executor._processes is None


class Crash(RuntimeError):
    pass


def test_stop_after_committed_shard_keeps_incomplete_plan_and_can_resume(
    tmp_path, replay
):
    output = tmp_path / "stopped"
    stop = StopController(0.1)

    def request_stop(phase):
        if phase == "receipt":
            stop.request()

    with pytest.raises(Stopped):
        producer.generate_main(
            output, replay["cfg"], replay["execution"], stop=stop, fault=request_stop
        )
    incomplete = read_manifest(
        output / "control/incomplete.manifest.json", require_complete=False
    )
    assert incomplete.state == "incomplete"
    assert len(incomplete.shards) == 1
    assert [r.rally_id for r in incomplete.shards[0].records] == [
        "train-rally-000000",
        "train-rally-000001",
    ]
    assert incomplete.expected["val"].rallies == 1
    assert incomplete.expected["test"].rallies == 1
    assert not (output / "control/result.json").exists()
    producer.generate_main(output, replay["cfg"], replay["execution"], resume=True)
    assert content_hashes(output) == content_hashes(replay["serial"])


def test_stop_during_committed_shard_reconciliation_blocks_next_publication(
    tmp_path, replay, monkeypatch
):
    output = tmp_path / "stop-reconcile"
    ledgers = []

    def crash_after_second(phase):
        if phase == "ledger":
            ledgers.append(phase)
            if len(ledgers) == 2:
                raise Crash("two durable shards")

    with pytest.raises(Crash):
        producer.generate_main(
            output, replay["cfg"], replay["execution"], fault=crash_after_second
        )
    assert len(list((output / "control/commits").glob("*.json"))) == 2
    stop = StopController(0.1)
    calls = []
    publish = producer.Publication.publish

    def request_after_publish(self, spec, **kwargs):
        assert not stop.requested
        result = publish(self, spec, **kwargs)
        calls.append(spec.path)
        stop.request()
        return result

    monkeypatch.setattr(producer.Publication, "publish", request_after_publish)
    with pytest.raises(Stopped):
        producer.generate_main(
            output, replay["cfg"], replay["execution"], resume=True, stop=stop
        )
    assert calls == ["dataset/shards/000000.parquet"]
    assert not (output / "control/result.json").exists()
    monkeypatch.setattr(producer.Publication, "publish", publish)
    producer.generate_main(output, replay["cfg"], replay["execution"], resume=True)
    assert content_hashes(output) == content_hashes(replay["serial"])


@pytest.mark.parametrize("boundary", ["shard", "catalog", "ledger", "receipt"])
def test_crash_boundaries_resume_exact_membership_and_hashes(
    tmp_path, replay, boundary
):
    output = tmp_path / boundary

    def crash(phase):
        if phase == boundary:
            raise Crash(phase)

    with pytest.raises(Crash, match=boundary):
        producer.generate_main(output, replay["cfg"], replay["execution"], fault=crash)
    assert not (output / "control/result.json").exists()
    assert not (output / "files/dataset/manifest.json").exists()
    incomplete = read_manifest(
        output / "control/incomplete.manifest.json", require_complete=False
    )
    assert incomplete.state == "incomplete"
    assert incomplete.expected["train"].rallies == 2
    assert incomplete.expected["train"].views == 4
    commits_before = {
        p.name: p.read_bytes() for p in (output / "control/commits").glob("*.json")
    }
    result = producer.generate_main(
        output,
        replay["cfg"],
        replay["execution"].model_copy(update={"workers": 2}),
        resume=True,
    )
    assert result["state"] == "complete"
    assert content_hashes(output) == content_hashes(replay["serial"])
    assert not (output / "control/active.json").exists()
    for name, content in commits_before.items():
        assert (output / "control/commits" / name).read_bytes() == content


@pytest.mark.parametrize(
    "difference", ["recipe", "layout", "source", "runtime", "destination"]
)
def test_incompatible_resume_is_rejected_before_generation(
    tmp_path, replay, monkeypatch, difference
):
    output = tmp_path / "incompatible"
    stop = StopController(0.1)
    stop.request()
    with pytest.raises(Stopped):
        producer.generate_main(output, replay["cfg"], replay["execution"], stop=stop)
    plan = (output / "control/plan.json").read_bytes()
    cfg, execution = replay["cfg"], replay["execution"]
    if difference == "recipe":
        cfg = cfg.model_copy(update={"seed": cfg.seed + 1})
    elif difference == "layout":
        execution = execution.model_copy(update={"shard_rallies": 1})
    elif difference == "source":
        monkeypatch.setattr(producer, "source_identity", lambda: "f" * 64)
    elif difference == "runtime":
        monkeypatch.setattr(
            producer, "runtime_identity", lambda: {"python": "different"}
        )
    else:
        execution = execution.model_copy(
            update={"repo_id": "owner/other", "revision": "main"}
        )
    with pytest.raises(ValueError, match="resume requires matching"):
        producer.generate_main(output, cfg, execution, resume=True)
    assert (output / "control/plan.json").read_bytes() == plan
    assert not list((output / "control/commits").glob("*.json"))


def test_impossible_admission_keeps_failed_planned_id_and_never_skips(
    tmp_path, replay, monkeypatch
):
    from tennis3d.synthetic import producer_records

    output = tmp_path / "admission"
    execution = replay["execution"].model_copy(update={"max_failures": 2})
    original = producer._produce
    attempted = []
    rejected = []

    def assemble(_sim, _pool, _cfg, seed):
        record = next(r for r in replay["records"].values() if r.metadata.seed == seed)
        return Rally(record.physical, record.metadata.attempts, record.metadata.fps)

    original_observe = producer_records.observe

    def no_evidence(arrays, cfg, seed):
        observed = original_observe(arrays, cfg, seed)
        observed["ball_observed"][:] = False
        observed["ball_uv"][:] = np.nan
        observed["ball_confidence"][:] = 0
        return observed

    monkeypatch.setattr(producer_records, "assemble", assemble)
    monkeypatch.setattr(
        producer_records,
        "_get_family",
        lambda cfg, plan, *args: (
            SimpleNamespace(
                sim=None,
                pool=SimpleNamespace(diagnostics=Counter()),
                descriptor=replay["pools"][(plan.split, plan.family_index)],
            ),
            {},
        ),
    )

    def impossible(task):
        args, _ = task
        rally_id = f"{args[1]}-rally-{args[2]:06d}"
        attempted.append(rally_id)
        if rally_id == "train-rally-000001":
            with monkeypatch.context() as patch:
                patch.setattr(producer_records, "observe", no_evidence)
                try:
                    return produce_record(*args, **task[1])
                except AdmissionError as error:
                    rejected.append(error.diagnostics)
                    raise
        return original(task)

    monkeypatch.setattr(producer, "_produce", impossible)
    for resume in (False, True):
        with pytest.raises(AdmissionError, match="train-rally-000001"):
            producer.generate_main(output, replay["cfg"], execution, resume=resume)
        assert not list((output / "control/commits").glob("*.json"))
        incomplete = read_manifest(
            output / "control/incomplete.manifest.json", require_complete=False
        )
        assert incomplete.state == "incomplete"
        assert incomplete.expected["train"].rallies == 2
        assert not incomplete.shards
    with pytest.raises(ValueError, match="persisted failure budget exhausted"):
        producer.generate_main(output, replay["cfg"], execution, resume=True)
    assert set(attempted) == {"train-rally-000000", "train-rally-000001"}
    assert not (output / "files/dataset/manifest.json").exists()
    assert not (output / "control/result.json").exists()
    assert load(output / "control/failures.json")["train-rally-000001"] >= 2
    assert len(rejected) == 2
    for diagnostic in rejected:
        view = diagnostic["views"][0]
        assert view["accepted_seed"] is None
        assert view["attempt_count"] == execution.max_proposals
        assert all(p["rejected_windows"] for p in view["proposals"])


class FakeHub:
    """Immutable snapshots, with one lost response after a successful upload."""

    def __init__(self):
        self.current = "0" * 40
        self.snapshots = {self.current: {}}
        self.uploaded = []
        self.lose_response = True

    def head(self, repo_id, revision):
        return self.current

    def upload(self, repo_id, revision, parent, path, local):
        assert parent == self.current
        snapshot = dict(self.snapshots[parent])
        snapshot[path] = local.read_bytes()
        self.uploaded.append(path)
        self.current = f"{len(self.uploaded):040x}"
        self.snapshots[self.current] = snapshot
        if self.lose_response:
            self.lose_response = False
            raise ConnectionError("commit succeeded before receipt")
        return self.current

    def fingerprint(self, repo_id, commit, path):
        try:
            content = self.snapshots[commit][path]
        except KeyError:
            raise RemoteMissing(path) from None
        return hashlib.sha256(content).hexdigest(), len(content)


def test_stop_after_new_ledger_blocks_its_upload_and_preserves_durable_shard(
    tmp_path, replay
):
    output = tmp_path / "stop-new-ledger"
    stop = StopController(0.1)
    hub = FakeHub()
    hub.lose_response = False
    execution = replay["execution"].model_copy(
        update={"repo_id": "owner/dataset", "revision": "qualification"}
    )

    def request(phase):
        if phase == "ledger":
            stop.request()

    with pytest.raises(Stopped):
        producer.generate_main(
            output, replay["cfg"], execution, stop=stop, backend=hub, fault=request
        )
    assert hub.uploaded == []
    assert len(list((output / "control/commits").glob("*.json"))) == 1
    result = producer.generate_main(
        output, replay["cfg"], execution, resume=True, backend=hub
    )
    assert result["state"] == "complete"
    assert hub.uploaded.count("dataset/shards/000000.parquet") == 1


def test_stop_during_main_upload_retains_receipt_without_complete_publication(
    tmp_path, replay, monkeypatch
):
    output = tmp_path / "stop-main-upload"
    stop = StopController(0.1)
    hub = FakeHub()
    hub.lose_response = False
    upload = hub.upload

    def request_after_metadata(*args):
        commit = upload(*args)
        if args[3].endswith(".json"):
            stop.request()
        return commit

    monkeypatch.setattr(hub, "upload", request_after_metadata)
    execution = replay["execution"].model_copy(
        update={"repo_id": "owner/dataset", "revision": "qualification"}
    )
    with pytest.raises(Stopped):
        producer.generate_main(output, replay["cfg"], execution, stop=stop, backend=hub)
    uploaded = list(hub.uploaded)
    assert uploaded[-1].endswith(".json")
    assert "dataset/manifest.json" not in uploaded
    assert not (output / "control/main-publication.json").exists()
    assert not (output / "control/result.json").exists()
    monkeypatch.setattr(hub, "upload", upload)
    result = producer.generate_main(
        output, replay["cfg"], execution, resume=True, backend=hub
    )
    assert result["state"] == "complete"
    assert all(hub.uploaded.count(path) == 1 for path in uploaded)


@pytest.mark.parametrize("evict", [False, True])
def test_remote_commit_before_receipt_is_reconciled_without_duplicate_upload(
    tmp_path, replay, monkeypatch, evict
):
    hub = FakeHub()
    output = tmp_path / "publication"
    execution = replay["execution"].model_copy(
        update={"repo_id": "owner/dataset", "revision": "qualification", "evict": evict}
    )
    downloads = []

    class Session:
        @contextmanager
        def stream(self, method, url, **kwargs):
            assert method == "GET"
            assert "authorization" not in kwargs["headers"]
            commit, path = url.split("/resolve/", 1)[1].split("/", 1)
            receipt = read_json(output / ".publication/receipts.json")["ledger"][
                "entries"
            ][path]
            assert receipt["state"] == "verified"
            assert receipt["commit"] == commit
            downloads.append((commit, path))
            import httpx

            yield httpx.Response(
                200,
                content=hub.snapshots[commit][path],
                request=httpx.Request("GET", url),
            )

    # Exercise the actual resolver, replacing only its anonymous HTTP transport.
    monkeypatch.setattr("huggingface_hub.utils.get_session", lambda: Session())
    with pytest.raises(ConnectionError, match="commit succeeded"):
        producer.generate_main(output, replay["cfg"], execution, backend=hub)
    first = "dataset/shards/000000.parquet"
    assert hub.uploaded == [first]
    assert (output / "files" / first).exists()
    ledger = read_json(output / ".publication/receipts.json")["ledger"]
    assert ledger["entries"][first]["state"] == "pending"
    assert not (output / "files/dataset/manifest.json").exists()
    result = producer.generate_main(
        output, replay["cfg"], execution, backend=hub, resume=True
    )
    assert result["state"] == "complete"
    assert hub.uploaded.count(first) == 1
    assert hub.uploaded[-1] == "dataset/manifest.json"
    ledger = read_json(output / ".publication/receipts.json")["ledger"]
    assert ledger["complete"]["commit"] == result["commit"]
    assert all(e["state"] == "verified" for e in ledger["entries"].values())
    release = hub.snapshots[result["commit"]]
    for path, entry in ledger["entries"].items():
        assert hashlib.sha256(release[path]).hexdigest() == entry["sha256"]
    if evict:
        assert downloads
        assert not list((output / "files/dataset/shards").glob("*.parquet"))
    else:
        assert not downloads


def test_fresh_directory_bundle_restore_resumes_exact_output(tmp_path, simulated):
    output = tmp_path / "interrupted"

    def crash(phase):
        if phase == "ledger":
            raise Crash(phase)

    with pytest.raises(Crash):
        producer.generate_main(
            output, simulated["cfg"], simulated["execution"], fault=crash
        )
    owned = [
        p.relative_to(output).as_posix()
        for directory in (output / "control", output / "files", output / "pools")
        for p in directory.rglob("*")
        if p.is_file() and not p.name.endswith(".lock")
    ]
    bundle = create_bundle(output, owned, tmp_path / "recovery.zip", max_bytes=16 << 20)
    restored = tmp_path / "restored"
    report = restore_bundle(bundle.path, restored, max_bytes=16 << 20)
    assert report.bundle_sha256 == bundle.sha256
    result = producer.generate_main(
        restored, simulated["cfg"], simulated["execution"], resume=True
    )
    assert result["state"] == "complete"
    assert content_hashes(restored) == content_hashes(simulated["serial"])


def subset_execution(tmp_path, replay):
    from tennis3d.artifacts import write_json

    policy = tmp_path / "policy.json"
    write_json(
        policy,
        {
            "seed": 17,
            "strata": [
                {"name": "train", "split": "train", "equals": {}, "rallies": 1},
                {"name": "val", "split": "val", "equals": {}, "rallies": 1},
            ],
        },
    )
    return replay["execution"].model_copy(update={"subset_policy": str(policy)})


@pytest.mark.parametrize(
    "boundary", ["audit", "subset", "finalized", "main-publication"]
)
def test_joined_audit_crash_recovery_and_fresh_bundle(
    tmp_path, replay, monkeypatch, boundary
):
    from tennis3d.synthetic import producer_finalize
    from tennis3d.synthetic.producer_bundle import recovery_bundle

    execution = subset_execution(tmp_path, replay)
    output = tmp_path / "joined"
    calls = []
    original = producer_finalize.finalize_manifest

    def audit(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(producer_finalize, "finalize_manifest", audit)

    def crash(phase):
        if phase == boundary:
            raise Crash(phase)

    with pytest.raises(Crash, match=boundary):
        producer.generate_main(output, replay["cfg"], execution, fault=crash)
    if boundary in {"audit", "subset"}:
        # A drained failure cleans exactly its journalled extraction root.
        assert not (output / "subsets").exists()
        assert not (output / "subset-transaction").exists()
    bundle = recovery_bundle(output, tmp_path / "joined.zip", 32 << 20)
    restored = tmp_path / "restored-joined"
    restore_bundle(bundle.path, restored, max_bytes=32 << 20)
    result = producer.generate_main(restored, replay["cfg"], execution, resume=True)
    assert result["state"] == "complete"
    assert result["subset"]["needs_binding"]
    assert len(calls) == (2 if boundary in {"audit", "subset"} else 1)
    compact = load(restored / "subsets/extracted.json")
    assert sum(len(s["records"]) for s in compact["shards"]) == 2
    assert all(len(r["views"]) == 2 for s in compact["shards"] for r in s["records"])
    assert not (restored / "control/subset-active.json").exists()
    main = read_manifest(restored / "files/dataset/manifest.json")
    assert SimulationConfig.model_validate(main.provenance.config) == replay["cfg"]


def test_changed_policy_cannot_resume_and_late_audit_failure_has_no_complete_release(
    tmp_path, replay, monkeypatch
):
    from tennis3d.artifacts import write_json
    from tennis3d.synthetic.producer_summary import NumericSummary

    execution = subset_execution(tmp_path, replay)
    output = tmp_path / "late-audit"

    def failed(self, record, diagnostic):
        if record.metadata.split == "test":
            raise ValueError("late numeric audit failure")

    monkeypatch.setattr(NumericSummary, "add", failed)
    with pytest.raises(ValueError, match="late numeric audit failure"):
        producer.generate_main(output, replay["cfg"], execution)
    assert not (output / "files/dataset/manifest.json").exists()
    assert not (output / "subsets").exists()
    assert not (output / "control/finalized.json").exists()
    raw = read_json(execution.subset_policy)
    raw["seed"] += 1
    write_json(execution.subset_policy, raw, overwrite=True)
    with pytest.raises(ValueError, match="resume requires matching"):
        producer.generate_main(output, replay["cfg"], execution, resume=True)


def test_joined_remote_bundle_preserves_original_main_binding_without_reaudit(
    tmp_path, replay, monkeypatch
):
    from tennis3d.synthetic import producer_finalize
    from tennis3d.synthetic.producer_bundle import recovery_bundle

    hub = FakeHub()
    hub.lose_response = False
    output = tmp_path / "remote-joined"
    execution = subset_execution(tmp_path, replay).model_copy(
        update={"repo_id": "owner/dataset", "revision": "qualification"}
    )
    result = producer.generate_main(output, replay["cfg"], execution, backend=hub)
    main_commit = result["commit"]
    assert main_commit != result["subset"]["commit"]
    descriptor = read_json(output / "subsets/dataset/search/subset.json")
    assert descriptor["descriptor"]["main_commit"] == main_commit
    uploaded = list(hub.uploaded)
    bundle = recovery_bundle(output, tmp_path / "remote-joined.zip", 32 << 20)
    restored = tmp_path / "remote-restored"
    restore_bundle(bundle.path, restored, max_bytes=32 << 20)

    def no_audit(*args, **kwargs):
        raise AssertionError("durable successful global audit must be reused")

    monkeypatch.setattr(producer_finalize, "finalize_manifest", no_audit)
    resumed = producer.generate_main(
        restored, replay["cfg"], execution, backend=hub, resume=True
    )
    assert resumed == result
    assert hub.uploaded == uploaded
    assert read_json(restored / "subsets/dataset/search/subset.json") == descriptor


def test_pool_preparation_failure_budget_is_durable(tmp_path, replay, monkeypatch):
    def failed(task):
        raise producer.PoolPreparationError(
            f"pool:{task[1]}:{task[2]}", "bounded pool preparation failure"
        )

    monkeypatch.setattr(producer, "_prepare", failed)
    execution = replay["execution"].model_copy(update={"max_failures": 2})
    output = tmp_path / "pool-failure"
    for resume in (False, True):
        with pytest.raises(ValueError, match="bounded pool preparation failure"):
            producer.generate_main(output, replay["cfg"], execution, resume=resume)
    assert load(output / "control/failures.json")["pool:train:0"] == 2
    with pytest.raises(ValueError, match="pool preparation failure budget exhausted"):
        producer.generate_main(output, replay["cfg"], execution, resume=True)
    assert not list((output / "control/commits").glob("*.json"))
    assert not (output / "files/dataset/manifest.json").exists()


@pytest.mark.parametrize("cache_limit", [1, 3])
def test_cold_preparation_obeys_configured_worker_cache_bound(
    tmp_path, monkeypatch, cache_limit
):
    seen = []

    class ModelFreeWorkers(SerialWorkers):
        def map(self, function, tasks):
            for task in tasks:
                if function is producer._produce:
                    raise Stopped("stop before numeric generation")
                result = function(task)
                seen.append(len(producer_records._CACHE))
                yield result

    def pool(path, provenance, sim, cfg, seed):
        path.write_bytes(str(seed).encode("ascii"))
        return SimpleNamespace(
            states={kind: np.zeros((cfg.pool_size, 9)) for kind in KINDS},
            attempts={kind: 1 for kind in KINDS},
            diagnostics=Counter(),
        )

    monkeypatch.setattr(producer, "OrderedWorkers", ModelFreeWorkers)
    monkeypatch.setattr(producer_records, "Simulator", lambda _: object())
    monkeypatch.setattr(producer_records, "load_or_build", pool)
    cfg = SimulationConfig(train_rallies=9, train_families=9, pool_size=1)
    execution = ProducerConfig(workers=1, inflight=1, worker_cache_families=cache_limit)
    output = tmp_path / "cold-cache"
    close_worker_cache()
    with pytest.raises(Stopped, match="before numeric generation"):
        producer.generate_main(output, cfg, execution)
    assert len(seen) == sum(cfg.family_count(s) for s in ("train", "val", "test"))
    assert max(seen) == cache_limit
    assert not producer_records._CACHE
    assert not list((output / "control/commits").glob("*.json"))
