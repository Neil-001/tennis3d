"""Explicit phase handoff with real audits and model-free admitted arrays."""

import copy
import shutil
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("pyarrow")

from test_dataset_parquet import sample  # noqa: F401
from test_producer import FakeHub, SerialWorkers, content_hashes, subset_execution

from tennis3d.artifacts import file_sha256, read_json, write_json
from tennis3d.datasets.io import read_manifest
from tennis3d.datasets.publication import VerificationError
from tennis3d.datasets.schema import Pool
from tennis3d.datasets.subset import read_descriptor
from tennis3d.synthetic import producer, producer_finalize, producer_publication
from tennis3d.synthetic.config import SimulationConfig
from tennis3d.synthetic.producer_bundle import recovery_bundle
from tennis3d.synthetic.producer_config import ProducerConfig
from tennis3d.synthetic.producer_parallel import StopController, Stopped
from tennis3d.synthetic.producer_publication import Publication
from tennis3d.synthetic.producer_records import (
    admission_failures,
    plan_family,
    plan_rally,
    proposal_seed,
    proposal_summary,
)
from tennis3d.synthetic.producer_recovery import restore_bundle
from tennis3d.synthetic.producer_state import load, save


@pytest.fixture
def admitted(sample, monkeypatch):  # noqa: F811
    """Mock physical generation only; retain numeric validation and publication."""
    record, _ = sample
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
    source = producer.source_identity()
    monkeypatch.setattr(producer, "source_identity", lambda: source)
    monkeypatch.setattr(producer, "OrderedWorkers", SerialWorkers)

    def prepare(task):
        config, split, index, cache, source_hash, runtimes, _ = task
        family = plan_family(config, split, index, source_hash, runtimes)
        # Finalization requires preserved hash-bound bytes, without loading NPZ
        # or rebuilding physics. These tiny fixtures test that exact contract.
        path = cache / family.cache_name
        path.write_bytes(f"original pool bytes for {family.family_id}".encode())
        pool = Pool(
            family_id=family.family_id,
            split=split,
            sha256=file_sha256(path),
            config=family.pool_config,
            attempts={"toss": 1},
        )
        return pool, {"pool_built": False}

    def produce(task):
        args, _ = task
        config, split, index, pool, *_ = args
        planned = plan_rally(config, split, index)
        saved = copy.deepcopy(record)
        saved.metadata = saved.metadata.model_copy(
            update={
                "rally_id": planned.rally_id,
                "seed": planned.seed,
                "split": split,
                "family_id": pool.family_id,
                "pool_sha256": pool.sha256,
            }
        )
        saved.views = [
            copy.deepcopy(saved.views[0]) for _ in range(config.views_per_rally)
        ]
        diagnostic = {
            "rally_id": planned.rally_id,
            "rally_seed": planned.seed,
            "views": [],
        }
        for view_id, view in enumerate(saved.views):
            seed = proposal_seed(planned.seed, view_id, 0)
            view.metadata = view.metadata.model_copy(
                update={"view_id": view_id, "view_seed": seed}
            )
            checked = admission_failures(view.arrays)
            assert checked["accepted"]
            diagnostic["views"].append(
                {
                    "view_id": view_id,
                    "accepted_seed": seed,
                    "attempt_count": 1,
                    "proposals": [
                        {
                            "attempt": 0,
                            "seed": seed,
                            **checked,
                            "summary": proposal_summary(view.arrays, config, seed),
                        }
                    ],
                }
            )
        return saved, diagnostic

    monkeypatch.setattr(producer, "_prepare", prepare)
    monkeypatch.setattr(producer, "_produce", produce)
    return {"cfg": cfg, "execution": execution}


class CountingHub(FakeHub):
    def __init__(self):
        super().__init__()
        self.lose_response = False
        self.reads = []
        self.fetches = []

    def fingerprint(self, repo_id, commit, path):
        result = super().fingerprint(repo_id, commit, path)
        self.reads.append((commit, path))
        return result


def remote_execution(admitted, *, evict=True):
    return admitted["execution"].model_copy(
        update={"repo_id": "owner/dataset", "revision": "qualification", "evict": evict}
    )


def remote_transport(monkeypatch, hub):
    def chunks(source, spec):
        assert len(source.revision) == 40
        payload = hub.snapshots[source.revision][spec.path]
        hub.fetches.append((source.revision, spec.path))
        yield payload

    monkeypatch.setattr(producer_publication, "hub_chunks", chunks)


def forbid_generation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("finalization entered the worker/physics/generation path")

    for name in (
        "OrderedWorkers",
        "_prepare",
        "_produce",
        "plan_families",
        "prepare_family",
        "produce_record",
    ):
        monkeypatch.setattr(producer, name, forbidden)
    monkeypatch.setattr("tennis3d.synthetic.producer_records.Simulator", forbidden)
    monkeypatch.setattr("tennis3d.synthetic.producer_records.plan_family", forbidden)


def assert_unpublished(output, hub):
    assert not any(path.endswith("/manifest.json") for path in hub.uploaded)
    assert not (output / "files/dataset/manifest.json").exists()
    assert not (output / "control/finalized.json").exists()
    assert not (output / "control/main-publication.json").exists()
    assert not (output / "control/result.json").exists()
    assert not (output / "subsets").exists()


def test_generation_archive_fresh_restore_finalization_matches_default_with_compact(
    tmp_path, admitted, monkeypatch
):
    execution = subset_execution(tmp_path, admitted).model_copy(
        update={"repo_id": "owner/dataset", "revision": "qualification", "evict": True}
    )
    cfg = admitted["cfg"]
    default, phased = tmp_path / "default", tmp_path / "phased"
    direct_hub, phased_hub = CountingHub(), CountingHub()
    remote_transport(monkeypatch, direct_hub)
    direct = producer.generate_main(default, cfg, execution, backend=direct_hub)
    remote_transport(monkeypatch, phased_hub)
    audits = []
    audit = producer_finalize.finalize_manifest

    def audited(*args, **kwargs):
        audits.append(True)
        return audit(*args, **kwargs)

    monkeypatch.setattr(producer_finalize, "finalize_manifest", audited)
    generation = producer.generate_main(
        phased, cfg, execution, phase="generate", backend=phased_hub
    )
    assert generation["state"] == "generation-complete"
    assert generation["counts"] == direct["counts"]
    assert audits == []
    assert_unpublished(phased, phased_hub)
    assert not (phased / "files/dataset/catalog.json").exists()
    assert (
        load(phased / "control/generation-complete.json")["state"]
        == "generation-complete"
    )
    incomplete = read_manifest(
        phased / "control/incomplete.manifest.json", require_complete=False
    )
    assert incomplete.state == "incomplete"
    shard_paths = [shard.path for shard in incomplete.shards]
    assert Counter(path for _, path in phased_hub.reads) == Counter(shard_paths)
    assert phased_hub.fetches == []
    assert all(not (phased / "files" / path).exists() for path in shard_paths)

    bundle = recovery_bundle(phased, tmp_path / "generation.zip", 32 << 20)
    assert "control/generation-complete.json" in [item.path for item in bundle.files]
    assert not any(item.path.startswith("pools/") for item in bundle.files)
    restored = tmp_path / "restored"
    restore_bundle(bundle.path, restored, 32 << 20)
    shutil.copytree(phased / "pools", restored / "pools")
    forbid_generation(monkeypatch)
    result = producer.generate_main(
        restored, cfg, execution, phase="finalize", resume=True, backend=phased_hub
    )
    assert result["state"] == "complete"
    assert result["commit"] == direct["commit"]
    assert result["subset"] == direct["subset"]
    assert audits == [True]
    assert (
        phased_hub.snapshots[result["subset"]["commit"]]
        == direct_hub.snapshots[direct["subset"]["commit"]]
    )
    assert content_hashes(restored) == content_hashes(default)
    compact = read_descriptor(
        restored / "subsets" / result["subset"]["descriptor"]["path"],
        expected_sha256=result["subset"]["descriptor_sha256"],
    )
    assert compact.main_commit == result["commit"]
    assert phased_hub.uploaded.index(
        "dataset/manifest.json"
    ) < phased_hub.uploaded.index("dataset/search/manifest.json")
    for path in shard_paths:
        # First upload, audit eviction, metadata-publication recheck, final parent
        # fingerprint, and the numeric audit fetch: the existing five reads.
        assert sum(p == path for _, p in phased_hub.reads) == 4
        assert sum(p == path for _, p in phased_hub.fetches) == 1
        assert phased_hub.uploaded.count(path) == 1
        assert (
            phased_hub.snapshots[result["commit"]][path]
            == direct_hub.snapshots[direct["commit"]][path]
        )


@pytest.mark.parametrize("phase", ["generate", "complete"])
def test_generation_resume_reuses_evicted_receipts_without_download(
    tmp_path, admitted, monkeypatch, phase
):
    output, hub = tmp_path / "resumed", CountingHub()
    execution = remote_execution(admitted)
    remote_transport(monkeypatch, hub)
    stop = StopController(0.1)

    def stop_after_first_receipt(boundary):
        if boundary == "receipt":
            stop.request()

    with pytest.raises(Stopped):
        producer.generate_main(
            output,
            admitted["cfg"],
            execution,
            phase="generate",
            backend=hub,
            stop=stop,
            fault=stop_after_first_receipt,
        )
    assert not (output / "control/generation-complete.json").exists()
    assert_unpublished(output, hub)
    first = "dataset/shards/000000.parquet"
    first_commit = hub.reads[0][0]
    assert hub.reads == [(first_commit, first)]
    assert hub.fetches == []
    result = producer.generate_main(
        output, admitted["cfg"], execution, phase=phase, backend=hub, resume=True
    )
    assert result["state"] == (
        "generation-complete" if phase == "generate" else "complete"
    )
    if phase == "generate":
        assert hub.reads.count((first_commit, first)) == 1
        assert hub.fetches == []
        assert_unpublished(output, hub)
    else:
        assert sum(path == first for _, path in hub.reads) == 4
        assert hub.fetches == [
            (first_commit, first),
            (hub.reads[1][0], "dataset/shards/000001.parquet"),
        ]
    assert hub.uploaded.count(first) == 1
    assert [
        row["rally_id"]
        for path in sorted((output / "control/commits").glob("*.json"))
        for row in load(path)["shard"]["records"]
    ] == [
        "train-rally-000000",
        "train-rally-000001",
        "val-rally-000000",
        "test-rally-000000",
    ]


@pytest.mark.parametrize(
    "damage",
    [
        "missing-boundary",
        "corrupt-boundary",
        "missing-commit",
        "changed-commit",
        "missing-descriptor",
        "changed-descriptor",
        "changed-provenance",
        "missing-pool",
        "changed-pool",
    ],
)
def test_finalization_refuses_damaged_generation_before_audit(
    tmp_path, admitted, monkeypatch, damage
):
    output, hub = tmp_path / "damaged", CountingHub()
    execution = remote_execution(admitted)
    producer.generate_main(
        output, admitted["cfg"], execution, phase="generate", backend=hub
    )
    control = output / "control"
    if damage == "missing-boundary":
        (control / "generation-complete.json").unlink()
    elif damage == "corrupt-boundary":
        save(control / "generation-complete.json", {}, overwrite=True)
    elif damage == "missing-commit":
        (control / "commits/000001.json").unlink()
    elif damage == "changed-commit":
        commit = load(control / "commits/000000.json")
        commit["shard"]["records"][0]["seed"] += 1
        save(control / "commits/000000.json", commit, overwrite=True)
    elif damage == "missing-descriptor":
        (control / "pools/train-000000.json").unlink()
    elif damage == "changed-descriptor":
        descriptor = load(control / "pools/train-000000.json")
        descriptor["pool"]["attempts"]["toss"] += 1
        save(control / "pools/train-000000.json", descriptor, overwrite=True)
    elif damage == "changed-provenance":
        provenance = load(control / "provenance.json")
        provenance["code_revision"] = "replacement"
        save(control / "provenance.json", provenance, overwrite=True)
    else:
        path = next((output / "pools").glob("*.npz"))
        if damage == "missing-pool":
            path.unlink()
        else:
            path.write_bytes(b"mutated pool")
    forbid_generation(monkeypatch)
    monkeypatch.setattr(
        producer,
        "finalization",
        lambda *a, **k: pytest.fail("damaged generation reached the audit"),
    )
    with pytest.raises((ValueError, OSError)):
        producer.generate_main(
            output,
            admitted["cfg"],
            execution,
            phase="finalize",
            resume=True,
            backend=hub,
        )
    assert_unpublished(output, hub)


@pytest.mark.parametrize("phase", ["generate", "finalize"])
@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "pending",
        "uploaded",
        "unverified",
        "hash",
        "size",
        "path",
        "commit",
        "checksum",
    ],
)
def test_evicted_committed_shards_require_exact_durable_verified_receipts(
    tmp_path, admitted, monkeypatch, phase, damage
):
    output, hub = tmp_path / "bad-receipt", CountingHub()
    execution = remote_execution(admitted)
    producer.generate_main(
        output, admitted["cfg"], execution, phase="generate", backend=hub
    )
    publication = Publication(output, execution, hub)
    assert publication.publisher is not None
    ledger = publication.publisher._load()
    path = "dataset/shards/000000.parquet"
    entry = ledger["entries"][path]
    if damage == "missing":
        del ledger["entries"][path]
    elif damage in {"pending", "uploaded", "unverified"}:
        entry["state"] = damage
    elif damage == "hash":
        entry["sha256"] = "f" * 64
    elif damage == "size":
        entry["size_bytes"] += 1
    elif damage == "path":
        entry["path"] = "dataset/shards/replacement.parquet"
    elif damage == "commit":
        entry["commit"] = "qualification"
    publication.publisher._save(ledger)
    if damage == "checksum":
        envelope = read_json(output / ".publication/receipts.json")
        envelope["sha256"] = "f" * 64
        write_json(output / ".publication/receipts.json", envelope, overwrite=True)
    reads = list(hub.reads)
    monkeypatch.setattr(
        producer,
        "OrderedWorkers",
        lambda *a, **k: pytest.fail("bad receipt started workers"),
    )
    monkeypatch.setattr(
        producer,
        "finalization",
        lambda *a, **k: pytest.fail("bad receipt reached finalization"),
    )
    with pytest.raises((ValueError, VerificationError)):
        producer.generate_main(
            output, admitted["cfg"], execution, phase=phase, resume=True, backend=hub
        )
    assert hub.reads == reads
    assert hub.fetches == []
    assert_unpublished(output, hub)


def test_retained_mutated_shard_is_not_evicted_or_replaced_on_resume(
    tmp_path, admitted
):
    output, hub = tmp_path / "retained", CountingHub()
    execution = remote_execution(admitted, evict=False)
    producer.generate_main(
        output, admitted["cfg"], execution, phase="generate", backend=hub
    )
    path = output / "files/dataset/shards/000000.parquet"
    path.write_bytes(b"replacement bytes")
    reads, uploads = list(hub.reads), list(hub.uploaded)
    with pytest.raises(ValueError, match="owned artifact differs"):
        producer.generate_main(
            output,
            admitted["cfg"],
            execution.model_copy(update={"evict": True}),
            phase="generate",
            resume=True,
            backend=hub,
        )
    assert path.read_bytes() == b"replacement bytes"
    assert hub.reads == reads
    assert hub.uploaded == uploads


def test_late_numeric_audit_failure_does_not_publish_main_or_compact(
    tmp_path, admitted, monkeypatch
):
    output, hub = tmp_path / "failed-audit", CountingHub()
    execution = subset_execution(tmp_path, admitted).model_copy(
        update={"repo_id": "owner/dataset", "revision": "qualification", "evict": True}
    )
    producer.generate_main(
        output, admitted["cfg"], execution, phase="generate", backend=hub
    )
    remote_transport(monkeypatch, hub)
    forbid_generation(monkeypatch)
    from tennis3d.synthetic.producer_summary import NumericSummary

    add = NumericSummary.add
    visited = []

    def fail_late(self, record, diagnostic):
        visited.append(record.metadata.rally_id)
        add(self, record, diagnostic)
        if record.metadata.split == "test":
            raise ValueError("late numeric audit failed")

    monkeypatch.setattr(NumericSummary, "add", fail_late)
    with pytest.raises(ValueError, match="late numeric audit failed"):
        producer.generate_main(
            output,
            admitted["cfg"],
            execution,
            phase="finalize",
            resume=True,
            backend=hub,
        )
    assert visited == [
        "train-rally-000000",
        "train-rally-000001",
        "val-rally-000000",
        "test-rally-000000",
    ]
    assert_unpublished(output, hub)
    assert (output / "control/generation-complete.json").exists()
    assert not (output / "subset-transaction").exists()


@pytest.mark.parametrize("missing", ["resume", "state"])
def test_finalization_requires_existing_generation_and_resume(
    tmp_path, admitted, missing
):
    output = tmp_path / "empty"
    with pytest.raises(ValueError, match="finalization requires"):
        producer.generate_main(
            output,
            admitted["cfg"],
            admitted["execution"],
            phase="finalize",
            resume=(missing == "state"),
        )
    assert not (output / "control/plan.json").exists()
    assert not (output / "control/generation-complete.json").exists()


@pytest.mark.parametrize(
    "identity", ["source", "runtime", "recipe", "layout", "destination", "subset"]
)
def test_finalization_cannot_change_frozen_identity(
    tmp_path, admitted, monkeypatch, identity
):
    output, hub = tmp_path / "identity", CountingHub()
    cfg = admitted["cfg"]
    execution = subset_execution(tmp_path, admitted).model_copy(
        update={"repo_id": "owner/dataset", "revision": "qualification"}
    )
    producer.generate_main(output, cfg, execution, phase="generate", backend=hub)
    if identity == "source":
        monkeypatch.setattr(producer, "source_identity", lambda: "f" * 64)
    elif identity == "runtime":
        original = producer.runtime_identity()
        monkeypatch.setattr(
            producer, "runtime_identity", lambda: {**original, "python": "0.0.0"}
        )
    elif identity == "recipe":
        cfg = cfg.model_copy(update={"seed": cfg.seed + 1})
    elif identity == "layout":
        execution = execution.model_copy(
            update={"family_cohort": execution.family_cohort + 1}
        )
    elif identity == "destination":
        execution = execution.model_copy(update={"revision": "other"})
    else:
        policy = read_json(execution.subset_policy)
        policy["seed"] += 1
        write_json(Path(execution.subset_policy), policy, overwrite=True)
    forbid_generation(monkeypatch)
    before = list(hub.uploaded)
    with pytest.raises(ValueError, match="resume requires matching"):
        producer.generate_main(
            output, cfg, execution, phase="finalize", resume=True, backend=hub
        )
    assert hub.uploaded == before
    assert_unpublished(output, hub)


def test_finalization_accepts_compatible_execution_limits(
    tmp_path, admitted, monkeypatch
):
    output = tmp_path / "compatible"
    execution = admitted["execution"]
    producer.generate_main(output, admitted["cfg"], execution, phase="generate")
    plan = (output / "control/plan.json").read_bytes()
    boundary = (output / "control/generation-complete.json").read_bytes()
    forbid_generation(monkeypatch)
    changed = execution.model_copy(
        update={
            "workers": 2,
            "inflight": 4,
            "staging_bytes": execution.staging_bytes * 2,
        }
    )
    result = producer.generate_main(
        output, admitted["cfg"], changed, phase="finalize", resume=True
    )
    assert result["state"] == "complete"
    assert (output / "control/plan.json").read_bytes() == plan
    assert (output / "control/generation-complete.json").read_bytes() == boundary


def test_generation_boundary_requires_successful_first_upload_readback(
    tmp_path, admitted, monkeypatch
):
    output, hub = tmp_path / "readback-failed", CountingHub()
    execution = remote_execution(admitted)
    fingerprint = hub.fingerprint

    def wrong_uploaded_bytes(repo, commit, path):
        _digest, size = fingerprint(repo, commit, path)
        return "f" * 64, size

    monkeypatch.setattr(hub, "fingerprint", wrong_uploaded_bytes)
    with pytest.raises(VerificationError, match="Remote content mismatch"):
        producer.generate_main(
            output, admitted["cfg"], execution, phase="generate", backend=hub
        )
    assert not (output / "control/generation-complete.json").exists()
    assert_unpublished(output, hub)
    shard = output / "files/dataset/shards/000000.parquet"
    assert shard.exists()
    publication = Publication(output, execution, hub)
    assert publication.publisher is not None
    receipt = publication.publisher._load()["entries"]["dataset/shards/000000.parquet"]
    assert receipt["state"] == "uploaded"


def test_final_parent_tree_verification_remains_mandatory(
    tmp_path, admitted, monkeypatch
):
    output, hub = tmp_path / "changed-parent", CountingHub()
    execution = remote_execution(admitted)
    producer.generate_main(
        output, admitted["cfg"], execution, phase="generate", backend=hub
    )
    remote_transport(monkeypatch, hub)
    forbid_generation(monkeypatch)
    upload = hub.upload

    def change_later_parent(*args):
        commit = upload(*args)
        if args[3].endswith("producer.json"):
            hub.snapshots[commit]["dataset/shards/000000.parquet"] = (
                b"changed parent tree"
            )
        return commit

    monkeypatch.setattr(hub, "upload", change_later_parent)
    with pytest.raises(VerificationError, match="Remote content mismatch"):
        producer.generate_main(
            output,
            admitted["cfg"],
            execution,
            phase="finalize",
            resume=True,
            backend=hub,
        )
    assert (output / "control/finalized.json").exists()
    assert not (output / "control/main-publication.json").exists()
    assert not (output / "control/result.json").exists()
    assert "dataset/manifest.json" not in hub.uploaded
