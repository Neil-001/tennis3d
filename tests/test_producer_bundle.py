"""Packaging must prove that archived checkpoints can reuse committed bytes."""

from types import SimpleNamespace
from typing import cast

import pytest

from tennis3d.artifacts import json_sha256
from tennis3d.datasets.publication import (
    ArtifactLike,
    BoundedStaging,
    HuggingFaceBackend,
    Publisher,
    RemoteMissing,
    VerificationError,
)
from tennis3d.datasets.schema import Artifact
from tennis3d.synthetic.producer_bundle import recovery_bundle
from tennis3d.synthetic.producer_publication import Publication
from tennis3d.synthetic.producer_recovery import restore_bundle
from tennis3d.synthetic.producer_state import artifact, load, save, verify

CAP = 1 << 20


@pytest.fixture
def committed(tmp_path, monkeypatch):
    output = tmp_path / "run"
    plan = {
        "layout": {"prefix": "dataset"},
        "publication": {"repo_id": "owner/dataset", "revision": "release"},
    }
    save(output / "control/plan.json", plan)
    specs = {}
    for key, relative in (
        ("shard", "shards/000000.parquet"),
        ("catalog", "catalog/000000.json"),
        ("admission", "admission/000000.json"),
    ):
        relative = "dataset/" + relative
        path = output / "files" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((key + " committed bytes").encode())
        specs[key] = artifact(output / "files", relative)
    save(
        output / "control/commits/000000.json",
        {key: spec.model_dump() for key, spec in specs.items()},
    )
    save(output / "control/pools/train-000000.json", {"pool": "hash-bound"})
    (output / "pools").mkdir()
    (output / "pools/train-000000.npz").write_bytes(b"separate cache")

    def no_transport(*args, **kwargs):
        pytest.fail("recovery packaging attempted network or authentication")

    monkeypatch.setattr(HuggingFaceBackend, "_run", no_transport)
    return output, plan, specs


def rejected(output, tmp_path, match):
    destination = tmp_path / "recovery.zip"
    with pytest.raises((ValueError, VerificationError), match=match):
        recovery_bundle(output, destination, CAP)
    assert not destination.exists()
    assert not list(tmp_path.glob(".recovery.zip.*.tmp"))


@pytest.mark.parametrize("key", ["shard", "catalog", "admission"])
def test_missing_committed_artifact_rejects_packaging(committed, tmp_path, key):
    output, _, specs = committed
    (output / "files" / specs[key].path).unlink()
    rejected(
        output,
        tmp_path,
        "missing exact verified publication receipt"
        if key == "shard"
        else "owned artifact differs",
    )


@pytest.mark.parametrize("key", ["shard", "catalog", "admission"])
@pytest.mark.parametrize("change", ["same-size", "different-size"])
def test_changed_committed_bytes_reject_packaging(committed, tmp_path, key, change):
    output, _, specs = committed
    path = output / "files" / specs[key].path
    content = path.read_bytes()
    path.write_bytes(
        bytes([content[0] ^ 1]) + content[1:]
        if change == "same-size"
        else content + b"changed"
    )
    rejected(output, tmp_path, "owned artifact differs")


class ReceiptBackend:
    """The publisher writes genuine verified receipts without external transport."""

    def head(self, repo_id, revision):
        return "0" * 40

    def upload(self, repo_id, revision, parent, path, local):
        self.spec = artifact(local.parent, local.name)
        return "1" * 40

    def fingerprint(self, repo_id, commit, path):
        if commit == "0" * 40:
            raise RemoteMissing(path)
        return self.spec.sha256, self.spec.size_bytes


def evict(output, plan, spec):
    staging = BoundedStaging(output, CAP, metadata_reserve_bytes=8192)
    publisher = Publisher(staging, ReceiptBackend(), **plan["publication"])
    commit = publisher.publish(spec, evict=True)
    assert not (output / "files" / spec.path).exists()
    return publisher, commit


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", "dataset/other.parquet"),
        ("sha256", "a" * 64),
        ("size_bytes", 1),
        ("state", "uploaded"),
        ("state", "pending"),
        ("commit", "release"),
        ("commit", "A" * 40),
        ("commit", None),
    ],
)
def test_invalid_eviction_receipt_rejects_packaging(committed, tmp_path, field, value):
    output, plan, specs = committed
    publisher, _ = evict(output, plan, specs["shard"])
    ledger = publisher._load()
    ledger["entries"][specs["shard"].path][field] = value
    publisher._save(ledger)
    rejected(output, tmp_path, "publication receipt|immutable commit")


@pytest.mark.parametrize(
    "field,value", [("repo_id", "other/dataset"), ("revision", "other-release")]
)
def test_receipt_destination_must_match_saved_plan(committed, tmp_path, field, value):
    output, plan, specs = committed
    publisher, _ = evict(output, plan, specs["shard"])
    ledger = publisher._load()
    ledger[field] = value
    publisher._save(ledger)
    rejected(output, tmp_path, "Invalid receipt ledger")


def test_receipt_checksum_rejects_packaging(committed, tmp_path):
    output, plan, specs = committed
    publisher, _ = evict(output, plan, specs["shard"])
    path = publisher.ledger_path
    path.write_bytes(path.read_bytes().replace(b'"verified"', b'"uploaded"'))
    rejected(output, tmp_path, "Invalid receipt ledger")


def test_malformed_receipt_entries_reject_packaging(committed, tmp_path):
    output, plan, specs = committed
    publisher, _ = evict(output, plan, specs["shard"])
    ledger = publisher._load()
    ledger["entries"] = []
    publisher._save(ledger)
    rejected(output, tmp_path, "invalid publication receipt entries")


def test_receipt_parse_bound_uses_actual_ledger_size(committed, tmp_path, monkeypatch):
    output, plan, specs = committed
    publisher, _ = evict(output, plan, specs["shard"])
    ledger_bytes = publisher.ledger_path.stat().st_size
    reserves = []
    original = Publisher._load

    def record_reserve(self):
        reserves.append(self.staging.metadata_reserve_bytes)
        return original(self)

    monkeypatch.setattr(Publisher, "_load", record_reserve)
    recovery_bundle(output, tmp_path / "recovery.zip", 4 << 30)
    assert reserves == [max(1024, 2 * ledger_bytes)]


def test_missing_local_only_shard_has_no_remote_recovery(committed, tmp_path):
    output, plan, specs = committed
    plan["publication"] = {"repo_id": None, "revision": None}
    save(output / "control/plan.json", plan, overwrite=True)
    (output / "files" / specs["shard"].path).unlink()
    rejected(output, tmp_path, "missing exact verified publication receipt")


@pytest.mark.parametrize("key", ["catalog", "admission"])
def test_receipt_cannot_replace_required_local_metadata(committed, tmp_path, key):
    output, plan, specs = committed
    evict(output, plan, specs[key])
    rejected(output, tmp_path, "owned artifact differs")


def finalized(output, plan):
    specs = []
    for relative in ("catalog.json", "manifest.json", "summary.json", "audit.json"):
        relative = "dataset/" + relative
        (output / "files" / relative).write_bytes((relative + " finalized").encode())
        specs.append(artifact(output / "files", relative).model_dump())
    (output / "subsets").mkdir()
    (output / "subsets/extracted.json").write_bytes(b"extraction checkpoint")
    saved = {
        "plan_sha256": json_sha256(plan),
        "manifest": specs[1],
        "artifacts": specs,
        "subset_artifacts": [
            artifact(output / "subsets", "extracted.json").model_dump()
        ],
    }
    save(output / "control/finalized.json", saved)
    return saved


@pytest.mark.parametrize(
    "name", ["catalog.json", "manifest.json", "summary.json", "audit.json"]
)
@pytest.mark.parametrize("change", ["missing", "changed"])
def test_finalized_main_metadata_must_match_checkpoint(
    committed, tmp_path, name, change
):
    output, plan, _ = committed
    finalized(output, plan)
    path = output / "files/dataset" / name
    if change == "missing":
        path.unlink()
    else:
        path.write_bytes(b"changed finalized bytes")
    rejected(output, tmp_path, "owned artifact differs")


def test_finalized_plan_binding_is_required(committed, tmp_path):
    output, plan, _ = committed
    saved = finalized(output, plan)
    saved["plan_sha256"] = "a" * 64
    save(output / "control/finalized.json", saved, overwrite=True)
    rejected(output, tmp_path, "finalized checkpoint differs from the producer plan")


def test_changed_finalized_checkpoint_is_rejected(committed, tmp_path):
    output, plan, _ = committed
    finalized(output, plan)
    path = output / "control/finalized.json"
    path.write_bytes(path.read_bytes().replace(b'"dataset/', b'"changed/', 1))
    rejected(output, tmp_path, "producer checkpoint checksum mismatch")


@pytest.mark.parametrize(
    "checkpoint", ["main-publication", "subset-publication", "result"]
)
def test_completed_output_requires_finalized_checkpoint(
    committed, tmp_path, checkpoint
):
    output, _, _ = committed
    save(output / f"control/{checkpoint}.json", {"state": "complete"})
    rejected(output, tmp_path, "missing its finalized checkpoint")


def test_finalized_subset_bytes_are_still_required(committed, tmp_path):
    output, plan, _ = committed
    finalized(output, plan)
    (output / "subsets/extracted.json").unlink()
    rejected(output, tmp_path, "owned artifact differs")


def main_publication(output, plan):
    saved = finalized(output, plan)
    spec = Artifact.model_validate(saved["manifest"])
    staging = BoundedStaging(output, CAP, metadata_reserve_bytes=8192)
    publisher = Publisher(staging, ReceiptBackend(), **plan["publication"])
    main = {
        "manifest": saved["manifest"],
        "commit": publisher.publish(cast(ArtifactLike, spec)),
    }
    save(output / "control/main-publication.json", main)
    return saved, main, publisher


@pytest.mark.parametrize(
    "field,value",
    [("manifest", {}), ("commit", "release"), ("commit", "2" * 40)],
)
def test_main_publication_checkpoint_must_match_finalization_and_receipt(
    committed, tmp_path, field, value
):
    output, plan, _ = committed
    _, main, _ = main_publication(output, plan)
    main[field] = value
    save(output / "control/main-publication.json", main, overwrite=True)
    rejected(output, tmp_path, "differs from release|immutable commit")


def compact_publication(output, plan):
    saved, main, _ = main_publication(output, plan)
    manifest = output / "subsets/dataset/search/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"compact manifest")
    spec = artifact(output / "subsets", "dataset/search/manifest.json")
    saved["subset_artifacts"].append(spec.model_dump())
    save(output / "control/finalized.json", saved, overwrite=True)
    local_artifacts = []
    for name in ("subset.json", "copy-audit.json"):
        relative = "subsets/dataset/search/" + name
        (output / relative).write_bytes((name + " publication metadata").encode())
        local_artifacts.append(artifact(output, relative).model_dump())
    compact = {
        **plan["publication"],
        "main_commit": main["commit"],
        "manifest": spec.model_dump(),
        "descriptor_sha256": "a" * 64,
        "local_artifacts": local_artifacts,
    }
    save(output / "control/subset-publication.json", compact)
    staging = BoundedStaging(
        output / "subset-publication", CAP, metadata_reserve_bytes=8192
    )
    staging.stage(cast(ArtifactLike, spec), [manifest.read_bytes()])
    publisher = Publisher(staging, ReceiptBackend(), **plan["publication"])
    result = {
        "main_commit": main["commit"],
        "manifest": compact["manifest"],
        "descriptor_sha256": compact["descriptor_sha256"],
        "commit": publisher.publish(cast(ArtifactLike, spec)),
    }
    save(output / "control/subset-publication-result.json", result)
    return compact, result


@pytest.mark.parametrize("damage", ["missing", "changed-commit", "changed-destination"])
def test_completed_compact_requires_exact_verified_receipt(committed, tmp_path, damage):
    output, plan, _ = committed
    compact_publication(output, plan)
    ledger_path = output / "subset-publication/.publication/receipts.json"
    if damage == "missing":
        ledger_path.unlink()
    else:
        from tennis3d.artifacts import read_json, write_json

        envelope = read_json(ledger_path)
        ledger = envelope["ledger"]
        if damage == "changed-commit":
            ledger["entries"]["dataset/search/manifest.json"]["commit"] = "2" * 40
        else:
            ledger["revision"] = "different"
        envelope["sha256"] = json_sha256(ledger)
        write_json(ledger_path, envelope, overwrite=True)
    rejected(output, tmp_path, "receipt|destination")


@pytest.mark.parametrize(
    "field,value",
    [
        ("repo_id", "other/dataset"),
        ("revision", "other-release"),
        ("main_commit", "3" * 40),
        ("manifest", {}),
    ],
)
def test_compact_checkpoint_must_bind_to_saved_main_publication(
    committed, tmp_path, field, value
):
    output, plan, _ = committed
    compact, _ = compact_publication(output, plan)
    compact[field] = value
    save(output / "control/subset-publication.json", compact, overwrite=True)
    rejected(output, tmp_path, "compact publication differs from main publication")


@pytest.mark.parametrize(
    "field,value",
    [
        ("main_commit", "3" * 40),
        ("manifest", {}),
        ("descriptor_sha256", "b" * 64),
        ("commit", "release"),
    ],
)
def test_compact_result_must_bind_to_checkpoint_and_immutable_commit(
    committed, tmp_path, field, value
):
    output, plan, _ = committed
    _, result = compact_publication(output, plan)
    result[field] = value
    save(output / "control/subset-publication-result.json", result, overwrite=True)
    rejected(output, tmp_path, "differs from release|immutable commit")


@pytest.mark.parametrize("checkpoint", ["main-publication", "subset-publication"])
def test_compact_publication_requires_upstream_checkpoint(
    committed, tmp_path, checkpoint
):
    output, plan, _ = committed
    compact_publication(output, plan)
    (output / f"control/{checkpoint}.json").unlink()
    rejected(output, tmp_path, "differs from main publication|differs from release")


@pytest.mark.parametrize("change", ["missing", "changed"])
def test_compact_publication_local_artifacts_are_required(committed, tmp_path, change):
    output, plan, _ = committed
    compact, _ = compact_publication(output, plan)
    path = output / compact["local_artifacts"][0]["path"]
    if change == "missing":
        path.unlink()
    else:
        path.write_bytes(b"changed descriptor")
    rejected(output, tmp_path, "owned artifact differs")


def test_compact_publication_bundle_preserves_original_main_commit(committed, tmp_path):
    output, plan, _ = committed
    compact, result = compact_publication(output, plan)
    bundle = recovery_bundle(output, tmp_path / "recovery.zip", CAP)
    restored = tmp_path / "restored"
    restore_bundle(bundle.path, restored, CAP)
    main = load(restored / "control/main-publication.json")
    assert main["commit"] == compact["main_commit"] == result["main_commit"]
    assert load(restored / "control/subset-publication.json") == compact
    assert load(restored / "control/subset-publication-result.json") == result
    for raw in compact["local_artifacts"]:
        verify(restored, Artifact.model_validate(raw))
    recovery_bundle(restored, tmp_path / "second.zip", CAP)


def test_drained_partial_output_without_finalization_can_be_bundled(
    committed, tmp_path
):
    output, _, specs = committed
    bundle = recovery_bundle(output, tmp_path / "partial.zip", CAP)
    restored = tmp_path / "restored"
    restore_bundle(bundle.path, restored, CAP)
    assert not (restored / "control/finalized.json").exists()
    for spec in specs.values():
        verify(restored / "files", spec)


@pytest.mark.parametrize("intent", ["active", "subset-active"])
def test_active_intent_rejects_before_archive_creation(
    committed, tmp_path, monkeypatch, intent
):
    output, _, _ = committed
    save(output / f"control/{intent}.json", {"active": True})

    def no_archive(*args, **kwargs):
        pytest.fail("active intent reached archive creation")

    monkeypatch.setattr("tennis3d.synthetic.producer_bundle.create_bundle", no_archive)
    rejected(output, tmp_path, "must drain before recovery packaging")


@pytest.mark.parametrize("evicted", [False, True])
def test_verified_bundle_restores_checkpoint_bytes_and_excludes_numeric_pool_cache(
    committed, tmp_path, evicted
):
    output, plan, specs = committed
    saved = finalized(output, plan)
    if evicted:
        _, commit = evict(output, plan, specs["shard"])
    bundle = recovery_bundle(output, tmp_path / "recovery.zip", CAP)
    restored = tmp_path / "restored"
    restore_bundle(bundle.path, restored, CAP)
    assert load(restored / "control/plan.json") == plan
    assert load(restored / "control/finalized.json") == saved
    assert load(restored / "control/commits/000000.json") == {
        key: spec.model_dump() for key, spec in specs.items()
    }
    assert load(restored / "control/pools/train-000000.json") == {"pool": "hash-bound"}
    assert not (restored / "pools").exists()
    for key in ("catalog", "admission"):
        verify(restored / "files", specs[key])
    for raw in saved["artifacts"]:
        verify(restored / "files", Artifact.model_validate(raw))
    if evicted:
        publication = Publication(
            restored,
            SimpleNamespace(
                staging_bytes=CAP,
                metadata_reserve_bytes=8192,
                **plan["publication"],
            ),
        )
        assert publication.receipt(specs["shard"])["commit"] == commit
        assert not (restored / "files" / specs["shard"].path).exists()
    else:
        verify(restored / "files", specs["shard"])
    recovery_bundle(restored, tmp_path / "second.zip", CAP)
