"""Separate compact publication, immutable binding, and lost-response recovery."""

import hashlib
import shutil
import weakref

import pytest

pytest.importorskip("pyarrow")

from test_producer_subsets import generated, sample  # noqa: F401

from tennis3d.artifacts import read_json, write_json
from tennis3d.datasets.audit import finalize_manifest
from tennis3d.datasets.publication import RemoteMissing
from tennis3d.datasets.schema import count_records
from tennis3d.datasets.subset import read_descriptor
from tennis3d.synthetic.producer_parallel import StopController, Stopped
from tennis3d.synthetic.producer_state import artifact, load, save, usage
from tennis3d.synthetic.producer_subset_publication import publish_subset
from tennis3d.synthetic.producer_subsets import ExtractionTransaction


class FakeHub:
    def __init__(self, initial):
        self.current = "a" * 40
        self.main_commit = self.current
        self.snapshots = {self.current: initial}
        self.uploaded = []
        self.lose_path = None

    def head(self, repo_id, revision):
        return self.current

    def upload(self, repo_id, revision, parent, path, local):
        assert parent == self.current
        snapshot = dict(self.snapshots[parent])
        snapshot[path] = local.read_bytes()
        self.uploaded.append(path)
        self.current = f"{len(self.uploaded):040x}"
        self.snapshots[self.current] = snapshot
        if self.lose_path == path:
            self.lose_path = None
            raise ConnectionError("commit succeeded before receipt")
        return self.current

    def fingerprint(self, repo_id, commit, path):
        try:
            content = self.snapshots[commit][path]
        except KeyError:
            raise RemoteMissing(path) from None
        return hashlib.sha256(content).hexdigest(), len(content)


@pytest.fixture
def ready(tmp_path, generated):  # noqa: F811
    files, manifest, catalog, policy, execution, _, _ = generated
    output = tmp_path / "producer"
    output.mkdir()
    transaction_root = output / "subset-transaction"
    reports = []
    main_path = f"{execution.prefix}/manifest.json"
    with ExtractionTransaction(
        transaction_root, manifest, catalog, policy, execution
    ) as transaction:

        def completed(report):
            reports.append(report)
            transaction.finish(report)

        finalize_manifest(
            files / main_path,
            manifest,
            root=files,
            on_record=transaction,
            on_audit_complete=completed,
        )
    transaction_root.rename(output / "subsets")
    catalog_path = f"{execution.prefix}/catalog/coverage.json"
    write_json(files / catalog_path, catalog.model_dump())
    main_spec, catalog_spec = artifact(files, main_path), artifact(files, catalog_path)
    execution = execution.model_copy(
        update={"repo_id": "owner/dataset", "revision": "qualification"}
    )
    initial = {
        spec.path: (files / spec.path).read_bytes()
        for spec in (main_spec, catalog_spec)
    }
    hub = FakeHub(initial)
    return output, execution, hub, main_spec, catalog_spec, files, reports[0]


def call(ready, **kwargs):
    output, execution, hub, main_spec, catalog_spec, files, report = ready
    return publish_subset(
        output,
        execution,
        hub.main_commit,
        main_spec,
        catalog_spec,
        files,
        report,
        backend=hub,
        **kwargs,
    )


def test_local_only_extraction_remains_pending_without_descriptor_or_upload(ready):
    output, execution, hub, main_spec, catalog_spec, files, report = ready
    result = publish_subset(
        output, execution, None, main_spec, catalog_spec, files, report, backend=hub
    )
    assert result["state"] == "extracted"
    assert result["needs_binding"] is True
    assert "commit" not in result
    assert hub.uploaded == []
    assert not (output / "subset-publication").exists()
    assert not list((output / "subsets").rglob("subset.json"))


def test_stop_during_compact_upload_preserves_receipt_and_resumes_without_duplicate(
    ready, monkeypatch
):
    output, _, hub, *_ = ready
    stop = StopController(0.1)
    upload = hub.upload

    def request_after_upload(*args):
        commit = upload(*args)
        stop.request()
        return commit

    monkeypatch.setattr(hub, "upload", request_after_upload)
    with pytest.raises(Stopped):
        call(ready, check_stop=stop.check)
    uploaded = list(hub.uploaded)
    assert len(uploaded) == 1
    assert not (output / "control/subset-publication-result.json").exists()
    monkeypatch.setattr(hub, "upload", upload)
    assert call(ready)["state"] == "complete"
    assert hub.uploaded.count(uploaded[0]) == 1


def test_stop_during_compact_stage_does_not_start_upload(ready, monkeypatch):
    from tennis3d.datasets.publication import BoundedStaging

    output, _, hub, *_ = ready
    stop = StopController(0.1)
    stage = BoundedStaging.stage

    def request_after_stage(self, *args, **kwargs):
        path = stage(self, *args, **kwargs)
        stop.request()
        return path

    monkeypatch.setattr(BoundedStaging, "stage", request_after_stage)
    with pytest.raises(Stopped):
        call(ready, check_stop=stop.check)
    assert hub.uploaded == []
    assert not (output / "control/subset-publication-result.json").exists()
    monkeypatch.setattr(BoundedStaging, "stage", stage)
    assert call(ready)["state"] == "complete"


def test_staged_compact_resume_does_not_reserve_the_same_copy_twice(ready, monkeypatch):
    from tennis3d.datasets.publication import BoundedStaging

    output, execution, hub, main_spec, catalog_spec, files, report = ready
    stop = StopController(0.1)
    stage = BoundedStaging.stage

    def request_after_stage(self, *args, **kwargs):
        path = stage(self, *args, **kwargs)
        stop.request()
        return path

    monkeypatch.setattr(BoundedStaging, "stage", request_after_stage)
    with pytest.raises(Stopped):
        call(ready, check_stop=stop.check)
    monkeypatch.setattr(BoundedStaging, "stage", stage)
    assert hub.uploaded == []
    cap = usage(output) + execution.metadata_reserve_bytes
    constrained = execution.model_copy(update={"staging_bytes": cap})
    next_stop = StopController(0.1)
    upload = hub.upload

    def request_after_upload(*args):
        commit = upload(*args)
        next_stop.request()
        return commit

    monkeypatch.setattr(hub, "upload", request_after_upload)
    with pytest.raises(Stopped):
        publish_subset(
            output,
            constrained,
            hub.main_commit,
            main_spec,
            catalog_spec,
            files,
            report,
            backend=hub,
            check_stop=next_stop.check,
        )
    assert len(hub.uploaded) == 1
    first = hub.uploaded[0]
    monkeypatch.setattr(hub, "upload", upload)
    assert call(ready)["state"] == "complete"
    assert hub.uploaded.count(first) == 1


def test_compact_copy_reserves_against_retained_whole_output(ready):
    output, execution, hub, main_spec, catalog_spec, files, report = ready
    extracted = load(output / "subsets/extracted.json")
    first_size = extracted["shards"][0]["size_bytes"]
    cap = usage(output) + execution.metadata_reserve_bytes + first_size - 1
    execution = execution.model_copy(update={"staging_bytes": cap})
    with pytest.raises(ValueError, match="staging cannot reserve compact"):
        publish_subset(
            output,
            execution,
            hub.main_commit,
            main_spec,
            catalog_spec,
            files,
            report,
            backend=hub,
        )
    assert hub.uploaded == []
    assert not list((output / "subset-publication/files").rglob("*.parquet"))


def test_compact_publication_binds_main_and_commits_manifest_last_without_rescan(
    ready, monkeypatch
):
    from tennis3d.datasets import audit, io

    def forbidden(*args, **kwargs):
        raise AssertionError("publication must not scan numeric shards")

    monkeypatch.setattr(audit, "audit_dataset", forbidden)
    monkeypatch.setattr(io.ShardReader, "__init__", forbidden)
    result = call(ready)
    output, execution, hub, main_spec, catalog_spec, _, report = ready
    assert result["state"] == "complete"
    assert result["main_commit"] == hub.main_commit
    prefix = f"{execution.prefix}/search"
    assert hub.uploaded[-1] == f"{prefix}/manifest.json"
    assert all(path.startswith(prefix + "/") for path in hub.uploaded)
    assert main_spec.path not in hub.uploaded
    assert catalog_spec.path not in hub.uploaded
    descriptor = read_descriptor(
        output / "subsets" / result["descriptor"]["path"],
        expected_sha256=result["descriptor_sha256"],
    )
    assert descriptor.main_commit == hub.main_commit
    assert descriptor.main_manifest == main_spec
    assert descriptor.main_catalog == catalog_spec
    copied_audit = read_json(output / "subsets" / result["audit"]["path"])
    assert copied_audit["main_audit_report"] == report
    assert copied_audit["counts"] == {
        s: c.model_dump() for s, c in count_records(descriptor.subset.shards).items()
    }
    for spec in [descriptor.subset.manifest, *descriptor.subset.shards]:
        assert (
            hub.snapshots[result["commit"]][spec.path]
            == (output / "subsets" / spec.path).read_bytes()
        )
    ledger = read_json(output / "subset-publication/.publication/receipts.json")[
        "ledger"
    ]
    assert ledger["complete"]["commit"] == result["commit"]
    assert not list((output / "subset-publication/files").rglob("*.parquet"))
    assert list((output / "subsets").rglob("*.parquet"))
    assert load(output / "control/subset-publication-result.json") == result
    uploaded = list(hub.uploaded)
    assert call(ready) == result
    assert hub.uploaded == uploaded


@pytest.mark.parametrize("lost", ["shard", "descriptor", "manifest"])
def test_lost_response_fresh_directory_resume_never_duplicates_upload(
    ready, tmp_path, lost
):
    output, execution, hub, main_spec, catalog_spec, files, report = ready
    extracted = load(output / "subsets/extracted.json")
    prefix = f"{execution.prefix}/search"
    hub.lose_path = {
        "shard": extracted["shards"][0]["path"],
        "descriptor": f"{prefix}/subset.json",
        "manifest": f"{prefix}/manifest.json",
    }[lost]
    lost_path = hub.lose_path
    with pytest.raises(ConnectionError, match="commit succeeded"):
        call(ready)
    assert hub.uploaded.count(lost_path) == 1
    fresh = tmp_path / "restored"
    shutil.copytree(output, fresh)
    result = publish_subset(
        fresh,
        execution,
        hub.main_commit,
        main_spec,
        catalog_spec,
        files,
        report,
        backend=hub,
    )
    assert result["state"] == "complete"
    assert hub.uploaded.count(lost_path) == 1
    assert hub.uploaded[-1] == f"{prefix}/manifest.json"
    assert load(fresh / "control/subset-publication-result.json") == result


@pytest.mark.parametrize(
    "damage",
    [
        "selection",
        "audit",
        "main-catalog",
        "compact-manifest",
        "numeric",
        "remote-main",
    ],
)
def test_invalid_source_or_copy_cannot_publish(ready, damage):
    output, execution, hub, main_spec, catalog_spec, files, report = ready
    extracted = load(output / "subsets/extracted.json")
    if damage == "selection":
        selection = load(output / "subsets/selection.json")
        selection["policy"]["seed"] += 1
        save(output / "subsets/selection.json", selection, overwrite=True)
    elif damage == "audit":
        report = {}
    elif damage == "main-catalog":
        (files / catalog_spec.path).write_text("{}")
    elif damage == "compact-manifest":
        (output / "subsets" / extracted["manifest"]["path"]).write_text("{}")
    elif damage == "numeric":
        (output / "subsets" / extracted["shards"][0]["path"]).write_text("bad")
    elif damage == "remote-main":
        hub.snapshots[hub.main_commit][main_spec.path] = b"wrong"
    with pytest.raises(ValueError):
        publish_subset(
            output,
            execution,
            hub.main_commit,
            main_spec,
            catalog_spec,
            files,
            report,
            backend=hub,
        )
    assert hub.uploaded == []


def test_changed_main_commit_cannot_rebind_existing_publication(ready):
    call(ready)
    output, execution, hub, main_spec, catalog_spec, files, report = ready
    uploaded = list(hub.uploaded)
    with pytest.raises(ValueError):
        publish_subset(
            output,
            execution,
            hub.current,
            main_spec,
            catalog_spec,
            files,
            report,
            backend=hub,
        )
    assert hub.uploaded == uploaded


def test_source_manifest_is_released_before_bind_release_allocates(ready, monkeypatch):
    from tennis3d.synthetic import producer_subset_publication as publication

    _, _, _, main_spec, _, files, _ = ready
    original_read = publication.read_manifest
    original_bind = publication.bind_release
    references = []

    def read(path):
        manifest = original_read(path)
        if path == files / main_spec.path:
            references.append(weakref.ref(manifest))
        return manifest

    def bind(*args, **kwargs):
        assert len(references) == 1
        assert references[0]() is None
        return original_bind(*args, **kwargs)

    monkeypatch.setattr(publication, "read_manifest", read)
    monkeypatch.setattr(publication, "bind_release", bind)
    assert call(ready)["state"] == "complete"
