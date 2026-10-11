"""Representative catalog selection, one-pass extraction and offline tamper checks."""

import copy
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

pytest.importorskip("pyarrow")

from test_dataset_parquet import assert_record_equal, sample  # noqa: F401

from tennis3d.artifacts import file_sha256, json_sha256, read_json, write_json
from tennis3d.datasets.audit import audit_rally
from tennis3d.datasets.io import ShardReader, ShardWriter, read_manifest, write_manifest
from tennis3d.datasets.schema import Artifact, Manifest, Pool, Split, count_records
from tennis3d.datasets.source import SourceSpec
from tennis3d.datasets.subset import (
    CatalogRally,
    CoverageCatalog,
    SelectionPolicy,
    StratumQuota,
    SubsetExtractor,
    bind_release,
    materialize_subset,
    select_subset,
    supported_counts,
    validate_source_subset,
    verify_materialized,
)


@pytest.fixture
def source(tmp_path, sample):  # noqa: F811
    record, provenance = sample
    splits: tuple[Split, ...] = ("train", "val", "test")
    provenance = provenance.model_copy(
        update={
            "pools": [
                Pool(
                    family_id=f"family-{split}",
                    split=split,
                    sha256=str(i + 1) * 64,
                    config={"seed": 3},
                    attempts={"toss": 1},
                )
                for i, split in enumerate(splits)
            ]
        }
    )
    root = tmp_path / "main"
    root.mkdir()
    shards, rows, originals = [], [], {}
    # Both strata and splits span all source shards. Every physical rally has two views.
    for shard_id in range(3):
        name = f"source-{shard_id}.parquet"
        with ShardWriter(root / name, provenance, row_group_rallies=2) as writer:
            pending = []
            for split_index, split in enumerate(splits):
                for category in ("short", "long"):
                    other = copy.deepcopy(record)
                    rid = f"{split}-{category}-{shard_id}"
                    other.metadata = other.metadata.model_copy(
                        update={
                            "rally_id": rid,
                            "split": split,
                            "family_id": f"family-{split}",
                            "pool_sha256": str(split_index + 1) * 64,
                        }
                    )
                    second = copy.deepcopy(other.views[0])
                    second.metadata = second.metadata.model_copy(update={"view_id": 8})
                    other.views.append(second)
                    writer.append(other)
                    originals[rid] = other
                    pending.append((other, category))
        shard = writer.close()
        shards.append(shard)
        for index, (other, category) in enumerate(pending):
            rows.append(
                CatalogRally(
                    shard_path=name,
                    shard_sha256=shard.sha256,
                    row_index=index,
                    record=other.index(),
                    counts=supported_counts(other),
                    physical_duration_s=float(other.physical["t_s"][-1]),
                    shot_kind_bits=3,
                    strata={"duration": category},
                )
            )
    manifest = Manifest(
        state="complete",
        provenance=provenance,
        shards=shards,
        expected=count_records(shards),
    )
    write_manifest(root / "manifest.json", manifest)
    catalog = CoverageCatalog(
        source_manifest_sha256=json_sha256(manifest.model_dump()),
        definitions={
            "duration": "Fixture declared short and long categories; no production bins."
        },
        definitions_source_sha256="9" * 64,
        records=rows,
    )
    write_json(root / "catalog.json", catalog.model_dump())
    policy = SelectionPolicy(
        seed=17,
        strata=[
            StratumQuota(
                name=category, split=split, equals={"duration": category}, rallies=2
            )
            for split in ("train", "val")
            for category in ("short", "long")
        ],
        reserved_rally_ids=["val-short-0", "val-long-0"],
    )
    return root, manifest, catalog, policy, originals


def audit_pass(root, manifest, visitor):
    """Test-side producer implements the documented optional visitor contract."""
    calls = 0
    for shard in manifest.shards:
        with ShardReader(
            root / shard.path, shard, provenance=manifest.provenance
        ) as reader:
            for index in range(len(shard.records)):
                record = reader.read(index)
                audit_rally(record)
                visitor(
                    record,
                    SimpleNamespace(
                        shard_path=shard.path,
                        shard_sha256=shard.sha256,
                        row_index=index,
                        split=record.metadata.split,
                        rally_id=record.metadata.rally_id,
                        view_ids=tuple(v.metadata.view_id for v in record.views),
                    ),
                )
                calls += 1
    return {
        "schema_version": "2.0",
        "shards": len(manifest.shards),
        "counts": {
            s: c.model_dump() for s, c in count_records(manifest.shards).items()
        },
    }, calls


def extract(tmp_path, source):
    root, manifest, catalog, policy, _ = source
    selection = select_subset(manifest, catalog, policy)
    destination = tmp_path / "compact"
    with SubsetExtractor(
        destination, manifest, catalog, selection, row_group_rallies=1, target_bytes=1
    ) as visitor:
        report, calls = audit_pass(root, manifest, visitor)
        result = visitor.finish(source_audit_report=report)
    assert calls == 18
    assert len(result.shards) == 8
    return destination, result


def descriptor(tmp_path, source):
    destination, result = extract(tmp_path, source)
    root = source[0]

    def artifact(name):
        path = root / name
        return Artifact(
            path=name, sha256=file_sha256(path), size_bytes=path.stat().st_size
        )

    bound = bind_release(
        result,
        main_repo_id="owner/main",
        main_commit="a" * 40,
        main_manifest=artifact("manifest.json"),
        main_catalog=artifact("catalog.json"),
        main_manifest_path=root / "manifest.json",
        main_catalog_path=root / "catalog.json",
    )
    return destination, bound


def test_declared_selection_order_independent_and_reserved(source):
    _, manifest, catalog, policy, _ = source
    first = select_subset(manifest, catalog, policy)
    reversed_catalog = catalog.model_copy(
        update={"records": list(reversed(catalog.records))}
    )
    second = select_subset(manifest, reversed_catalog, policy)
    assert first.records == second.records
    assert first.catalog_sha256 != second.catalog_sha256
    assert len(first.records) == 8
    assert all(r.record.split != "test" for r in first.records)
    assert not set(policy.reserved_rally_ids) & {
        r.record.rally_id for r in first.records
    }
    assert len({r.shard_path for r in first.records}) >= 2
    assert all([v.view_id for v in r.record.views] == [0, 8] for r in first.records)
    changed = policy.model_copy(update={"seed": 31})
    assert select_subset(manifest, catalog, changed).records != first.records


@pytest.mark.parametrize(
    "damage", ["sparse", "overlap", "catalog-row", "missing", "definition"]
)
def test_selection_rejects_invalid_inputs(source, damage):
    _, manifest, catalog, policy, _ = source
    if damage == "sparse":
        policy.strata[0] = policy.strata[0].model_copy(update={"rallies": 4})
    elif damage == "overlap":
        policy.strata.append(
            StratumQuota(name="overlap", split="train", equals={}, rallies=1)
        )
    elif damage == "catalog-row":
        catalog.records[0] = catalog.records[0].model_copy(update={"row_index": 1})
    elif damage == "missing":
        catalog.records.pop()
    else:
        policy.strata[0] = policy.strata[0].model_copy(
            update={"equals": {"candidate_score": "good"}}
        )
    with pytest.raises(ValueError):
        select_subset(manifest, catalog, policy)


def test_extraction_exact_arrays_masks_counts_and_source_bindings(tmp_path, source):
    destination, result = extract(tmp_path, source)
    original_root, main, catalog, _, originals = source
    compact = read_manifest(destination / result.manifest.path)
    assert compact.provenance == main.provenance
    assert compact.expected["test"].rallies == 0
    assert compact.expected["train"].rallies == compact.expected["val"].rallies == 4
    for shard in compact.shards:
        with ShardReader(
            destination / shard.path, shard, provenance=compact.provenance
        ) as reader:
            for i in range(len(shard.records)):
                copied = reader.read(i)
                assert_record_equal(originals[copied.metadata.rally_id], copied)
    assert result.selection.catalog_sha256 == json_sha256(catalog.model_dump())
    assert result.selection.source_manifest_sha256 == json_sha256(main.model_dump())
    assert result.manifest.sha256 != file_sha256(original_root / "manifest.json")
    source_shards = {s.sha256 for s in main.shards}
    assert not source_shards & {s.sha256 for s in result.shards}


@pytest.mark.parametrize(
    "failure", ["late-audit", "bad-report", "missing-selected", "counts"]
)
def test_failed_audit_never_finalizes_subset(tmp_path, source, failure):
    root, manifest, catalog, policy, _ = source
    if failure == "counts":
        for row in catalog.records:
            row.counts[0] = row.counts[0].model_copy(
                update={
                    "supported": 0,
                    "supported_occluded": 0,
                    "gap_bins": {k: 0 for k in row.counts[0].gap_bins},
                    "distance_bins": {k: 0 for k in row.counts[0].distance_bins},
                }
            )
    selection = select_subset(manifest, catalog, policy)
    output = tmp_path / "failed"
    with (
        pytest.raises(ValueError),
        SubsetExtractor(output, manifest, catalog, selection) as visitor,
    ):
        if failure == "missing-selected":
            visitor.finish(
                source_audit_report={
                    "schema_version": "2.0",
                    "shards": 3,
                    "counts": {s: c.model_dump() for s, c in manifest.expected.items()},
                }
            )
        report, _ = audit_pass(root, manifest, visitor)
        if failure == "late-audit":
            raise ValueError("later unselected source record failed audit")
        visitor.finish(source_audit_report={} if failure == "bad-report" else report)
    assert not output.exists()


def test_materialize_compact_only_then_verify_without_remote(tmp_path, source):
    compact_root, bound = descriptor(tmp_path, source)
    digest = json_sha256(bound.model_dump())
    fetched = []

    @contextmanager
    def resolve(artifact):
        fetched.append(artifact.path)
        yield compact_root / artifact.path

    local = tmp_path / "offline"
    path = materialize_subset(
        bound, local, resolver=resolve, expected_descriptor_sha256=digest
    )
    assert path == local / "manifest.json"
    for split in ("train", "val"):
        spec = read_json(local / f"{split}.source.json")
        assert SourceSpec.model_validate(spec).root == str(local.resolve())
        assert spec["root"] == str(local.resolve())
        assert spec["split"] == split
        assert spec["subset_sha256"] == digest
        assert spec["revision"] == bound.main_commit
    assert fetched == [
        bound.subset.manifest.path,
        *[s.path for s in bound.subset.shards],
    ]
    assert verify_materialized(local, expected_descriptor_sha256=digest) == bound
    assert bound.main_commit == "a" * 40
    assert (
        validate_source_subset(
            local, digest, bound.subset.manifest, bound.main_repo_id, bound.main_commit
        )
        == bound
    )
    with pytest.raises(ValueError, match="publication"):
        validate_source_subset(
            local, digest, bound.subset.manifest, bound.main_repo_id, "b" * 40
        )
    with pytest.raises(ValueError, match="manifest"):
        validate_source_subset(
            local, digest, bound.main_manifest, bound.main_repo_id, bound.main_commit
        )
    shard = local / bound.subset.shards[0].path
    shard.unlink()
    with pytest.raises(FileNotFoundError):
        verify_materialized(local, expected_descriptor_sha256=digest)
    assert len(fetched) == 9


@pytest.mark.parametrize(
    "damage", ["numbers", "counts", "membership", "commit", "manifest"]
)
def test_tamper_invalidates_local_frozen_identity(tmp_path, source, damage):
    compact_root, bound = descriptor(tmp_path, source)
    digest = json_sha256(bound.model_dump())

    @contextmanager
    def resolve(artifact):
        yield compact_root / artifact.path

    local = tmp_path / "offline"
    materialize_subset(
        bound, local, resolver=resolve, expected_descriptor_sha256=digest
    )
    if damage == "numbers":
        with (local / bound.subset.shards[0].path).open("ab") as stream:
            stream.write(b"tampered")
    elif damage == "manifest":
        with (local / "manifest.json").open("ab") as stream:
            stream.write(b" ")
    else:
        raw = read_json(local / "subset.json")
        payload = raw["descriptor"]
        if damage == "counts":
            payload["subset"]["selection"]["records"][0]["counts"][0]["supported"] += 1
        elif damage == "membership":
            payload["subset"]["selection"]["records"].pop()
        else:
            payload["main_commit"] = "b" * 40
        # Recomputing the envelope still cannot change the externally frozen identity.
        raw["sha256"] = json_sha256(payload)
        write_json(local / "subset.json", raw, overwrite=True)
    with pytest.raises(ValueError):
        verify_materialized(local, expected_descriptor_sha256=digest)


def test_materialization_failure_removes_owned_partial_directory(tmp_path, source):
    compact_root, bound = descriptor(tmp_path, source)
    missing = bound.subset.shards[-1].path
    (compact_root / missing).unlink()

    @contextmanager
    def resolve(artifact):
        yield compact_root / artifact.path

    local = tmp_path / "offline"
    with pytest.raises(FileNotFoundError):
        materialize_subset(
            bound,
            local,
            resolver=resolve,
            expected_descriptor_sha256=json_sha256(bound.model_dump()),
        )
    assert not local.exists()
    with pytest.raises(ValueError):
        bind_release(
            bound.subset,
            main_repo_id="owner/main",
            main_commit="main",
            main_manifest=bound.main_manifest,
            main_catalog=bound.main_catalog,
            main_manifest_path=source[0] / "manifest.json",
            main_catalog_path=source[0] / "catalog.json",
        )


@pytest.mark.parametrize("damage", ["original-row", "duplicate"])
def test_audit_callback_source_errors_poison_extraction(tmp_path, source, damage):
    _, manifest, catalog, policy, originals = source
    selected = select_subset(manifest, catalog, policy)
    row = catalog.records[0]
    record = originals[row.record.rally_id]
    identity = SimpleNamespace(
        shard_path=row.shard_path,
        shard_sha256=row.shard_sha256,
        row_index=row.row_index,
        split=row.record.split,
        rally_id=row.record.rally_id,
        view_ids=tuple(v.view_id for v in row.record.views),
    )
    output = tmp_path / "bad-callback"
    with SubsetExtractor(output, manifest, catalog, selected) as visitor:
        if damage == "original-row":
            identity.row_index += 1
        else:
            visitor(record, identity)
        with pytest.raises(ValueError):
            visitor(record, identity)
        assert not output.exists()
        with pytest.raises(ValueError, match="closed"):
            visitor.finish(source_audit_report={})


def test_materializer_rejects_oversized_input_before_overrun(tmp_path, source):
    compact_root, bound = descriptor(tmp_path, source)
    oversized = compact_root / bound.subset.manifest.path
    with oversized.open("ab") as stream:
        stream.write(b"extra bytes")

    @contextmanager
    def resolve(artifact):
        yield compact_root / artifact.path

    local = tmp_path / "overrun"
    with pytest.raises(ValueError, match="exceeds declared size"):
        materialize_subset(
            bound,
            local,
            resolver=resolve,
            expected_descriptor_sha256=json_sha256(bound.model_dump()),
        )
    assert not local.exists()


@pytest.mark.parametrize(
    "damage", ["artifact-hash", "manifest", "catalog", "incomplete", "provenance"]
)
def test_release_binding_checks_actual_original_metadata(tmp_path, source, damage):
    _, bound = descriptor(tmp_path, source)
    root, manifest, catalog, _, _ = source
    extracted = bound.subset
    manifest_artifact, catalog_artifact = bound.main_manifest, bound.main_catalog
    if damage == "artifact-hash":
        manifest_artifact = manifest_artifact.model_copy(update={"sha256": "f" * 64})
    elif damage == "manifest":
        other = manifest.model_copy(
            update={"provenance": manifest.provenance.model_copy(update={"seed": 4})}
        )
        write_manifest(root / "manifest.json", other, overwrite=True)
    elif damage == "catalog":
        changed = catalog.model_dump()
        changed["definitions"]["duration"] = "different frozen bin definition"
        write_json(root / "catalog.json", changed, overwrite=True)
    elif damage == "incomplete":
        write_manifest(
            root / "manifest.json",
            manifest.model_copy(update={"state": "incomplete"}),
            overwrite=True,
        )
    else:
        extracted = extracted.model_copy(update={"source_provenance_sha256": "f" * 64})
    # Rehashed file specs must still match the actual canonical selection inputs.
    if damage in {"manifest", "incomplete"}:
        path = root / "manifest.json"
        manifest_artifact = Artifact(
            path="manifest.json",
            sha256=file_sha256(path),
            size_bytes=path.stat().st_size,
        )
    if damage == "catalog":
        path = root / "catalog.json"
        catalog_artifact = Artifact(
            path="catalog.json",
            sha256=file_sha256(path),
            size_bytes=path.stat().st_size,
        )
    with pytest.raises(ValueError):
        bind_release(
            extracted,
            main_repo_id=bound.main_repo_id,
            main_commit=bound.main_commit,
            main_manifest=manifest_artifact,
            main_catalog=catalog_artifact,
            main_manifest_path=root / "manifest.json",
            main_catalog_path=root / "catalog.json",
        )


def test_materialization_checks_original_provenance_binding(tmp_path, source):
    compact_root, bound = descriptor(tmp_path, source)
    inconsistent = bound.model_copy(
        update={
            "subset": bound.subset.model_copy(
                update={"source_provenance_sha256": "f" * 64}
            )
        }
    )

    @contextmanager
    def resolve(artifact):
        yield compact_root / artifact.path

    with pytest.raises(ValueError, match="provenance differs"):
        materialize_subset(
            inconsistent,
            tmp_path / "bad-provenance",
            resolver=resolve,
            expected_descriptor_sha256=json_sha256(inconsistent.model_dump()),
        )


def test_materialization_group_admission_precedes_decode(tmp_path, source, monkeypatch):
    compact_root, bound = descriptor(tmp_path, source)
    digest = json_sha256(bound.model_dump())

    @contextmanager
    def resolve(artifact):
        yield compact_root / artifact.path

    local = tmp_path / "bounded"
    limit = 32 << 20
    materialize_subset(
        bound,
        local,
        resolver=resolve,
        expected_descriptor_sha256=digest,
        max_group_bytes=limit,
    )
    assert read_json(local / "train.source.json")["max_group_bytes"] == limit
    import pyarrow.parquet as pq

    def forbidden(*args, **kwargs):
        pytest.fail("oversize materialization reached numeric decode")

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", forbidden)
    with pytest.raises(ValueError, match="admission bound"):
        verify_materialized(local, expected_descriptor_sha256=digest, max_group_bytes=1)
    failed = tmp_path / "too-small"
    with pytest.raises(ValueError, match="admission bound"):
        materialize_subset(
            bound,
            failed,
            resolver=resolve,
            expected_descriptor_sha256=digest,
            max_group_bytes=1,
        )
    assert not failed.exists()
    with pytest.raises(ValueError, match="positive"):
        verify_materialized(local, expected_descriptor_sha256=digest, max_group_bytes=0)
    with pytest.raises(ValueError, match="positive"):
        materialize_subset(
            bound,
            failed,
            resolver=resolve,
            expected_descriptor_sha256=digest,
            max_group_bytes=0,
        )
    assert not failed.exists()


def test_indexed_selection_matches_exhaustive_policy_and_revalidates_mutations(source):
    _, manifest, catalog, policy, _ = source
    for row in catalog.records:
        row.strata["family"] = row.record.family_id
    catalog.definitions["family"] = "Declared fixture family identity."
    # Different equality key sets are allowed when their actual membership is disjoint.
    policy.strata[1].equals["family"] = "family-train"
    selected = select_subset(manifest, catalog, policy)
    expected = []
    for quota in sorted(policy.strata, key=lambda s: (s.split, s.name)):
        candidates = [
            row
            for row in catalog.records
            if row.record.split == quota.split
            and row.record.rally_id not in policy.reserved_rally_ids
            and all(row.strata[k] == v for k, v in quota.equals.items())
        ]
        expected.extend(
            sorted(
                candidates,
                key=lambda row: (
                    json_sha256(
                        [
                            policy.algorithm,
                            policy.seed,
                            quota.split,
                            quota.name,
                            row.record.rally_id,
                        ]
                    ),
                    row.record.rally_id,
                ),
            )[: quota.rallies]
        )
    assert selected.records == expected
    assert selected.catalog_sha256 == json_sha256(catalog.model_dump())
    # Mutating a formerly valid quota creates overlap and must be seen immediately.
    policy.strata[0].equals["duration"] = "long"
    with pytest.raises(ValueError, match="overlap"):
        select_subset(manifest, catalog, policy)
