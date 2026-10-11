"""Producer catalog joins and compact extraction during one real final audit."""

import copy
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

from test_dataset_parquet import assert_record_equal, sample  # noqa: F401

from tennis3d.artifacts import file_sha256, json_sha256
from tennis3d.datasets import subset
from tennis3d.datasets.audit import finalize_manifest
from tennis3d.datasets.io import ShardReader, ShardWriter, read_manifest
from tennis3d.datasets.schema import Manifest, Pool, Split, count_records
from tennis3d.datasets.subset import (
    CoverageCatalog,
    SelectionPolicy,
    StratumQuota,
    supported_counts,
)
from tennis3d.evaluation import metrics
from tennis3d.synthetic import producer_catalog
from tennis3d.synthetic.producer_catalog import definition_hash, summarize
from tennis3d.synthetic.producer_config import ProducerConfig
from tennis3d.synthetic.producer_state import artifact, load, save
from tennis3d.synthetic.producer_subsets import ExtractionTransaction, build_catalog


@pytest.fixture
def generated(tmp_path, sample):  # noqa: F811
    record, provenance = sample
    splits: tuple[Split, ...] = ("train", "val", "test")
    provenance = provenance.model_copy(
        update={
            "pools": [
                Pool(
                    family_id=f"family-{s}",
                    split=s,
                    sha256=str(i + 1) * 64,
                    config={"seed": 3},
                    attempts={"toss": 1},
                )
                for i, s in enumerate(splits)
            ]
        }
    )
    files = tmp_path / "files"
    files.mkdir()
    shards, commits, originals = [], [], {}
    for shard_id in range(2):
        path = f"qualification/producer/shards/{shard_id:06d}.parquet"
        records = []
        with ShardWriter(files / path, provenance, row_group_rallies=1) as writer:
            for split_id, split in enumerate(splits):
                other = copy.deepcopy(record)
                other.metadata = other.metadata.model_copy(
                    update={
                        "rally_id": f"{split}-{shard_id}",
                        "split": split,
                        "family_id": f"family-{split}",
                        "pool_sha256": str(split_id + 1) * 64,
                    }
                )
                second = copy.deepcopy(other.views[0])
                second.metadata = second.metadata.model_copy(
                    update={"view_id": 8, "view_seed": 18}
                )
                other.views.append(second)
                writer.append(other)
                records.append(summarize(other))
                originals[other.metadata.rally_id] = other
        shard = writer.close().model_copy(update={"path": path})
        shards.append(shard)
        part = f"qualification/producer/catalog/{shard_id:06d}.json"
        save(files / part, {"records": records})
        commits.append(
            {"shard": shard.model_dump(), "catalog": artifact(files, part).model_dump()}
        )
    manifest = Manifest(
        state="complete",
        provenance=provenance,
        shards=shards,
        expected=count_records(shards),
    )
    catalog = build_catalog(files, manifest, commits)
    policy = SelectionPolicy(
        seed=17,
        reserved_rally_ids=["val-0"],
        strata=[
            StratumQuota(
                name="train", split="train", equals={"duration": "under_5s"}, rallies=2
            ),
            StratumQuota(
                name="val", split="val", equals={"duration": "under_5s"}, rallies=1
            ),
        ],
    )
    execution = ProducerConfig(prefix="qualification/producer", row_group_rallies=1)
    return files, manifest, catalog, policy, execution, commits, originals


def test_catalog_counts_partitions_and_exact_source_binding(generated):
    files, manifest, catalog, _, _, _, originals = generated
    assert catalog.source_manifest_sha256 == json_sha256(manifest.model_dump())
    assert (
        catalog.definitions_source_sha256
        == json_sha256(
            {
                "producer_catalog.py": file_sha256(Path(producer_catalog.__file__)),
                "metrics.py": file_sha256(Path(metrics.__file__)),
                "subset.py": file_sha256(Path(subset.__file__)),
            }
        )
        == definition_hash()
    )
    for row in catalog.records:
        shard = next(s for s in manifest.shards if s.path == row.shard_path)
        assert row.shard_sha256 == shard.sha256
        assert row.record == shard.records[row.row_index]
        assert row.record is shard.records[row.row_index]
        assert row.counts == supported_counts(originals[row.record.rally_id])
        assert row.shot_kind_bits == 3
        assert row.physical_duration_s == 0.08
        assert row.recipe_category is None
        assert row.strata["family"] == row.record.family_id
        assert row.strata["fps"] == "25"
        assert row.strata["shot_toss"] == "present"
        assert row.strata["shot_out"] == "absent"
        for counts in row.counts:
            assert sum(counts.gap_bins.values()) == counts.supported_occluded
            assert sum(counts.distance_bins.values()) == counts.supported
    assert not list(files.rglob("search"))
    copied = CoverageCatalog.model_validate(catalog.model_dump())
    assert copied.model_dump() == catalog.model_dump()
    assert json_sha256(copied.model_dump()) == json_sha256(catalog.model_dump())


@pytest.mark.parametrize(
    "damage", ["part-hash", "row-identity", "counts", "ledger-order", "missing-part"]
)
def test_catalog_refuses_changed_or_incomplete_parts(generated, damage):
    files, manifest, _, _, _, commits, _ = generated
    commits = copy.deepcopy(commits)
    spec = commits[0]["catalog"]
    if damage == "part-hash":
        (files / spec["path"]).write_text("{}")
    elif damage == "ledger-order":
        commits.reverse()
    elif damage == "missing-part":
        commits.pop()
    else:
        part = load(files / spec["path"])
        if damage == "row-identity":
            part["records"][0]["index"]["rally_id"] = "different"
        else:
            part["records"][0]["views"][0]["distance_counts"][0] += 1
        save(files / spec["path"], part, overwrite=True)
        commits[0]["catalog"] = artifact(files, spec["path"]).model_dump()
    with pytest.raises(ValueError):
        build_catalog(files, manifest, commits)


def test_one_finalize_audit_copies_exact_selected_records_all_views(
    tmp_path, generated, monkeypatch
):
    files, manifest, catalog, policy, execution, _, originals = generated
    from tennis3d.datasets import audit

    calls, scans = [], []
    original_audit = audit.audit_dataset

    def counted(*args, **kwargs):
        scans.append(1)
        return original_audit(*args, **kwargs)

    monkeypatch.setattr(audit, "audit_dataset", counted)
    output = tmp_path / "transaction"
    with ExtractionTransaction(
        output, manifest, catalog, policy, execution
    ) as transaction:

        def visit(record, source):
            assert not record.physical["xyz_m"].flags.writeable
            calls.append((source.shard_path, source.row_index))
            transaction(record, source)

        results = []
        finalize_manifest(
            files / "final.json",
            manifest,
            root=files,
            on_record=visit,
            on_audit_complete=lambda report: results.append(transaction.finish(report)),
        )
    assert scans == [1]
    assert calls == [
        (s.path, i) for s in manifest.shards for i in range(len(s.records))
    ]
    result = results[0]
    assert load(output / "selection.json") == result.selection.model_dump()
    assert load(output / "extracted.json") == result.model_dump()
    compact = read_manifest(output / result.manifest.path)
    assert compact.shards == result.shards
    assert result.manifest.path == "qualification/producer/search/manifest.json"
    assert (output / result.manifest.path).stat().st_size == result.manifest.size_bytes
    assert file_sha256(output / result.manifest.path) == result.manifest.sha256
    copied = []
    for shard in result.shards:
        assert shard.path.startswith("qualification/producer/search/")
        with ShardReader(
            output / shard.path, shard, provenance=manifest.provenance
        ) as reader:
            for i in range(len(shard.records)):
                record = reader.read(i)
                assert_record_equal(record, originals[record.metadata.rally_id])
                assert len(record.views) == 2
                copied.append(record.metadata.rally_id)
    assert set(copied) == {"train-0", "train-1", "val-1"}


@pytest.mark.parametrize(
    "failure", ["callback", "late-global", "after-finish", "bad-report"]
)
def test_failed_audit_removes_only_owned_subset_and_no_final_manifest(
    tmp_path, generated, failure
):
    files, manifest, catalog, policy, execution, _, _ = generated
    output = tmp_path / "transaction"
    unrelated = tmp_path / "existing-pilot"
    unrelated.mkdir()
    (unrelated / "keep").write_text("immutable")
    with (
        pytest.raises(ValueError),
        ExtractionTransaction(
            output, manifest, catalog, policy, execution
        ) as transaction,
    ):

        def visit(record, source):
            transaction(record, source)
            if failure == "callback":
                raise ValueError("callback failed")

        def completed(report):
            if failure == "after-finish":
                transaction.finish(report)
            if failure == "bad-report":
                transaction.finish({})
            raise ValueError("late global check failed")

        finalize_manifest(
            files / "final.json",
            manifest,
            root=files,
            on_record=visit,
            on_audit_complete=completed,
        )
    assert not output.exists()
    assert not (files / "final.json").exists()
    assert (unrelated / "keep").read_text() == "immutable"


def test_external_infeasible_policy_and_existing_directory_are_preserved(
    tmp_path, generated
):
    _, manifest, catalog, policy, execution, _, _ = generated
    output = tmp_path / "transaction"
    infeasible = policy.model_copy(deep=True)
    infeasible.strata[0] = infeasible.strata[0].model_copy(update={"rallies": 3})
    with pytest.raises(ValueError, match="infeasible"):
        ExtractionTransaction(output, manifest, catalog, infeasible, execution)
    assert not output.exists()
    output.mkdir()
    (output / "keep").write_text("existing")
    with pytest.raises(FileExistsError):
        ExtractionTransaction(output, manifest, catalog, policy, execution)
    assert (output / "keep").read_text() == "existing"


def test_catalog_fresh_outer_validation_rejects_incompatible_expected_counts(generated):
    files, manifest, _, _, _, commits, _ = generated
    expected = dict(manifest.expected)
    expected["train"] = expected["train"].model_copy(update={"rallies": 3})
    incompatible = manifest.model_copy(update={"expected": expected})
    with pytest.raises(
        ValueError, match="complete dataset does not match expected counts"
    ):
        build_catalog(files, incompatible, commits)


@pytest.mark.parametrize("available", [False, True])
def test_selected_copy_requires_whole_output_staging_reservation(
    tmp_path, generated, monkeypatch, available
):
    from tennis3d.synthetic import producer_subsets

    files, manifest, catalog, policy, execution, _, _ = generated
    output = tmp_path / "transaction"
    checked_roots = []
    boundary = execution.staging_bytes - execution.shard_reservation()

    def measured(root):
        checked_roots.append(root)
        return boundary if available else boundary + 1

    monkeypatch.setattr(producer_subsets, "usage", measured)
    final_path = files / "staging-final.json"

    def extract():
        with ExtractionTransaction(
            output, manifest, catalog, policy, execution
        ) as transaction:
            assert transaction.selected_ids == frozenset(
                {"train-0", "train-1", "val-1"}
            )
            finalize_manifest(
                final_path,
                manifest,
                root=files,
                on_record=transaction,
                on_audit_complete=transaction.finish,
            )

    if available:
        extract()
        assert checked_roots == [output.parent] * 3
        assert output.exists()
        assert final_path.exists()
    else:
        with pytest.raises(
            ValueError, match="staging cannot reserve selected subset copy"
        ):
            extract()
        assert checked_roots == [output.parent]
        assert not output.exists()
        assert not final_path.exists()
    assert all((files / shard.path).exists() for shard in manifest.shards)
