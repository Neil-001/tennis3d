"""Audited record visitor ordering, identities and finalization failure behavior."""

import copy
from contextlib import contextmanager
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from tennis3d.artifacts import json_sha256
from tennis3d.datasets import audit as dataset_audit
from tennis3d.datasets.arrays import RallyRecord, ViewRecord
from tennis3d.datasets.audit import (
    AuditedRecordSource,
    audit_dataset,
    finalize_manifest,
)
from tennis3d.datasets.io import ShardWriter
from tennis3d.datasets.schema import (
    Manifest,
    Pool,
    Provenance,
    RallyMetadata,
    ViewMetadata,
    count_records,
)
from tennis3d.synthetic.cameras import observe
from tennis3d.synthetic.config import SimulationConfig


@pytest.fixture
def sample():
    physical = {
        "t_s": np.array([0.0, 0.04, 0.08]),
        "xyz_m": np.array([[0.0, -4.0, 1.0], [0.0, -3.0, 2.0], [0.0, -2.0, 1.0]]),
        "velocity_m_s": np.array(
            [[0.0, 25.0, 1.0], [0.0, 25.0, 0.0], [0.0, 25.0, -1.0]]
        ),
        "spin_rad_s": np.zeros((3, 3)),
        "label_valid": np.array([True, False, True]),
        "segment_id": np.array([0, 1, 1], dtype=np.int64),
        "prior_id": np.array([0, 1], dtype=np.int64),
        "segment_kind": np.array([0, 1], dtype=np.int64),
        "event_t_s": np.array([0.04]),
        "event_code": np.array([2], dtype=np.int64),
        "event_xyz_m": np.array([[0.0, -3.0, 2.0]]),
        "event_precision_s": np.array([0.0]),
        "hit_before": np.array([[0.0, -3.0, 2.0, 0.0, 25.0, 1.0, 0.0, 0.0, 0.0]]),
        "hit_after": np.array([[0.0, -3.0, 2.0, 0.0, 25.0, 0.0, 0.0, 0.0, 0.0]]),
    }
    config = SimulationConfig()
    arrays = observe(physical, config, 10)
    metadata = RallyMetadata(
        rally_id="rally-0",
        family_id="family-0",
        pool_sha256="1" * 64,
        split="train",
        seed=3,
        fps=25,
        attempts=1,
    )
    record = RallyRecord(
        metadata,
        physical,
        [
            ViewRecord(
                ViewMetadata(view_id=7, view_seed=10, physical_time_origin_s=0.0),
                arrays,
            )
        ],
    )
    provenance = Provenance(
        generator_version="fixture",
        generator_sha256="0" * 64,
        code_revision="fixture",
        runtime_versions={"numpy": np.__version__},
        config_sha256=json_sha256(config.model_dump()),
        config=config.model_dump(),
        seed=3,
        pools=[
            Pool(
                family_id="family-0",
                split="train",
                sha256="1" * 64,
                config={"seed": 3},
                attempts={"toss": 1},
            )
        ],
    )
    return record, provenance


def _write_shard(path, relative_path, sample, rally_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    record, provenance = sample
    with ShardWriter(
        path,
        provenance,
        relative_path=relative_path,
        row_group_rallies=2,
    ) as writer:
        for rally_id in rally_ids:
            item = copy.deepcopy(record)
            item.metadata = item.metadata.model_copy(update={"rally_id": rally_id})
            writer.append(item)
    return writer.close()


def _manifest(provenance, shards):
    return Manifest(
        provenance=provenance,
        shards=shards,
        expected=count_records(shards),
    )


def test_visitor_order_source_resolver_and_read_only_arrays(tmp_path, sample):
    local_paths = {
        "shards/b.parquet": tmp_path / "cache" / "local-b.parquet",
        "shards/a.parquet": tmp_path / "cache" / "local-a.parquet",
    }
    shards = [
        _write_shard(
            local_paths["shards/b.parquet"],
            "shards/b.parquet",
            sample,
            ["rally-0", "rally-1"],
        ),
        _write_shard(
            local_paths["shards/a.parquet"],
            "shards/a.parquet",
            sample,
            ["rally-2"],
        ),
    ]
    manifest = _manifest(sample[1], shards)
    resolved = []
    visited = []

    @contextmanager
    def resolver(shard):
        resolved.append(shard.path)
        yield local_paths[shard.path]

    def visitor(record, source):
        assert record.metadata.rally_id == source.rally_id
        assert all(
            not array.flags.writeable
            for arrays in (record.physical, *(view.arrays for view in record.views))
            for array in arrays.values()
        )
        with pytest.raises(ValueError, match="read-only"):
            record.physical["t_s"][0] = 1.0
        with pytest.raises(FrozenInstanceError):
            source.row_index = 10
        visited.append(source)

    output = tmp_path / "manifest.json"
    finalized = finalize_manifest(
        output, manifest, resolver=resolver, on_record=visitor
    )

    assert finalized.state == "complete"
    assert resolved == ["shards/b.parquet", "shards/a.parquet"]
    assert visited == [
        AuditedRecordSource(
            shard_path=shards[0].path,
            shard_sha256=shards[0].sha256,
            row_index=0,
            split="train",
            rally_id="rally-0",
            view_ids=(7,),
        ),
        AuditedRecordSource(
            shard_path=shards[0].path,
            shard_sha256=shards[0].sha256,
            row_index=1,
            split="train",
            rally_id="rally-1",
            view_ids=(7,),
        ),
        AuditedRecordSource(
            shard_path=shards[1].path,
            shard_sha256=shards[1].sha256,
            row_index=0,
            split="train",
            rally_id="rally-2",
            view_ids=(7,),
        ),
    ]


def test_visitor_exception_propagates_without_complete_manifest(tmp_path, sample):
    shard = _write_shard(
        tmp_path / "shards" / "part.parquet",
        "shards/part.parquet",
        sample,
        ["rally-0"],
    )
    manifest = _manifest(sample[1], [shard])
    failure = RuntimeError("visitor failed")

    def visitor(record, source):
        raise failure

    output = tmp_path / "manifest.json"
    with pytest.raises(RuntimeError, match="visitor failed") as raised:
        finalize_manifest(output, manifest, on_record=visitor)
    assert raised.value is failure
    assert not output.exists()


def test_late_global_failure_after_visits_prevents_complete_manifest(
    tmp_path, sample, monkeypatch
):
    shard = _write_shard(
        tmp_path / "shards" / "part.parquet",
        "shards/part.parquet",
        sample,
        ["rally-0", "rally-1"],
    )
    manifest = _manifest(sample[1], [shard])
    visited = []

    def fail_counts(shards):
        raise ValueError("late global count failure")

    monkeypatch.setattr(dataset_audit, "count_records", fail_counts)
    output = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="late global count failure"):
        finalize_manifest(
            output,
            manifest,
            on_record=lambda record, source: visited.append(source.rally_id),
        )
    assert visited == ["rally-0", "rally-1"]
    assert not output.exists()


def test_no_callback_keeps_audit_report_unchanged(tmp_path, sample):
    shard = _write_shard(
        tmp_path / "shards" / "part.parquet",
        "shards/part.parquet",
        sample,
        ["rally-0"],
    )
    manifest = _manifest(sample[1], [shard])

    assert audit_dataset(tmp_path, manifest) == audit_dataset(
        tmp_path, manifest, on_record=None
    )


def test_final_audit_report_callback_precedes_complete_manifest(tmp_path, sample):
    record, provenance = sample
    shard = _write_shard(
        tmp_path / "part.parquet", "part.parquet", sample, [record.metadata.rally_id]
    )
    manifest = Manifest(
        provenance=provenance, shards=[shard], expected=count_records([shard])
    )
    target = tmp_path / "complete.json"
    reports = []

    def accepted(report):
        assert not target.exists()
        reports.append(report)

    finalize_manifest(target, manifest, on_audit_complete=accepted)
    assert reports[0]["counts"] == {
        s: c.model_dump() for s, c in manifest.expected.items()
    }


def test_final_audit_report_callback_failure_prevents_manifest(tmp_path, sample):
    record, provenance = sample
    shard = _write_shard(
        tmp_path / "part.parquet", "part.parquet", sample, [record.metadata.rally_id]
    )
    manifest = Manifest(
        provenance=provenance, shards=[shard], expected=count_records([shard])
    )
    target = tmp_path / "complete.json"

    def rejected(_):
        raise RuntimeError("subset finalization failed")

    with pytest.raises(RuntimeError, match="subset finalization"):
        finalize_manifest(target, manifest, on_audit_complete=rejected)
    assert not target.exists()
