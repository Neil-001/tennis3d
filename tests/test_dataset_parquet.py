"""Numeric fidelity, corruption refusal, immutable handles and bounded final audits."""

import copy
from contextlib import contextmanager

import numpy as np
import pytest

pytest.importorskip("pyarrow")
import pyarrow as pa
import pyarrow.parquet as pq

from tennis3d.artifacts import file_sha256, json_sha256, read_json, write_json
from tennis3d.datasets.arrays import (
    RallyRecord,
    ViewRecord,
    arrow_schema,
    decode,
    decode_row,
    encode,
)
from tennis3d.datasets.audit import audit_rally, finalize_manifest
from tennis3d.datasets.io import (
    ShardReader,
    ShardWriter,
    read_manifest,
    read_rally,
    write_manifest,
)
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
    cfg = SimulationConfig()
    arrays = observe(physical, cfg, 10)
    # Preserve a rejected off-image candidate and all missing labels.
    arrays["ball_observed"][1] = False
    arrays["ball_uv"][1] = np.nan
    arrays["ball_confidence"][1] = 0
    arrays["ball_candidate_uv"][1] = [-100.0, 2000.0]
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
                ViewMetadata(view_id=0, view_seed=10, physical_time_origin_s=0.0),
                arrays,
            )
        ],
    )
    provenance = Provenance(
        generator_version="fixture",
        generator_sha256="0" * 64,
        code_revision="fixture",
        runtime_versions={"numpy": np.__version__},
        config_sha256=json_sha256(cfg.model_dump()),
        config=cfg.model_dump(),
        seed=3,
        pools=[
            Pool.model_validate(
                {
                    "family_id": "family-0",
                    "split": "train",
                    "sha256": "1" * 64,
                    "config": {"seed": 3},
                    "attempts": {"toss": 1},
                }
            )
        ],
    )
    return record, provenance


def write_sample(tmp_path, sample, count=1):
    record, provenance = sample
    path = tmp_path / "part.parquet"
    with ShardWriter(path, provenance, row_group_rallies=2, target_bytes=1) as writer:
        for i in range(count):
            other = copy.deepcopy(record)
            other.metadata = other.metadata.model_copy(
                update={"rally_id": f"rally-{i}"}
            )
            writer.append(other)
        assert writer.target_reached
    return path, writer.close()


def assert_record_equal(a, b):
    assert a.metadata == b.metadata
    assert a.index() == b.index()
    for left, right in [
        (a.physical, b.physical),
        *[(x.arrays, y.arrays) for x, y in zip(a.views, b.views)],
    ]:
        assert left.keys() == right.keys()
        for key in left:
            assert left[key].dtype == right[key].dtype
            np.testing.assert_array_equal(left[key], right[key])


def test_typed_roundtrip_row_groups_and_masks(tmp_path, sample):
    path, shard = write_sample(tmp_path, sample, count=3)
    record, provenance = sample
    assert pq.ParquetFile(path).num_row_groups == 2
    with ShardReader(path, shard, provenance=provenance) as reader:
        assert_record_equal(record, reader.read(0))
        assert reader.read(2).metadata.rally_id == "rally-2"
        assert_record_equal(record, reader.read(0))
    assert_record_equal(record, read_rally(path, shard, 0, provenance=provenance))


def distinct_rally(record, i):
    # Different values per rally and per view, so wrong offsets cannot match.
    other = copy.deepcopy(record)
    other.metadata = other.metadata.model_copy(update={"rally_id": f"rally-{i}"})
    for key in ("xyz_m", "velocity_m_s", "spin_rad_s"):
        other.physical[key] = other.physical[key] + i
    other.physical["segment_id"] = other.physical["segment_id"] + i
    for v in range(1 + i % 3):
        view = copy.deepcopy(record.views[0])
        view.metadata = view.metadata.model_copy(update={"view_id": v})
        for key in ("xyz_m", "velocity_m_s", "spin_rad_s"):
            view.arrays[key] = other.physical[key].copy()
        view.arrays["court_uv"] = view.arrays["court_uv"] + 100 * i + v
        view.arrays["camera_t"] = view.arrays["camera_t"] + 10 * i + v
        view.arrays["image_size"] = view.arrays["image_size"] + i + v
        view.arrays["ball_in_frame"] = np.roll(view.arrays["ball_in_frame"], v)
        if v:
            other.views.append(view)
        else:
            other.views = [view]
    return other


def test_arrow_decode_matches_python_decode(tmp_path, sample):
    rows = [encode(distinct_rally(sample[0], i)) for i in range(5)]
    path = tmp_path / "rows.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=arrow_schema()), path, row_group_size=2
    )
    file = pq.ParquetFile(path)
    assert file.num_row_groups == 3
    assert {len(r["views"]) for r in rows} == {1, 2, 3}
    for group in range(file.num_row_groups):
        table = file.read_row_group(group)
        for i in range(table.num_rows):
            old = decode(table.slice(i, 1).to_pylist()[0])
            new = decode_row(table.slice(i, 1))
            assert old.metadata == new.metadata
            assert len(old.views) == len(new.views)
            pairs = [(old.physical, new.physical)] + [
                (a.arrays, b.arrays) for a, b in zip(old.views, new.views)
            ]
            for x, y in pairs:
                assert x.keys() == y.keys()
                for key in x:
                    assert x[key].dtype == y[key].dtype
                    assert x[key].shape == y[key].shape
                    assert np.array_equal(x[key], y[key], equal_nan=True)
            for a, b in zip(old.views, new.views):
                assert a.metadata == b.metadata


def nullable(kind):
    if pa.types.is_struct(kind):
        return pa.struct(
            [nullable_field(kind.field(i)) for i in range(kind.num_fields)]
        )
    if pa.types.is_fixed_size_list(kind):
        return pa.list_(nullable_field(kind.value_field), kind.list_size)
    if pa.types.is_list(kind):
        return pa.list_(nullable_field(kind.value_field))
    return kind


def nullable_field(field):
    return pa.field(field.name, nullable(field.type), nullable=True)


@pytest.mark.parametrize("where", ["array", "list", "element", "scalar", "view"])
def test_arrow_decode_rejects_nulls_at_every_level(sample, where):
    raw = encode(sample[0])
    if where == "array":
        raw["physical"]["xyz_m"] = None
    elif where == "list":
        raw["physical"]["xyz_m"][0] = None
    elif where == "element":
        raw["physical"]["xyz_m"][0][1] = None
    elif where == "scalar":
        raw["physical"]["t_s"][1] = None
    else:
        raw["views"][0]["arrays"]["court_uv"][0][3][1] = None
    schema = pa.schema(nullable_field(f) for f in arrow_schema())
    table = pa.Table.from_pylist([raw], schema=schema)
    with pytest.raises(ValueError, match="Arrow null in numeric array"):
        decode(table.to_pylist()[0])
    with pytest.raises(ValueError, match="Arrow null in numeric array"):
        decode_row(table)


def test_arrow_decode_rejects_wrong_inner_size(sample):
    raw = encode(sample[0])
    schema = arrow_schema()
    bad = pa.struct(
        [
            pa.field(
                f.name,
                pa.list_(pa.list_(pa.float64())) if f.name == "xyz_m" else f.type,
                nullable=False,
            )
            for f in schema.field("physical").type
        ]
    )
    raw["physical"]["xyz_m"][0] = [0.0, 1.0]
    fields = [
        pa.field("physical", bad, nullable=False) if f.name == "physical" else f
        for f in schema
    ]
    table = pa.Table.from_pylist([raw], schema=pa.schema(fields))
    with pytest.raises(ValueError):
        decode(table.to_pylist()[0])
    with pytest.raises(ValueError):
        decode_row(table)


def test_arrow_decoded_arrays_outlive_row_group_and_reader(tmp_path, sample):
    path, shard = write_sample(tmp_path, sample, count=5)
    with ShardReader(path, shard, provenance=sample[1]) as reader:
        first = reader.read(1)
        reader.read(4)
        reader.read(2)
    expected = sample[0]
    arrays = [first.physical, *[v.arrays for v in first.views]]
    for got, want in zip(arrays, [expected.physical, expected.views[0].arrays]):
        for key, value in got.items():
            assert value.flags.writeable and value.flags.c_contiguous
            assert np.array_equal(value, want[key], equal_nan=True)
            value[...] = value  # Writable memory, not a read-only Arrow view.


@pytest.mark.parametrize(
    "damage", ["truncate", "append", "provenance", "schema", "membership"]
)
def test_reject_corrupt_shards(tmp_path, sample, damage):
    path, shard = write_sample(tmp_path, sample)
    provenance = sample[1]
    if damage == "truncate":
        path.write_bytes(path.read_bytes()[:100])
    elif damage == "append":
        with path.open("ab") as f:
            f.write(b"tampered")
    elif damage == "provenance":
        provenance = provenance.model_copy(update={"generator_sha256": "9" * 64})
    elif damage == "schema":
        table = pq.read_table(path).drop(["physical"])
        pq.write_table(table, path)
        shard = shard.model_copy(
            update={"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
        )
    else:
        shard = shard.model_copy(
            update={"records": [shard.records[0].model_copy(update={"seed": 4})]}
        )
    with pytest.raises(ValueError):
        read_rally(path, shard, 0, provenance=provenance)


def test_reader_reuses_hash_but_detects_change(tmp_path, sample, monkeypatch):
    path, shard = write_sample(tmp_path, sample)
    from tennis3d.datasets import io

    original = io.hashlib.file_digest
    calls = []

    def digest(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(io.hashlib, "file_digest", digest)
    with ShardReader(path, shard, provenance=sample[1]) as reader:
        reader.read(0)
        reader.read(0)
        assert len(calls) == 1
        replacement = tmp_path / "replacement"
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
        with pytest.raises(ValueError, match="changed"):
            reader.read(0)


@pytest.mark.parametrize(
    "damage", ["dtype", "physical", "projection", "origin", "extra"]
)
def test_writer_audits_before_publication(tmp_path, sample, damage):
    record, provenance = sample
    if damage == "dtype":
        record.views[0].arrays["xyz_m"] = (
            record.views[0].arrays["xyz_m"].astype(np.float32)
        )
    elif damage == "physical":
        record.physical = {**record.physical, "xyz_m": record.physical["xyz_m"] + 1}
    elif damage == "projection":
        record.views[0].arrays["court_uv"][0, 0, 0] += 1
    elif damage == "origin":
        record.views[0].metadata = record.views[0].metadata.model_copy(
            update={"physical_time_origin_s": 0.001}
        )
    else:
        record.physical["unknown"] = np.ones(3)
    with (
        pytest.raises(ValueError),
        ShardWriter(tmp_path / "bad.parquet", provenance) as writer,
    ):
        writer.append(record)
    assert not (tmp_path / "bad.parquet").exists()
    assert not list(tmp_path.glob(".*"))


def test_finalization_resolves_evicted_shards_one_at_a_time(tmp_path, sample):
    path, shard = write_sample(tmp_path, sample)
    remote = tmp_path / "remote"
    remote.mkdir()
    path.replace(remote / path.name)
    manifest = Manifest(
        provenance=sample[1], shards=[shard], expected=count_records([shard])
    )
    write_manifest(tmp_path / "incomplete.json", manifest)
    with pytest.raises(ValueError, match="incomplete"):
        read_manifest(tmp_path / "incomplete.json")
    calls = []

    @contextmanager
    def resolver(artifact):
        assert not path.exists()
        calls.append(artifact.sha256)
        path.write_bytes((remote / artifact.path).read_bytes())
        try:
            yield path
        finally:
            path.unlink()

    result = finalize_manifest(tmp_path / "manifest.json", manifest, resolver=resolver)
    assert result.state == "complete" and len(calls) == 1 and not path.exists()
    assert read_manifest(tmp_path / "manifest.json") == result
    raw = read_json(tmp_path / "manifest.json")
    raw["manifest"]["provenance"]["seed"] += 1
    write_json(tmp_path / "manifest.json", raw, overwrite=True)
    with pytest.raises(ValueError, match="checksum"):
        read_manifest(tmp_path / "manifest.json")


def test_counts_leakage_and_false_completion(tmp_path, sample):
    path, shard = write_sample(tmp_path, sample)
    manifest = Manifest(
        provenance=sample[1], shards=[shard], expected=count_records([shard])
    )
    expected = count_records([shard])
    expected["train"] = expected["train"].model_copy(update={"rallies": 2})
    incomplete = manifest.model_copy(update={"expected": expected})
    with pytest.raises(ValueError, match="expected counts"):
        finalize_manifest(tmp_path / "complete.json", incomplete)
    assert not (tmp_path / "complete.json").exists()
    row = shard.records[0].model_copy(update={"rally_id": "val-rally", "split": "val"})
    bad = shard.model_copy(update={"records": [shard.records[0], row]})
    with pytest.raises(ValueError, match="cross-split"):
        Manifest(provenance=sample[1], shards=[bad], expected=count_records([bad]))
    # A complete count does not make a corrupted physical artifact pass finalization.
    path.write_bytes(b"broken")
    with pytest.raises(ValueError):
        finalize_manifest(tmp_path / "complete.json", manifest)


def test_physical_audit_rejects_bad_events(sample):
    record = sample[0]
    record.physical["event_code"][0] = 4
    with pytest.raises(ValueError, match="hit count"):
        audit_rally(record)


@pytest.mark.parametrize("value", [np.nan, np.inf, 42.0])
def test_masked_physical_states_roundtrip_without_changing_observations(
    tmp_path, sample, value
):
    record, provenance = sample
    # observe shares physical frame arrays; give both independent identical states.
    for key in ("xyz_m", "velocity_m_s", "spin_rad_s"):
        record.physical[key] = record.physical[key].copy()
        record.physical[key][1] = value
        record.views[0].arrays[key] = record.physical[key].copy()
    assert not record.physical["label_valid"][1]
    audit_rally(record)
    path, shard = write_sample(tmp_path, sample)
    restored = read_rally(path, shard, 0, provenance=provenance)
    assert_record_equal(record, restored)
    manifest = Manifest(
        provenance=provenance, shards=[shard], expected=count_records([shard])
    )
    finalize_manifest(tmp_path / "manifest.json", manifest)
    # Equality with matching NaNs must still reject a different masked state.
    record.views[0].arrays["xyz_m"][1, 0] = -17.0
    with pytest.raises(ValueError, match="view/physical xyz_m mismatch"):
        audit_rally(record)


@pytest.mark.parametrize("failure", ["hash", "stat"])
def test_descriptor_failure_does_not_publish_shard(
    tmp_path, sample, monkeypatch, failure
):
    from tennis3d.datasets import io

    path = tmp_path / "part.parquet"
    writer = ShardWriter(path, sample[1])
    writer.append(sample[0])

    def fail(*args):
        raise OSError("injected descriptor failure")

    with monkeypatch.context() as patch:
        patch.setattr(
            io.hashlib if failure == "hash" else io.os,
            "sha256" if failure == "hash" else "fstat",
            fail,
        )
        with pytest.raises(OSError, match="injected descriptor"):
            writer.close()
    assert not path.exists()
    assert not list(tmp_path.iterdir())
    # The same destination can be retried immediately after a failed close.
    with ShardWriter(path, sample[1]) as retry:
        retry.append(sample[0])
    assert_record_equal(
        sample[0], read_rally(path, retry.close(), 0, provenance=sample[1])
    )
