"""Full streaming sources retain one bounded rally with independently owned arrays."""

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("torch")
from test_dataset_parquet import sample  # noqa: F401
from test_streaming_data import descriptor, release  # noqa: F401

from tennis3d.datasets import source as source_module
from tennis3d.datasets.io import ShardReader
from tennis3d.lifting.data import ViewDataset


def same(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key].dtype == expected[key].dtype
        np.testing.assert_array_equal(actual[key], expected[key])


def test_latest_rally_reuse_mutation_and_shard_switch(tmp_path, release, monkeypatch):  # noqa: F811
    path = descriptor(tmp_path, release)
    calls = []
    original = ShardReader.read

    def counted(reader, row):
        calls.append((reader.path.name, row))
        return original(reader, row)

    monkeypatch.setattr(ShardReader, "read", counted)
    with ViewDataset(path) as data:
        source = data._source
        assert source is not None
        for index in [0, 1, 0, 2, 3, 6, 7, 0]:
            si, ri, vi = source.locations[index]
            # Independent lower-level decode checks physical arrays/masks/dtypes.
            with ShardReader(
                release[0] / release[1].shards[si].path,
                release[1].shards[si],
                provenance=release[1].provenance,
            ) as reader:
                expected = original(reader, ri).views[vi].arrays
            actual = data[index]
            same(actual, expected)
            for key, array in actual.items():
                assert not np.shares_memory(array, source._rally_views[vi][key])
                array.flat[0] = 0
            same(data[index], expected)
            assert not source.views and source.view_bytes == 0
            assert (
                sum(a.nbytes for view in source._rally_views for a in view.values())
                <= source_module.RALLY_CACHE_BYTES
            )
        assert calls == [
            ("shard-0.parquet", 0),
            ("shard-0.parquet", 1),
            ("shard-1.parquet", 0),
            ("shard-0.parquet", 0),
        ]
        reader = source.reader
    assert source._rally_row is source._rally_views is source.reader is None
    assert reader._stream.closed


@pytest.mark.parametrize("offset,decodes", [(-1, 2), (0, 1)])
def test_rally_byte_admission_exact_boundary(
    tmp_path,
    release,  # noqa: F811
    monkeypatch,
    offset,
    decodes,
):
    path = descriptor(tmp_path, release)
    with ViewDataset(path) as probe:
        probe[0]
        size = sum(
            a.nbytes for view in probe._source._rally_views for a in view.values()
        )
    monkeypatch.setattr(source_module, "RALLY_CACHE_BYTES", size + offset)
    calls = []
    original = ShardReader.read

    def counted(reader, row):
        calls.append(row)
        return original(reader, row)

    monkeypatch.setattr(ShardReader, "read", counted)
    with ViewDataset(path) as data:
        data[0]
        data[1]
        assert (data._source._rally_views is not None) == (offset == 0)
        assert len(calls) == decodes


@pytest.mark.parametrize("damage", ["touch", "replace", "truncate"])
def test_reused_rally_rejects_changed_verified_shard(tmp_path, release, damage):  # noqa: F811
    path = descriptor(tmp_path, release)
    with ViewDataset(path) as data:
        data[0]
        shard = data._source.reader.path
        if damage == "touch":
            shard.touch()
        elif damage == "replace":
            replacement = shard.with_suffix(".replacement")
            replacement.write_bytes(shard.read_bytes())
            replacement.replace(shard)
        else:
            shard.write_bytes(b"invalid")
        with pytest.raises(ValueError, match="verified shard changed"):
            data[1]


def test_failed_new_rally_clears_previous_decode(tmp_path, release, monkeypatch):  # noqa: F811
    path = descriptor(tmp_path, release)
    with ViewDataset(path) as data:
        data[0]
        original = ShardReader.read

        def fail(reader, row):
            raise OSError("injected decode failure")

        with monkeypatch.context() as patch:
            patch.setattr(ShardReader, "read", fail)
            with pytest.raises(OSError, match="decode failure"):
                data[2]
            assert data._source._rally_row is data._source._rally_views is None
        expected = original(data._source.reader, 0).views[0].arrays
        same(data[0], expected)
