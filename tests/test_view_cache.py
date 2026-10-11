"""The compact-subset view cache is exact, bounded, read-only and subset-only."""

import warnings
from contextlib import contextmanager

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("torch")
from test_dataset_parquet import sample  # noqa: F401
from test_dataset_subset import descriptor, source  # noqa: F401

from tennis3d.artifacts import json_sha256, read_json, write_json
from tennis3d.datasets import source as source_module
from tennis3d.datasets.io import ShardReader
from tennis3d.datasets.subset import materialize_subset
from tennis3d.lifting.data import ViewDataset, WindowSampler
from tennis3d.lifting.settings import TrainConfig


def _open(dataset):
    opened = dataset._source
    assert opened is not None
    return opened


@pytest.fixture
def compact(tmp_path, source):  # noqa: F811
    root, bound = descriptor(tmp_path, source)
    digest = json_sha256(bound.model_dump())

    @contextmanager
    def resolve(artifact):
        yield root / artifact.path

    local = tmp_path / "offline"
    materialize_subset(
        bound, local, resolver=resolve, expected_descriptor_sha256=digest
    )
    return local / "train.source.json"


@pytest.fixture
def plain(tmp_path, compact):
    raw = read_json(compact)
    raw["subset_sha256"] = None
    path = tmp_path / "plain.source.json"
    write_json(path, raw)
    return path


@pytest.fixture
def reads(monkeypatch):
    calls = []
    original = ShardReader.read

    def counted(self, row):
        calls.append(row)
        return original(self, row)

    monkeypatch.setattr(ShardReader, "read", counted)
    return calls


def uncached(path, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(source_module, "VIEW_CACHE_BYTES", 0)
        with ViewDataset(path) as dataset:
            return [dataset[i] for i in range(len(dataset))]


def assert_same(left, right):
    assert left.keys() == right.keys()
    for key, value in left.items():
        assert value.dtype == right[key].dtype and value.shape == right[key].shape
        np.testing.assert_array_equal(value, right[key])


def test_second_pass_served_from_memory_read_only(compact, reads, monkeypatch):
    expected = uncached(compact, monkeypatch)
    reads.clear()
    with ViewDataset(compact) as dataset:
        for i in range(len(dataset)):
            assert_same(dataset[i], expected[i])
        first = len(reads)
        assert first == len(dataset) // 2
        for i in range(len(dataset)):
            arrays = dataset[i]
            assert_same(arrays, expected[i])
            assert all(not a.flags.writeable for a in arrays.values())
            arrays["extra"] = None
            assert "extra" not in dataset[i]
        assert len(reads) == first
        total = sum(
            a.nbytes
            for views in _open(dataset).views.values()
            for v in views.values()
            for a in v.values()
        )
        assert _open(dataset).view_bytes == total > 0


def test_byte_bound_stops_growth_and_stays_correct(compact, reads, monkeypatch):
    expected = uncached(compact, monkeypatch)
    with ViewDataset(compact) as probe:
        probe[0]
        one = _open(probe).view_bytes
    monkeypatch.setattr(source_module, "VIEW_CACHE_BYTES", one + one // 2)
    with ViewDataset(compact) as dataset:
        for _ in range(2):
            for i in range(len(dataset)):
                assert_same(dataset[i], expected[i])
        assert len(_open(dataset).views) == 1
        assert _open(dataset).view_bytes == one


def test_non_subset_source_reuses_only_latest_rally(plain, reads):
    with ViewDataset(plain) as dataset:
        for _ in range(2):
            for i in range(len(dataset)):
                dataset[i]
        assert len(reads) == len(dataset)
        assert not _open(dataset).views and _open(dataset).view_bytes == 0
        assert _open(dataset)._rally_views is not None


def test_sampler_bit_identical_with_and_without_cache(compact, monkeypatch):
    cfg = TrainConfig(seed=5, batch_size=3, min_frames=2, max_frames=3)

    def run(bound):
        with monkeypatch.context() as patch:
            patch.setattr(source_module, "VIEW_CACHE_BYTES", bound)
            with ViewDataset(compact) as dataset:
                sampler = WindowSampler(dataset, cfg)
                out = []
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    while sampler.epoch < 2:
                        batch = sampler.next_batch()
                        out.append(
                            [
                                getattr(batch[0], f)
                                for f in batch[0].__dataclass_fields__
                            ]
                            + list(batch[1:])
                        )
                return out, sampler.state_dict()

    (cached, cached_state), (plain, plain_state) = run(8 << 30), run(0)
    assert len(cached) == len(plain) > 1
    for a, b in zip(cached, plain, strict=True):
        for x, y in zip(a, b, strict=True):
            assert x.dtype == y.dtype and x.equal(y)
    assert cached_state == plain_state


def test_close_drops_cache(compact):
    dataset = ViewDataset(compact)
    dataset[0]
    assert _open(dataset).views
    dataset.close()
    assert not _open(dataset).views and _open(dataset).view_bytes == 0


def test_cache_hit_rejects_changed_shard(compact):
    with ViewDataset(compact) as dataset:
        dataset[0]
        source = dataset._source
        assert source is not None
        path, _ = source.view_files[source.locations[0][0]]
        path.touch()
        with pytest.raises(ValueError, match="shard changed"):
            dataset[0]
