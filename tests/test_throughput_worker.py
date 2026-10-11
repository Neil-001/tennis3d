"""Consumer diagnostics tolerate producer shard and cache transitions."""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pyarrow")
torch = pytest.importorskip("torch")

from tennis3d.datasets import source as source_module
from tennis3d.lifting.data import ViewDataset, WindowSampler
from tennis3d.lifting.settings import TrainConfig
from tennis3d.throughput import worker
from tennis3d.throughput.pipeline import create_pipeline

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/lifting/train.manifest.json"


def test_measure_consumes_batch_while_producer_reopens_next_shard(
    tmp_path, monkeypatch
):
    entered, release = threading.Event(), threading.Event()
    reads, downloads, updates, pipelines = [], [], [], []
    owned_cache = tmp_path / "source-cache"
    owned_cache.mkdir()
    group_bytes, cache_bytes = 256 << 20, 4 << 30
    cfg = TrainConfig(batch_size=1, min_frames=20, max_frames=32, device="cpu")

    with ViewDataset(FIXTURE) as retained:
        arrays = retained[0]

        class Reader:
            def __init__(self, path, shard, *, provenance=None, max_group_bytes=None):
                self.path, self.shard = path, shard
                self.max_group_bytes = max_group_bytes

            def read(self, _):
                reads.append(self.shard.path)
                return SimpleNamespace(views=[SimpleNamespace(arrays=arrays)])

            def close(self):
                pass

        class Cache:
            root = owned_cache
            pinned = None

            def get(self, shard):
                downloads.append(shard.path)
                entered.set()
                assert release.wait(10), (
                    "consumer did not release second-shard download"
                )
                return self.root / shard.path

            def close(self):
                pass

        # Use the maintained V2Source.read state machine. Its reader is legally
        # None while the producer fetches the shard after closing the old reader.
        source = object.__new__(source_module.V2Source)
        source._rally_row = source._rally_views = None
        source.locations = [(0, 0, 0), (1, 0, 0)]
        source.views, source.view_files = {}, {}
        source.view_bytes = 0
        source.root = None
        source.__dict__["spec"] = SimpleNamespace(
            subset_sha256=None, max_group_bytes=group_bytes, cache_bytes=cache_bytes
        )
        shards = [SimpleNamespace(path="a" * 64), SimpleNamespace(path="b" * 64)]
        source.__dict__["manifest"] = SimpleNamespace(shards=shards, provenance=None)
        source.__dict__["cache"] = Cache()
        source.__dict__["reader"] = Reader(
            owned_cache / shards[0].path, shards[0], max_group_bytes=group_bytes
        )
        source.active = 0

        class Dataset(ViewDataset):
            identity = retained.identity
            format_version = retained.format_version
            _source = source

            def __init__(self):
                pass

            def __len__(self):
                return len(source.locations)

            def simulation_config(self):
                return retained.simulation_config()

            def make_order(self, rng):
                # Only sampler metadata is constructed. Physical arrays are the
                # unchanged checksummed retained fixture above.
                return np.arange(len(self))

            def __getitem__(self, index):
                return source.read(index)

            def verify_identity(self):
                retained.verify_identity()

            def close(self):
                source.close()

        dataset = Dataset()
        initial = WindowSampler(dataset, cfg).state_dict()

        def synchronized_pipeline(*args, **kwargs):
            pipeline = create_pipeline(*args, **kwargs)
            pipelines.append(pipeline)
            next_batch = pipeline.next_batch

            def next_with_transition():
                batch = next_batch()
                if pipeline.metrics["consumed_batches"] == 1:
                    assert entered.wait(10), "producer did not begin shard transition"
                    assert source.reader is None
                    assert not release.is_set()
                return batch

            monkeypatch.setattr(pipeline, "next_batch", next_with_transition)
            return pipeline

        def update(_model, _ema, _optimizer, _cfg, batch, _device):
            if not updates:
                assert source.reader is None
                assert entered.is_set() and not release.is_set()
                release.set()
            updates.append(worker.batch_digest(batch))

        monkeypatch.setattr(source_module, "ShardReader", Reader)
        # Track the original transport so measure's wrapper is restored too.
        monkeypatch.setattr(source_module, "hub_chunks", source_module.hub_chunks)
        monkeypatch.setattr(worker, "ViewDataset", lambda _: dataset)
        monkeypatch.setattr(worker, "recipe", lambda *args: cfg)
        monkeypatch.setattr(worker, "model_state", lambda *args: (None, None, None))
        monkeypatch.setattr(worker, "update", update)
        monkeypatch.setattr(worker, "create_pipeline", synchronized_pipeline)
        # Ownership/admission checks have independent tests; this reproduces only
        # the consumer race after a valid reader has already been admitted.
        monkeypatch.setattr(
            worker, "install_source_guards", lambda *a, **k: owned_cache
        )
        monkeypatch.setattr(worker, "synchronize", lambda _: None)
        monkeypatch.setattr(worker, "environment", lambda _: {"device": "cpu"})
        monkeypatch.setattr(worker, "random_state", dict)
        monkeypatch.setattr(worker.importlib.metadata, "version", lambda _: "fixture")
        monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
        descriptor = tmp_path / "source.json"
        descriptor.write_text(json.dumps({"max_group_bytes": group_bytes}))
        request = {
            "source": str(ROOT),
            "rules": {
                "threads": 1,
                "cache_bytes": cache_bytes,
                "prefetch_bytes": 64 << 20,
                "steps": 2,
                "steady_skip_steps": 1,
            },
            "workload": {"seed": cfg.seed},
            "process_started": time.perf_counter(),
            "descriptor": str(descriptor),
            "data_identity": dataset.identity,
            "initial_state": initial,
            "depth": 2,
            "source_sha256": "c" * 64,
            "cache_template_sha256": "d" * 64,
        }
        try:
            report = worker.measure(request)
        finally:
            release.set()
            for pipeline in pipelines:
                pipeline.close()
            dataset.close()
        assert len(updates) == report["steps"] == report["accepted_batches"] == 2
        assert report["pipeline"]["generated_batches"] == 2
        assert report["pipeline"]["consumed_batches"] == 2
        assert report["draws"] == 2
        assert reads == [shard.path for shard in shards]
        assert downloads == [shards[1].path]


def test_cache_usage_tolerates_partial_rename_between_listing_and_lstat(
    tmp_path, monkeypatch
):
    partial, final = tmp_path / ("a" * 64 + ".partial"), tmp_path / ("a" * 64)
    partial.write_bytes(b"owned")
    entered, renamed = threading.Event(), threading.Event()
    original = Path.lstat

    def transition():
        assert entered.wait(10)
        partial.replace(final)
        renamed.set()

    def delayed_lstat(path, *args, **kwargs):
        if path == partial:
            entered.set()
            assert renamed.wait(10)
        return original(path, *args, **kwargs)

    thread = threading.Thread(target=transition)
    thread.start()
    try:
        monkeypatch.setattr(Path, "lstat", delayed_lstat)
        assert worker.cache_usage(tmp_path, 5) in (0, 5)
    finally:
        entered.set()
        thread.join(10)
    assert not thread.is_alive()
    assert renamed.is_set()
    assert final.read_bytes() == b"owned"
    assert worker.cache_usage(tmp_path, 5) == 5


@pytest.mark.parametrize(
    "invalid", ["symlink", "unknown-name", "directory", "over-cap"]
)
def test_cache_usage_still_rejects_existing_unowned_entries(tmp_path, invalid):
    cache = tmp_path / "cache"
    cache.mkdir()
    entry = cache / ("a" * 64)
    if invalid == "symlink":
        target = tmp_path / "outside"
        target.write_bytes(b"owned")
        entry.symlink_to(target)
    elif invalid == "unknown-name":
        (cache / "unowned").write_bytes(b"owned")
    elif invalid == "directory":
        entry.mkdir()
    else:
        entry.write_bytes(b"owned")
    message = "disk bound" if invalid == "over-cap" else "unowned path"
    with pytest.raises(ValueError, match=message):
        worker.cache_usage(cache, 4)
