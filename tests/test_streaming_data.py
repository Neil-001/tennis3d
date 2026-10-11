"""Pinned multi-shard parity, locality, transactional failure and cold resume."""

import copy
import shutil

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("torch")
from test_dataset_parquet import sample  # noqa: F401

from tennis3d.artifacts import file_sha256, read_json, write_json
from tennis3d.datasets.io import ShardWriter, write_manifest
from tennis3d.datasets.schema import Artifact, Manifest, count_records
from tennis3d.datasets.source import SourceSpec, V2Source
from tennis3d.lifting.data import ViewDataset, WindowSampler
from tennis3d.lifting.model import ModelConfig
from tennis3d.lifting.settings import TrainConfig


@pytest.fixture
def release(tmp_path, sample):  # noqa: F811
    original, provenance = sample
    root = tmp_path / "release"
    root.mkdir()
    pools = provenance.pools + [
        provenance.pools[0].model_copy(
            update={"family_id": "val-family", "sha256": "2" * 64, "split": "val"}
        )
    ]
    provenance = provenance.model_copy(update={"pools": pools})
    shards = []
    for si in range(3):
        with ShardWriter(
            root / f"shard-{si}.parquet", provenance, row_group_rallies=2
        ) as writer:
            for ri in range(3):
                record = copy.deepcopy(original)
                update = {"rally_id": f"rally-{si}-{ri}"}
                if si == 2:
                    update.update(
                        split="val", family_id="val-family", pool_sha256="2" * 64
                    )
                record.metadata = record.metadata.model_copy(update=update)
                second = copy.deepcopy(record.views[0])
                second.metadata = second.metadata.model_copy(update={"view_id": 7})
                record.views.append(second)
                writer.append(record)
        shards.append(writer.close())
    manifest = Manifest(
        state="complete",
        provenance=provenance,
        shards=shards,
        expected=count_records(shards),
    )
    path = root / "manifest.json"
    write_manifest(path, manifest)
    artifact = Artifact(
        path=path.name, sha256=file_sha256(path), size_bytes=path.stat().st_size
    )
    return root, manifest, artifact


def descriptor(
    tmp_path, release, *, remote=False, split="train", cache="cache", **kwargs
):
    root, manifest, artifact = release
    spec = SourceSpec(
        manifest=artifact,
        split=split,
        root=None if remote else str(root),
        repo_id="test/fixture",
        revision="a" * 40,
        cache_dir=str(tmp_path / cache),
        cache_bytes=max(s.size_bytes for s in manifest.shards) + artifact.size_bytes,
        **kwargs,
    )
    path = tmp_path / f"{cache}-{split}.source.json"
    write_json(path, spec.model_dump())
    return path


def transport(monkeypatch, root):
    from tennis3d.datasets import source

    calls = []

    def chunks(spec, artifact):
        calls.append(artifact.path)
        data = (root / artifact.path).read_bytes()
        for i in range(0, len(data), 4096):
            yield data[i : i + 4096]

    monkeypatch.setattr(source, "hub_chunks", chunks)
    return calls


def test_local_remote_parity_and_bounded_refetch(tmp_path, release, monkeypatch):
    calls = transport(monkeypatch, release[0])
    local = descriptor(tmp_path, release, cache="local")
    remote = descriptor(tmp_path, release, remote=True)
    with ViewDataset(local) as a, ViewDataset(remote) as b:
        assert a.identity == b.identity
        assert a.records == b.records
        for i in [0, 1, 6, 7, 0]:
            for key, value in a[i].items():
                np.testing.assert_array_equal(value, b[i][key])
            assert b._source.cache.usage() <= b._source.spec.cache_bytes
        assert calls.count("shard-0.parquet") == 2
        assert calls.count("shard-1.parquet") == 1
        assert not list((tmp_path / "cache").glob("*.partial"))


def test_complete_pinned_and_oversize_rejection(tmp_path, release, monkeypatch):
    calls = transport(monkeypatch, release[0])
    path = descriptor(tmp_path, release, remote=True)
    raw = read_json(path)
    raw["revision"] = "main"
    with pytest.raises(ValueError, match="immutable"):
        V2Source(path, raw)
    raw = read_json(path)
    raw["cache_bytes"] = 1
    with pytest.raises(ValueError, match="before fetch"):
        V2Source(path, raw)
    assert not calls
    incomplete = release[1].model_copy(update={"state": "incomplete"})
    write_manifest(release[0] / "manifest.json", incomplete, overwrite=True)
    raw = read_json(path)
    raw["manifest"]["sha256"] = file_sha256(release[0] / "manifest.json")
    raw["manifest"]["size_bytes"] = (release[0] / "manifest.json").stat().st_size
    with pytest.raises(ValueError, match="incomplete"):
        V2Source(path, raw)


def test_hierarchical_order_and_transactional_epoch_failure(
    tmp_path, release, monkeypatch
):
    transport(monkeypatch, release[0])
    local = descriptor(tmp_path, release, cache="local")
    remote = descriptor(tmp_path, release, remote=True)
    cfg = TrainConfig(seed=12, batch_size=2, min_frames=2, max_frames=3)
    with ViewDataset(local) as a, ViewDataset(remote) as b:
        left, right = WindowSampler(a, cfg), WindowSampler(b, cfg)
        assert left.state_dict() == right.state_dict()
        order = left.order.tolist()
        assert sorted(order) == list(range(len(a)))
        # Each shard and row group is contiguous; each rally's two views stay together.
        assert all(order[i] // 2 == order[i + 1] // 2 for i in range(0, len(order), 2))
        for i in range(0, len(order), 6):
            assert len({a._source.locations[x][0] for x in order[i : i + 6]}) == 1
        right.cursor = len(right.order) - 1
        right.draws = right.cursor
        before = copy.deepcopy(right.state_dict())
        original = b._source.read
        draws = []

        def fail_second(index):
            draws.append(index)
            if len(draws) == 2:
                raise OSError("boundary failure")
            return original(index)

        monkeypatch.setattr(b._source, "read", fail_second)
        with pytest.raises(OSError, match="boundary"):
            right.next_batch()
        assert right.state_dict() == before
        monkeypatch.setattr(b._source, "read", original)
        left.load_state_dict(before)
        from test_lifting_train import equal_state

        left_batch, right_batch = left.next_batch(), right.next_batch()
        equal_state(vars(left_batch[0]), vars(right_batch[0]))
        equal_state(left_batch[1:], right_batch[1:])
        assert left.state_dict() == right.state_dict()


def test_download_retry_and_partial_cleanup(tmp_path, release, monkeypatch):
    from tennis3d.datasets import source

    calls = transport(monkeypatch, release[0])
    normal = source.hub_chunks
    attempts = []

    def interrupted(spec, artifact):
        attempts.append(artifact.path)
        if len(attempts) == 1:
            yield b"partial"
            raise OSError("injected failure")
        yield from normal(spec, artifact)

    monkeypatch.setattr(source, "hub_chunks", interrupted)
    path = descriptor(tmp_path, release, remote=True)
    with ViewDataset(path) as dataset:
        dataset[0]
        assert attempts[:2] == ["manifest.json", "manifest.json"]
        assert calls.count("manifest.json") == 1
        assert not list((tmp_path / "cache").glob("*.partial"))


def test_exact_model_optimizer_ema_rng_cold_cache_resume(
    tmp_path, release, monkeypatch
):
    from test_lifting_train import equal_state

    from tennis3d.lifting.checkpoint import read_checkpoint
    from tennis3d.lifting.train import TrainControl, train

    transport(monkeypatch, release[0])
    cfg = TrainConfig(
        train_manifest=str(descriptor(tmp_path, release, remote=True)),
        val_manifest=str(
            descriptor(tmp_path, release, remote=True, split="val", cache="val-cache")
        ),
        model=ModelConfig(width=16, depth=2, dropout=0.1),
        seed=23,
        max_steps=4,
        min_frames=2,
        max_frames=3,
        batch_size=4,
        device="cpu",
    )

    def control(**kwargs):
        return TrainControl(
            validate_during_training=False, evaluate_at_end=False, **kwargs
        )

    full = train(cfg, tmp_path / "full", control=control())
    first = train(cfg, tmp_path / "resumed", control=control(stop_after_steps=2))
    shutil.rmtree(tmp_path / "cache")
    shutil.rmtree(tmp_path / "val-cache")
    resumed = train(
        cfg, tmp_path / "resumed", resume=first["last_checkpoint"], control=control()
    )
    a, b = (
        read_checkpoint(full["last_checkpoint"]),
        read_checkpoint(resumed["last_checkpoint"]),
    )
    for field in ("model", "optimizer", "ema", "sampler", "rng"):
        equal_state(a[field], b[field])
    assert a["sampler"]["epoch"] >= 1


def test_frozen_v2_identity_counts_and_local_only_verification(
    tmp_path, release, monkeypatch
):
    from tennis3d.autoresearch.inputs import data_identity, verify_data
    from tennis3d.autoresearch.protocol import Protocol
    from tennis3d.datasets import source

    train_path = descriptor(tmp_path, release, cache="local")
    val_path = descriptor(tmp_path, release, split="val", cache="val-local")
    protocol = Protocol(train_manifest=train_path, val_manifest=val_path, device="cpu")

    def forbidden(*args):
        pytest.fail("network inside frozen local preparation/trial")

    monkeypatch.setattr(source, "hub_chunks", forbidden)
    frozen = data_identity(protocol)
    assert frozen["counts"]["train"] == {
        "physical_rallies": 6,
        "views": 12,
        "supported_frames": 24,
    }
    assert frozen["counts"]["val"] == {
        "physical_rallies": 3,
        "views": 6,
        "supported_frames": 12,
    }
    verify_data(frozen, protocol)
    shard = release[0] / release[1].shards[0].path
    shard.unlink()
    with pytest.raises(FileNotFoundError):
        verify_data(frozen, protocol)


@pytest.mark.parametrize("failure", ["read", "post_admission"])
def test_retry_batch_after_retained_group_failure(
    tmp_path, release, monkeypatch, failure
):
    from test_lifting_train import equal_state

    path = descriptor(tmp_path, release, cache="local")
    cfg = TrainConfig(seed=12, batch_size=2, min_frames=2, max_frames=3)
    with ViewDataset(path) as a, ViewDataset(path) as b:
        left, right = WindowSampler(a, cfg), WindowSampler(b, cfg)
        right.order = np.arange(len(b))
        right.cursor = right.draws = 3  # Second view of last rally in group zero.
        before = right.state_dict()
        left.load_state_dict(before)
        b[3]
        reader = b._source.reader
        original_read = reader._parquet.read_row_group
        original_admit = reader._admit
        reached_decode, failures = [], []

        def read(group, **kwargs):
            if group == 1 and not failures:
                reached_decode.append(group)
                if failure == "read":
                    failures.append(group)
                    raise OSError("injected group read failure")
            return original_read(group, **kwargs)

        def admit(size):
            if failure == "post_admission" and reached_decode and not failures:
                failures.append(size)
                raise ValueError("injected post-read admission failure")
            original_admit(size)

        monkeypatch.setattr(reader._parquet, "read_row_group", read)
        monkeypatch.setattr(reader, "_admit", admit)
        with pytest.raises((OSError, ValueError), match="injected"):
            right.next_batch()
        assert right.state_dict() == before
        assert reader._group is None and reader._rows is None
        expected, actual = left.next_batch(), right.next_batch()
        equal_state(vars(expected[0]), vars(actual[0]))
        equal_state(expected[1:], actual[1:])
        assert left.state_dict() == right.state_dict()
