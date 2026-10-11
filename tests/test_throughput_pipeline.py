"""Retained lifting fixture parity across speculation, rejection and recovery."""

import copy
import pickle
import random
import threading
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tennis3d.lifting.data import ViewDataset, WindowSampler
from tennis3d.lifting.settings import TrainConfig
from tennis3d.throughput.pipeline import create_pipeline

FIXTURE = Path(__file__).parent / "fixtures" / "lifting" / "train.manifest.json"


def config(**updates):
    return TrainConfig(
        batch_size=2, min_frames=20, max_frames=32, device="cpu"
    ).model_copy(update=updates)


def equal_batch(left, right):
    for key in vars(left[0]):
        assert torch.equal(getattr(left[0], key), getattr(right[0], key))
    assert torch.equal(left[1], right[1])
    assert torch.equal(left[2], right[2])


def wait_for_queue(pipeline, count):
    with pipeline._condition:
        assert pipeline._condition.wait_for(
            lambda: len(pipeline._queue) >= count or pipeline._failure is not None,
            timeout=10,
        )
        assert pipeline._failure is None


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_exact_batches_state_epochs_and_global_rng(depth):
    cfg = config()
    python_rng = random.getstate()
    numpy_rng = pickle.dumps(np.random.get_state())
    torch_rng = torch.get_rng_state().clone()
    with ViewDataset(FIXTURE) as dataset:
        reference = WindowSampler(dataset, cfg)
        pipeline = create_pipeline(dataset, cfg, depth=depth, max_batches=5)
        try:
            assert pipeline.metrics["generated_batches"] == 0
            assert pipeline._thread is None
            for _ in range(5):
                equal_batch(reference.next_batch(), pipeline.next_batch())
                assert pipeline.state_dict() == reference.state_dict()
            with pytest.raises(StopIteration):
                pipeline.next_batch()
            metrics = pipeline.metrics
            assert metrics["generated_batches"] == metrics["consumed_batches"] == 5
            assert metrics["reserved_peak_bytes"] <= pipeline.max_bytes
            assert metrics["outstanding_peak_batches"] <= max(1, depth)
            assert len(metrics["stall_seconds"]) == 5
            assert len(metrics["queue_delay_seconds"]) == 5
            assert pipeline.state_dict()["epoch"] > 0
        finally:
            pipeline.close()
    assert random.getstate() == python_rng
    assert pickle.dumps(np.random.get_state()) == numpy_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)


@pytest.mark.parametrize("depth", [1, 2])
def test_checkpoint_with_queued_work_stop_and_resumed_continuation(depth):
    cfg = config()
    with ViewDataset(FIXTURE) as dataset:
        reference = WindowSampler(dataset, cfg)
        pipeline = create_pipeline(dataset, cfg, depth=depth)
        try:
            equal_batch(reference.next_batch(), pipeline.next_batch())
            wait_for_queue(pipeline, depth)
            checkpoint = pipeline.state_dict()
            assert checkpoint == reference.state_dict()
            assert pipeline._sampler.draws > checkpoint["draws"]
            # Returned state belongs to the caller and cannot alter commitment.
            altered = pipeline.state_dict()
            altered["order"].clear()
            assert pipeline.state_dict() == checkpoint
            pipeline.close()
            assert pipeline.state_dict() == checkpoint
            assert pipeline.metrics["discarded_batches"] == depth
            with pytest.raises(RuntimeError, match="closed"):
                pipeline.next_batch()
            pipeline.load_state_dict(checkpoint)
            for _ in range(4):
                equal_batch(reference.next_batch(), pipeline.next_batch())
                assert pipeline.state_dict() == reference.state_dict()
        finally:
            pipeline.close()


@pytest.mark.parametrize("depth", [0, 2])
def test_failed_draw_rolls_back_and_restore_retries_same_batch(depth, monkeypatch):
    cfg = config()
    with ViewDataset(FIXTURE) as dataset:
        reference = WindowSampler(dataset, cfg)
        pipeline = create_pipeline(dataset, cfg, depth=depth)
        initial = pipeline.state_dict()
        original = pipeline._sampler._next_view
        calls = []

        def fail_second_read():
            arrays = original()
            calls.append(1)
            if len(calls) == 2:
                raise OSError("injected reader failure after advancing sampler")
            return arrays

        monkeypatch.setattr(pipeline._sampler, "_next_view", fail_second_read)
        try:
            with pytest.raises(OSError, match="injected reader"):
                pipeline.next_batch()
            assert pipeline.state_dict() == initial
            assert pipeline._sampler.state_dict() == initial
            # Repeated calls cannot advance past the failed batch.
            with pytest.raises(OSError, match="injected reader"):
                pipeline.next_batch()
            assert len(calls) == 2
            monkeypatch.setattr(pipeline._sampler, "_next_view", original)
            pipeline.load_state_dict(initial)
            for _ in range(3):
                equal_batch(reference.next_batch(), pipeline.next_batch())
                assert pipeline.state_dict() == reference.state_dict()
        finally:
            pipeline.close()


def test_close_during_inflight_read_and_resume(monkeypatch):
    cfg = config()
    entered, release = threading.Event(), threading.Event()
    with ViewDataset(FIXTURE) as dataset:
        pipeline = create_pipeline(dataset, cfg, depth=2)
        reference = WindowSampler(dataset, cfg)
        equal_batch(reference.next_batch(), pipeline.next_batch())
        pipeline.close()
        checkpoint = pipeline.state_dict()
        pipeline.load_state_dict(checkpoint)
        original = pipeline._sampler.next_batch

        def delayed_read():
            entered.set()
            assert release.wait(10)
            return original()

        monkeypatch.setattr(pipeline._sampler, "next_batch", delayed_read)
        errors = []

        def consume():
            try:
                pipeline.next_batch()
            except RuntimeError as error:
                errors.append(str(error))

        consumer = threading.Thread(target=consume)
        consumer.start()
        assert entered.wait(10)
        closer = threading.Thread(target=pipeline.close)
        closer.start()
        with pipeline._condition:
            assert pipeline._condition.wait_for(lambda: pipeline._closed, timeout=10)
        release.set()
        consumer.join(10)
        closer.join(10)
        assert not consumer.is_alive() and not closer.is_alive()
        assert errors == ["pipeline is closed"]
        assert pipeline.state_dict() == checkpoint
        monkeypatch.setattr(pipeline._sampler, "next_batch", original)
        pipeline.load_state_dict(checkpoint)
        try:
            equal_batch(reference.next_batch(), pipeline.next_batch())
            assert pipeline.state_dict() == reference.state_dict()
        finally:
            pipeline.close()


def test_prefetched_success_precedes_failure_without_committing_failed_work(
    monkeypatch,
):
    cfg = config()
    with ViewDataset(FIXTURE) as dataset:
        reference = WindowSampler(dataset, cfg)
        pipeline = create_pipeline(dataset, cfg, depth=2)
        original = pipeline._sampler.next_batch
        calls = []

        def fail_second_batch():
            calls.append(1)
            if len(calls) == 2:
                pipeline._sampler._next_view()
                raise OSError("failed after first queued batch")
            return original()

        monkeypatch.setattr(pipeline._sampler, "next_batch", fail_second_batch)
        try:
            equal_batch(reference.next_batch(), pipeline.next_batch())
            checkpoint = pipeline.state_dict()
            with pytest.raises(OSError, match="first queued"):
                pipeline.next_batch()
            assert pipeline.state_dict() == reference.state_dict() == checkpoint
            monkeypatch.setattr(pipeline._sampler, "next_batch", original)
            pipeline.load_state_dict(checkpoint)
            equal_batch(reference.next_batch(), pipeline.next_batch())
            assert pipeline.state_dict() == reference.state_dict()
        finally:
            pipeline.close()


def test_real_corruption_rejection_counters_match():
    cfg = config(seed=42, min_frames=1, max_frames=1, batch_size=4)
    with ViewDataset(FIXTURE) as dataset:
        # Retained physical arrays stay unchanged; real corruption produces
        # missing evidence and the maintained sampler rejects those crops.
        reference = WindowSampler(dataset, cfg)
        pipeline = create_pipeline(dataset, cfg, depth=2, max_batches=8)
        reference.corruption_config = reference.corruption_config.model_copy(
            update={"missing_probability": 0.5}
        )
        pipeline._sampler.corruption_config = reference.corruption_config
        try:
            for _ in range(8):
                equal_batch(reference.next_batch(), pipeline.next_batch())
                assert pipeline.state_dict() == reference.state_dict()
            assert pipeline.state_dict()["rejected"] > 0
        finally:
            pipeline.close()


def test_bounds_limit_and_invalid_restore():
    cfg = config()
    with ViewDataset(FIXTURE) as dataset:
        for kwargs in ({"depth": 3}, {"max_bytes": 1}, {"max_batches": -1}):
            with pytest.raises(ValueError):
                create_pipeline(dataset, cfg, **kwargs)
        # This cap permits one producer's intermediates but cannot also hold a
        # queued batch, so depth2 must wait for consumption before reading more.
        budget = 2 * cfg.batch_size * cfg.max_frames * 152
        pipeline = create_pipeline(
            dataset, cfg, depth=2, max_bytes=budget, max_batches=3
        )
        for _ in range(3):
            pipeline.next_batch()
        with pytest.raises(StopIteration):
            pipeline.next_batch()
        assert pipeline.metrics["generated_batches"] == 3
        assert pipeline.metrics["reserved_peak_bytes"] <= budget
        assert pipeline.metrics["outstanding_peak_batches"] == 1
        saved = pipeline.state_dict()
        broken = copy.deepcopy(saved)
        broken["cursor"] = -1
        with pytest.raises(ValueError, match="counter"):
            pipeline.load_state_dict(broken)
        assert pipeline.state_dict() == saved
        broken = copy.deepcopy(saved)
        broken["rng"]["state"]["state"] += 1
        broken["data_rng"] = {}
        with pytest.raises(ValueError):
            pipeline.load_state_dict(broken)
        assert pipeline.state_dict() == pipeline._sampler.state_dict() == saved
        pipeline.load_state_dict(saved)
        pipeline.next_batch()
        pipeline.close()
