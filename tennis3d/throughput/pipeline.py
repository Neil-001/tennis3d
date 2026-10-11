"""Bounded CPU prefetch with checkpoints at the last consumed batch.

The dataset belongs to the caller. Only the producer touches its sampler while
running. No producer starts until the first next_batch call. A generation limit
also limits speculative reads, so a measured workload never prepares batch N+1.
"""

import copy
import threading
import time
from collections import deque
from dataclasses import fields, is_dataclass
from typing import TypedDict

import torch

from tennis3d.lifting.data import WindowSampler

MAX_BYTES = 64 * 1024 * 1024
# Features, target and label bytes per padded frame in the maintained sampler.
BYTES_PER_FRAME = 152


class _Metrics(TypedDict):
    stall_seconds: list[float]
    queue_delay_seconds: list[float]
    producer_seconds: float
    setup_seconds: float
    close_seconds: float
    queued_peak_bytes: int
    reserved_peak_bytes: int
    outstanding_peak_batches: int
    discarded_batches: int


def _capture(sampler):
    # Copy the NumPy permutation without constructing a large Python list at
    # every batch. Copies protect checkpoints even if a candidate mutates order.
    return {
        "version": 2,
        "identity": copy.deepcopy(sampler.dataset.identity),
        "rng": copy.deepcopy(sampler.rng.bit_generator.state),
        "data_rng": copy.deepcopy(sampler.data_rng.bit_generator.state),
        "order": sampler.order.copy(),
        **{
            key: getattr(sampler, key)
            for key in ("cursor", "epoch", "draws", "rejected")
        },
    }


def _public(state):
    return {
        key: value.tolist() if key == "order" else copy.deepcopy(value)
        for key, value in state.items()
    }


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if is_dataclass(value):
        return sum(_tensor_bytes(getattr(value, field.name)) for field in fields(value))
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


class Pipeline:
    """Single-consumer pipeline; depth bounds queued plus in-flight batches.

    max_bytes bounds retained queued tensors plus a conservative reservation for
    the producer's per-window tensors and final stacked batch. It excludes the
    caller's consumed batch, source decoding, sampler metadata and model memory.
    Process RAM must still be measured independently. The padded tensor schema
    is fixed; a candidate exceeding its reservation fails closed.

    load_state_dict drains the old producer, discards speculation and resets the
    generation limit relative to the restored state. Failure is terminal until
    restoration. state_dict remains usable after failure or close.
    """

    def __init__(self, dataset, config, *, depth, max_bytes, max_batches):
        started = time.perf_counter()
        if type(depth) is not int or not 0 <= depth <= 2:
            raise ValueError("prefetch depth must be an integer between 0 and 2")
        if type(max_bytes) is not int or not 0 < max_bytes <= MAX_BYTES:
            raise ValueError("prefetch byte budget must be between 1 and 64 MiB")
        if max_batches is not None and (
            type(max_batches) is not int or max_batches < 0
        ):
            raise ValueError("max_batches must be a nonnegative integer or None")
        self._batch_bytes = config.batch_size * config.max_frames * BYTES_PER_FRAME
        self._reservation = 2 * self._batch_bytes
        if self._reservation > max_bytes:
            raise ValueError("one producer batch exceeds the prefetch byte budget")
        self.depth, self.max_bytes, self.max_batches = depth, max_bytes, max_batches
        self._sampler = WindowSampler(dataset, config)
        self._committed = _capture(self._sampler)
        self._condition = threading.Condition()
        self._queue = deque()
        self._thread = None
        self._closed = False
        self._failure = None
        self._generated = self._consumed = self._queued_bytes = self._inflight = 0
        self._metrics: _Metrics = {
            "stall_seconds": [],
            "queue_delay_seconds": [],
            "producer_seconds": 0.0,
            "setup_seconds": time.perf_counter() - started,
            "close_seconds": 0.0,
            "queued_peak_bytes": 0,
            "reserved_peak_bytes": 0,
            "outstanding_peak_batches": 0,
            "discarded_batches": 0,
        }

    @property
    def metrics(self):
        with self._condition:
            return {
                **copy.deepcopy(self._metrics),
                "generated_batches": self._generated,
                "consumed_batches": self._consumed,
                "queued_batches": len(self._queue),
                "queued_bytes": self._queued_bytes,
                "inflight_reserved_bytes": self._inflight,
            }

    def state_dict(self):
        with self._condition:
            return _public(self._committed)

    def _limit_reached(self):
        return self.max_batches is not None and self._generated >= self.max_batches

    def _record_peak(self):
        self._metrics["queued_peak_bytes"] = max(
            self._metrics["queued_peak_bytes"], self._queued_bytes
        )
        self._metrics["reserved_peak_bytes"] = max(
            self._metrics["reserved_peak_bytes"], self._queued_bytes + self._inflight
        )
        self._metrics["outstanding_peak_batches"] = max(
            self._metrics["outstanding_peak_batches"],
            len(self._queue) + bool(self._inflight),
        )

    def _build(self):
        before = _capture(self._sampler)
        started = time.perf_counter()
        try:
            batch = self._sampler.next_batch()
            size = _tensor_bytes(batch)
            if size > self._batch_bytes:
                raise ValueError("candidate batch exceeds frozen tensor reservation")
            return batch, _capture(self._sampler), size, time.perf_counter()
        except BaseException:
            self._sampler.load_state_dict(_public(before))
            raise
        finally:
            with self._condition:
                self._metrics["producer_seconds"] += time.perf_counter() - started

    def _produce(self):
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: (
                        self._closed
                        or self._limit_reached()
                        or (
                            len(self._queue) < self.depth
                            and self._queued_bytes + self._reservation <= self.max_bytes
                        )
                    )
                )
                if self._closed or self._limit_reached():
                    return
                self._inflight = self._reservation
                self._record_peak()
            try:
                item = self._build()
            except BaseException as error:  # noqa: BLE001 - forward producer failure
                with self._condition:
                    self._inflight = 0
                    # Tracebacks can retain half-built tensor batches after a
                    # failed read. Keep the exception, release those frames.
                    self._failure = error.with_traceback(None)
                    self._condition.notify_all()
                return
            with self._condition:
                self._inflight = 0
                self._generated += 1
                if self._closed:
                    self._metrics["discarded_batches"] += 1
                    return
                self._queue.append(item)
                self._queued_bytes += item[2]
                self._record_peak()
                self._condition.notify_all()
            # Queue ownership must be the only producer reference between reads.
            del item

    def next_batch(self):
        started = time.perf_counter()
        with self._condition:
            if self._closed:
                raise RuntimeError("pipeline is closed")
            if not self.depth:
                if self._failure is not None:
                    raise self._failure
                if self._limit_reached():
                    raise StopIteration
                self._inflight = self._reservation
                self._record_peak()
                try:
                    item = self._build()
                except BaseException as error:  # noqa: BLE001 - release tensor frames
                    self._failure = error.with_traceback(None)
                    raise self._failure
                finally:
                    self._inflight = 0
                self._generated += 1
            else:
                if self._thread is None:
                    self._thread = threading.Thread(
                        target=self._produce, name="throughput-prefetch", daemon=True
                    )
                    self._thread.start()
                self._condition.wait_for(
                    lambda: (
                        self._queue
                        or self._failure is not None
                        or self._closed
                        or self._limit_reached()
                    )
                )
                if self._closed:
                    raise RuntimeError("pipeline is closed")
                if not self._queue:
                    if self._failure is not None:
                        raise self._failure
                    raise StopIteration
                item = self._queue.popleft()
                self._queued_bytes -= item[2]
            self._committed = item[1]
            self._consumed += 1
            self._metrics["stall_seconds"].append(time.perf_counter() - started)
            self._metrics["queue_delay_seconds"].append(time.perf_counter() - item[3])
            self._condition.notify_all()
            return item[0]

    def close(self):
        started = time.perf_counter()
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join()
        with self._condition:
            self._metrics["discarded_batches"] += len(self._queue)
            self._queue.clear()
            self._queued_bytes = self._inflight = 0
            self._metrics["close_seconds"] += time.perf_counter() - started

    def load_state_dict(self, state):
        self.close()
        # The maintained loader validates identity, permutation and counters.
        # Restore the prior checkpoint if validation fails partway through.
        try:
            self._sampler.load_state_dict(copy.deepcopy(state))
        except BaseException:
            self._sampler.load_state_dict(_public(self._committed))
            raise
        with self._condition:
            self._committed = _capture(self._sampler)
            self._generated = self._consumed = 0
            self._failure = self._thread = None
            self._closed = False


def create_pipeline(dataset, config, *, depth=0, max_bytes=MAX_BYTES, max_batches=None):
    return Pipeline(
        dataset, config, depth=depth, max_bytes=max_bytes, max_batches=max_batches
    )
