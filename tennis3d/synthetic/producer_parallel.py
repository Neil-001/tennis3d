"""Spawned workers, bounded ordered delivery, and deadline-aware draining."""

import multiprocessing
import signal
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError


class Stopped(RuntimeError):
    pass


class StopController:
    def __init__(self, drain_seconds: float):
        self.requested = False
        self.deadline = float("inf")
        self.drain_seconds = drain_seconds
        self._previous = {}

    def request(self, *_):
        if not self.requested:
            self.requested = True
            self.deadline = time.monotonic() + self.drain_seconds

    def check(self):
        if self.requested:
            raise Stopped("generation stopped before complete planned output")

    def __enter__(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._previous[sig] = signal.signal(sig, self.request)
        return self

    def __exit__(self, *_):
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)


def _initialize():
    # The coordinator receives the process-group signal and owns the drain.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class OrderedWorkers:
    def __init__(self, workers: int, inflight: int, stop: StopController):
        self.stop, self.inflight = stop, inflight
        self.executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize,
        )

    def map(self, function, tasks):
        iterator, pending = iter(tasks), []
        exhausted = False
        try:
            while pending or not exhausted:
                while (
                    not exhausted
                    and len(pending) < self.inflight
                    and not self.stop.requested
                ):
                    try:
                        task = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.append(self.executor.submit(function, task))
                self.stop.check()
                if not pending:
                    break
                while True:
                    self.stop.check()
                    try:
                        result = pending[0].result(timeout=0.1)
                        break
                    except TimeoutError:
                        if pending[0].done():
                            raise
                        continue
                pending.pop(0)
                # Includes running, queued and completed-but-not-yielded tasks.
                yield result
        finally:
            for future in pending:
                future.cancel()

    def close(self):
        processes = list(self.executor._processes.values())
        self.executor.shutdown(wait=False, cancel_futures=True)
        while any(process.is_alive() for process in processes):
            if self.stop.requested and time.monotonic() >= self.stop.deadline:
                for process in processes:
                    if process.is_alive():
                        process.kill()
                break
            for process in processes:
                process.join(timeout=0.05)
        for process in processes:
            # The executor manager also joins workers. A concurrent join can
            # return before that thread publishes the exit status on this object.
            while process.is_alive():
                process.join(timeout=0.05)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is not None:
            self.stop.request()
        self.close()
