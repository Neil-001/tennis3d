"""Frozen accounting around the editable trainer's standard optimizer/data hooks."""

import hashlib
import time
from contextlib import ExitStack
from unittest.mock import patch

import torch
from torch.optim.optimizer import (
    register_optimizer_step_post_hook,
    register_optimizer_step_pre_hook,
)

from tennis3d.lifting.runtime import synchronize


def tensor_digest(mapping) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(mapping.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class Observer:
    """Require fresh initialization and charge every completed optimization step.

    This checks reviewed project code, not hostile Python. The evaluator runs in
    a separate interpreter and source hashes are checked before/after execution.
    """

    def __init__(self, config, control):
        self.config, self.control = config, control
        self.active = None
        self.seconds = 0.0
        self.steps = 0
        self.pending = 0
        self.batches = 0
        self.initial = None
        self.last_duration = 0.0
        self.optimizer = None
        self.expected_initial = None

    def install(self):
        from tennis3d.lifting import checkpoint
        from tennis3d.lifting.data import WindowSampler
        from tennis3d.lifting.model import Lifter

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            self.expected_initial = tensor_digest(
                Lifter(self.config.model).state_dict()
            )
        next_batch = WindowSampler.next_batch
        save_checkpoint = checkpoint.save_checkpoint

        def batch(sampler):
            if self.active is None:
                if self.seconds >= self.config.max_train_seconds:
                    raise ValueError(
                        "candidate requested training beyond the fixed budget"
                    )
                synchronize(self.config.device)
                self.active = time.perf_counter()
            self.batches += 1
            return next_batch(sampler)

        def save(path, state):
            if self.initial is None:
                if (
                    state["step"] != 0
                    or state["optimizer"]["state"]
                    or int(state["ema"]["n_averaged"]) != 1
                ):
                    raise ValueError(
                        "trial must begin with fresh optimizer and EMA state"
                    )
                self.initial = tensor_digest(state["model"])
                if self.initial != self.expected_initial:
                    raise ValueError(
                        "trial did not start at its seeded random model weights"
                    )
                ema = {
                    k.removeprefix("module."): v
                    for k, v in state["ema"].items()
                    if k.startswith("module.")
                }
                if tensor_digest(ema) != self.initial:
                    raise ValueError("EMA must start as a fresh model copy")
            save_checkpoint(path, state)

        self.patches = ExitStack()
        self.patches.enter_context(patch.object(WindowSampler, "next_batch", batch))
        self.patches.enter_context(patch.object(checkpoint, "save_checkpoint", save))
        self.handles = [
            register_optimizer_step_pre_hook(self.pre),
            register_optimizer_step_post_hook(self.post),
        ]

    def pre(self, optimizer, args, kwargs):
        if self.active is None or self.initial is None or self.pending:
            raise ValueError("uncounted optimizer update outside a training step")
        if self.optimizer is None:
            if optimizer.state:
                raise ValueError("optimizer was not initialized fresh")
            self.optimizer = optimizer
        elif optimizer is not self.optimizer:
            raise ValueError("trial cannot switch to an uncounted optimizer")

    def post(self, optimizer, args, kwargs):
        self.pending += 1

    def update(self, row):
        if row.get("phase") == "training" and "loss" in row:
            if (
                self.active is None
                or self.pending != 1
                or row["step"] != self.steps + 1
            ):
                raise ValueError(
                    "training callback disagrees with observed optimizer steps"
                )
            synchronize(self.config.device)
            self.last_duration = time.perf_counter() - self.active
            self.seconds += self.last_duration
            self.active, self.pending = None, 0
            self.steps += 1
            if self.seconds >= self.config.max_train_seconds:
                self.control.stop_requested = True

    def finish(self, summary):
        if (
            self.active is not None
            or self.pending
            or not self.steps
            or summary["step"] != self.steps
        ):
            raise ValueError("incomplete or uncounted training updates")
        if self.initial is None or self.seconds < self.config.max_train_seconds:
            raise ValueError("candidate stopped before the common training budget")
        return {
            "steps": self.steps,
            "batches": self.batches,
            "train_seconds": self.seconds,
            "train_overrun_seconds": self.seconds - self.config.max_train_seconds,
            "last_step_seconds": self.last_duration,
            "initial_weights_sha256": self.initial,
        }

    def close(self):
        self.patches.close()
        for handle in self.handles:
            handle.remove()
