"""Frozen plan for forking a new candidate run from a finished run's checkpoint.

A fork is a new run with its own identity, output and journals. It restores the
source model, optimizer moments, EMA, sampler and RNG state exactly, keeps the
global step and sampler counters, and appends a cosine learning-rate tail that
starts at the saved final rates. The source run and its history are only read.
"""

import math
from pathlib import Path

from pydantic import Field, model_validator

from tennis3d.artifacts import file_sha256, read_json
from tennis3d.schemas import StrictModel

TAIL_KIND = "cosine-appended-tail-v1"
PLANNED_KIND = "cosine-planned-warmup-v1"
TAIL_MIN_RATIO = 0.05


class ContinuationSource(StrictModel):
    checkpoint: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    step: int = Field(ge=1)
    # Baseline binding: the source run's own selected score, never a candidate record.
    best_score: float | None


class ContinuationPlan(StrictModel):
    schema_version: str = Field(default="1.0", pattern=r"^1\.0$")
    kind: str = Field(default=TAIL_KIND, pattern=f"^{TAIL_KIND}$")
    # Fixed before the first allocation so a restarted fork keeps one identity.
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    source: ContinuationSource
    additional_steps: int = Field(ge=1)
    min_ratio: float = TAIL_MIN_RATIO
    # Saved final rate of every optimizer group, in group order.
    start_learning_rates: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_recipe(self):
        if self.run_id == self.source.run_id:
            raise ValueError("a fork needs a run identity distinct from its source")
        if self.min_ratio != TAIL_MIN_RATIO:
            raise ValueError("the appended tail decays to 5% of its starting rate")
        if any(not math.isfinite(r) or r <= 0 for r in self.start_learning_rates):
            raise ValueError("starting learning rates must be finite and positive")
        return self


def tail_policy(start_lr, additional_steps, offset):
    """Appended tail for one group; step stays the global update counter."""
    return {
        "kind": TAIL_KIND,
        "peak_lr": start_lr,
        "total_steps": additional_steps,
        "min_ratio": TAIL_MIN_RATIO,
        "offset": offset,
        "step": offset,
    }


def tail_rate(policy):
    """Rate for the next update: cosine from the start rate to 5% of it."""
    if (
        type(policy["offset"]) is not int
        or policy["offset"] < 1
        or policy["step"] < policy["offset"]
        or policy["min_ratio"] != TAIL_MIN_RATIO
    ):
        raise ValueError("invalid appended learning-rate tail")
    completed = policy["step"] + 1 - policy["offset"]
    progress = min(completed, policy["total_steps"]) / policy["total_steps"]
    factor = TAIL_MIN_RATIO + (1.0 - TAIL_MIN_RATIO) * (
        (1.0 + math.cos(math.pi * progress)) / 2.0
    )
    return policy["peak_lr"] * factor


def load_plan(path):
    """Return the validated plan and the hash of its exact frozen bytes."""
    path = Path(path)
    return ContinuationPlan.model_validate(read_json(path)), file_sha256(path)


def bind_source(plan: ContinuationPlan, plan_sha256, state):
    """Check the source checkpoint against the plan and describe the lineage.

    The caller has read `state` from plan.source.checkpoint. Only a run whose
    planned schedule is exhausted can take an appended tail, so a fork can never
    restart at the peak rate or re-enter warmup.
    """
    source = plan.source
    if file_sha256(source.checkpoint) != source.sha256:
        raise ValueError("fork source checkpoint differs from the frozen plan")
    if "continuation" in state:
        raise ValueError("fork source is itself a continuation; resume it instead")
    if (
        state["run_id"] != source.run_id
        or state["step"] != source.step
        or state["best_score"] != source.best_score
    ):
        raise ValueError("fork source run, step or selected score differs from plan")
    groups = state["optimizer"]["param_groups"]
    policies = [group.get("lr_schedule") for group in groups]
    if any(
        not isinstance(policy, dict)
        or policy.get("kind") != PLANNED_KIND
        or policy["step"] != policy["total_steps"]
        or policy["step"] != source.step
        for policy in policies
    ):
        raise ValueError("fork source has not finished its planned learning rates")
    # A parameter without saved moments would silently restart from zero.
    moments = state["optimizer"]["state"]
    if len(moments) != sum(len(group["params"]) for group in groups) or any(
        "momentum_buffer" not in moment and "exp_avg" not in moment
        for moment in moments.values()
    ):
        raise ValueError("fork source lacks complete optimizer moments")
    rates = [group["lr"] for group in groups]
    if rates != plan.start_learning_rates:
        raise ValueError("fork source final learning rates differ from the plan")
    sampler = state["sampler"]
    return {
        "schema_version": "1.0",
        "kind": plan.kind,
        "plan_sha256": plan_sha256,
        "additional_steps": plan.additional_steps,
        "final_step": source.step + plan.additional_steps,
        "tail": {
            "min_ratio": plan.min_ratio,
            "offset": source.step,
            "start_learning_rates": rates,
            "final_learning_rates": [rate * plan.min_ratio for rate in rates],
        },
        "source": {
            "run_id": source.run_id,
            "checkpoint_sha256": source.sha256,
            "step": source.step,
            "best_score": state["best_score"],
            "best_step": state.get("best_step"),
            "budget": dict(state["budget"]),
            "source_sha256": state.get("provenance", {}).get("source_sha256"),
            "learning_rate_policies": [dict(policy) for policy in policies],
            "sampler": {
                name: sampler[name] for name in ("draws", "rejected", "epoch", "cursor")
            },
        },
    }


def added_exposure(continuation, *, step, batch_size, sampler):
    """Exposure added by the candidate, separate from the global counters."""
    source = continuation["source"]
    added = step - source["step"]
    return {
        "source_run_id": source["run_id"],
        "source_step": source["step"],
        "additional_steps": added,
        "additional_windows": added * batch_size,
        "additional_sampled_views": sampler.draws - source["sampler"]["draws"],
        "additional_rejected_crops": sampler.rejected - source["sampler"]["rejected"],
    }
