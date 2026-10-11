"""Reusable finite-budget optimizer/EMA training; no experiment search policy."""

import json
import math
import random
import shutil
import time
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim._functional import adam as functional_adam
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from tennis3d.artifacts import file_sha256, read_json, write_json
from tennis3d.config import record_config
from tennis3d.evaluation.synthetic import evaluate_model, preflight
from tennis3d.lifting.checkpoint import (
    CHECKPOINT_VERSION,
    carry_best_checkpoint,
    random_state,
    read_checkpoint,
    restore_random_state,
    save_checkpoint,
)
from tennis3d.lifting.continuation import (
    TAIL_KIND,
    added_exposure,
    bind_source,
    load_plan,
    tail_policy,
    tail_rate,
)
from tennis3d.lifting.data import ViewDataset, WindowSampler, validate_split_pair
from tennis3d.lifting.features import WORLD_SCALE_M
from tennis3d.lifting.losses import position_loss, vertical_radial_position_loss
from tennis3d.lifting.model import Lifter, build_lifter
from tennis3d.lifting.runtime import environment, resolve_device, synchronize
from tennis3d.lifting.settings import TrainConfig
from tennis3d.provenance import code_identity


def _muon_orthogonalize(update, group, *, dtype: torch.dtype):
    """Normalize in FP32 before the configured Newton-Schulz matrix precision."""
    update = update.float()
    transpose = update.shape[0] > update.shape[1]
    if transpose:
        update = update.T
    update = update / update.norm().clamp(min=group["eps"])
    update = update.to(dtype=dtype)
    a, b, c = group["ns_coefficients"]
    for _ in range(group["ns_steps"]):
        gram = update @ update.T
        polynomial = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        update = torch.addmm(update, polynomial, update, beta=a)
    if transpose:
        update = update.T
    return update.float()


def _muon_brake(direction, parameter, group):
    """Damp outward row components of W -= rate * direction."""
    brake = group.get("outward_radial_brake", 0.0)
    if brake == 0:
        return direction
    if (
        isinstance(brake, bool)
        or not isinstance(brake, (int, float))
        or not 0 < brake <= 1
    ):
        raise ValueError("invalid outward radial brake")
    radius_squared = parameter.square().sum(dim=-1, keepdim=True)
    coefficient = (parameter * direction).sum(
        dim=-1, keepdim=True
    ) / radius_squared.clamp_min(group["eps"])
    return direction - brake * coefficient.clamp_max(0.0) * parameter


def _muon_step(group, state):
    """One counted Muon step retains FP32 weights and momentum on both devices."""
    if group.get("batched_muon", False):
        return _muon_batched_step(group, state)
    precision = group.get("ns_precision", "fp32")
    if precision not in ("fp32", "cuda-fp16"):
        raise ValueError("unknown Muon Newton-Schulz precision")
    split_indices = (
        set(group["qkv_indices"])
        if group["algorithm"] in ("muon-fp16-split-qkv", "muon-fp16-split-qkv-gates")
        else set()
    )
    gate_indices = (
        set(group["gate_indices"])
        if group["algorithm"] == "muon-fp16-split-qkv-gates"
        else set()
    )
    for index, parameter in enumerate(group["params"]):
        gradient = parameter.grad
        if gradient is None:
            continue
        if gradient.is_sparse or gradient.ndim != 2:
            raise ValueError("Muon requires dense matrix gradients")
        parameter_state = state[parameter]
        if "momentum_buffer" not in parameter_state:
            parameter_state["momentum_buffer"] = torch.zeros_like(gradient)
        buffer = parameter_state["momentum_buffer"]
        buffer.lerp_(gradient, 1.0 - group["momentum"])
        update = (
            gradient.lerp(buffer, group["momentum"]) if group["nesterov"] else buffer
        )
        dtype = (
            torch.float16
            if precision == "cuda-fp16" and parameter.is_cuda
            else torch.float32
        )
        if index in split_indices or index in gate_indices:
            pieces = 3 if index in split_indices else 2
            if index in split_indices and parameter.shape[0] != 3 * parameter.shape[1]:
                raise ValueError("split QKV requires three packed square matrices")
            if index in gate_indices and parameter.shape[0] % 2:
                raise ValueError("split SwiGLU requires two equal row slices")
            if group["adjust_lr_fn"] != "match_rms_adamw":
                raise ValueError("split Muon requires RMS-matching LR scaling")
            # Preserve the packed forward and momentum storage, but give each
            # projection its own orthogonalization and RMS matching.
            for target, part in zip(
                parameter.chunk(pieces, 0), update.chunk(pieces, 0)
            ):
                direction = _muon_orthogonalize(part, group, dtype=dtype)
                direction = _muon_brake(direction, target, group)
                rate = group["lr"] * 0.2 * math.sqrt(max(target.shape))
                target.mul_(1.0 - group["lr"] * group["weight_decay"])
                target.add_(direction, alpha=-rate)
            continue
        update = _muon_orthogonalize(update, group, dtype=dtype)
        update = _muon_brake(update, parameter, group)
        if group["adjust_lr_fn"] != "match_rms_adamw":
            raise ValueError("FP32 Muon requires RMS-matching LR scaling")
        rate = group["lr"] * 0.2 * math.sqrt(max(parameter.shape))
        parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
        parameter.add_(update, alpha=-rate)


def _muon_batched_orthogonalize(update, group, *, dtype: torch.dtype):
    """The same polynomial, with an independent norm for each stacked matrix."""
    update = update.float()
    transpose = update.shape[-2] > update.shape[-1]
    if transpose:
        update = update.transpose(-2, -1)
    norms = torch.linalg.vector_norm(update, dim=(-2, -1), keepdim=True)
    update = (update / norms.clamp_min(group["eps"])).to(dtype=dtype)
    a, b, c = group["ns_coefficients"]
    for _ in range(group["ns_steps"]):
        gram = update @ update.transpose(-2, -1)
        polynomial = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        update = torch.baddbmm(update, polynomial, update, beta=a)
    if transpose:
        update = update.transpose(-2, -1)
    return update.float()


def _muon_batched_step(group, state):
    """Batch equal-shaped projections without mixing their norms or momentum."""
    if (
        group["algorithm"] != "muon-fp16-split-qkv-gates"
        or group["ns_precision"] != "cuda-fp16"
        or group["adjust_lr_fn"] != "match_rms_adamw"
    ):
        raise ValueError("batched Muon requires the split QKV/gate precision policy")
    qkv, gates = set(group["qkv_indices"]), set(group["gate_indices"])
    buckets = {}
    for index, parameter in enumerate(group["params"]):
        gradient = parameter.grad
        if gradient is None:
            continue
        if gradient.is_sparse or gradient.ndim != 2:
            raise ValueError("Muon requires dense matrix gradients")
        parameter_state = state[parameter]
        if "momentum_buffer" not in parameter_state:
            parameter_state["momentum_buffer"] = torch.zeros_like(gradient)
        buffer = parameter_state["momentum_buffer"]
        buffer.lerp_(gradient, 1.0 - group["momentum"])
        update = (
            gradient.lerp(buffer, group["momentum"]) if group["nesterov"] else buffer
        )
        pieces = 3 if index in qkv else 2 if index in gates else 1
        if index in qkv and parameter.shape[0] != 3 * parameter.shape[1]:
            raise ValueError("split QKV requires three packed square matrices")
        if index in gates and parameter.shape[0] % 2:
            raise ValueError("split SwiGLU requires two equal row slices")
        dtype = torch.float16 if parameter.is_cuda else torch.float32
        for target, part in zip(parameter.chunk(pieces, 0), update.chunk(pieces, 0)):
            key = (target.device, dtype, tuple(target.shape))
            buckets.setdefault(key, []).append((target, part))
    for (_, dtype, _), entries in buckets.items():
        targets, parts = zip(*entries)
        directions = _muon_batched_orthogonalize(torch.stack(parts), group, dtype=dtype)
        directions = _muon_brake(directions, torch.stack(targets), group)
        rate = group["lr"] * 0.2 * math.sqrt(max(targets[0].shape))
        for target, direction in zip(targets, directions.unbind(0)):
            target.mul_(1.0 - group["lr"] * group["weight_decay"])
            target.add_(direction, alpha=-rate)


def _adam_step(group, state):
    parameters, gradients, averages, squares, maxima, steps = [], [], [], [], [], []
    for parameter in group["params"]:
        gradient = parameter.grad
        if gradient is None:
            continue
        if gradient.is_sparse:
            raise ValueError("Adam requires dense gradients")
        parameter_state = state[parameter]
        if not parameter_state:
            parameter_state["step"] = torch.zeros((), dtype=torch.float32, device="cpu")
            parameter_state["exp_avg"] = torch.zeros_like(parameter)
            parameter_state["exp_avg_sq"] = torch.zeros_like(parameter)
            if group["amsgrad"]:
                parameter_state["max_exp_avg_sq"] = torch.zeros_like(parameter)
        parameters.append(parameter)
        gradients.append(gradient)
        averages.append(parameter_state["exp_avg"])
        squares.append(parameter_state["exp_avg_sq"])
        steps.append(parameter_state["step"])
        if group["amsgrad"]:
            maxima.append(parameter_state["max_exp_avg_sq"])
    beta1, beta2 = group["betas"]
    functional_adam(
        parameters,
        gradients,
        averages,
        squares,
        maxima,
        steps,
        foreach=group["foreach"],
        capturable=False,
        differentiable=False,
        fused=None,
        has_complex=any(p.is_complex() for p in parameters),
        decoupled_weight_decay=False,
        amsgrad=group["amsgrad"],
        beta1=beta1,
        beta2=beta2,
        lr=group["lr"],
        weight_decay=group["weight_decay"],
        eps=group["eps"],
        maximize=False,
    )


def _cosine_policy(peak_lr, total_steps=2000):
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError("planned update count must be a positive integer")
    return {
        "kind": "cosine-planned-warmup-v1",
        "peak_lr": peak_lr,
        "total_steps": total_steps,
        "warmup_steps": min(total_steps - 1, (total_steps + 49) // 50),
        "min_ratio": 0.05,
        "step": 0,
    }


def _cosine_rate(policy):
    """Rate for the next update, with a persistent per-group step counter."""
    if not isinstance(policy, dict) or not policy:
        raise ValueError("invalid cosine learning-rate policy")
    planned = policy.get("kind") == "cosine-planned-warmup-v1"
    tail = policy.get("kind") == TAIL_KIND
    keys = {
        "kind",
        "peak_lr",
        "total_steps",
        "min_ratio",
        "step",
    }
    if planned:
        keys.add("warmup_steps")
    if tail:
        keys.add("offset")
    if set(policy) != keys:
        raise ValueError("invalid cosine learning-rate policy")
    if (
        type(policy["total_steps"]) is not int
        or policy["total_steps"] < 1
        or type(policy["step"]) is not int
        or policy["step"] < 0
        or isinstance(policy["peak_lr"], bool)
        or not isinstance(policy["peak_lr"], (float, int))
        or not math.isfinite(policy["peak_lr"])
        or policy["peak_lr"] <= 0
    ):
        raise ValueError("invalid cosine learning-rate policy")
    if tail:
        return tail_rate(policy)
    if planned:
        horizon = policy["total_steps"]
        warmup = min(horizon - 1, (horizon + 49) // 50)
        if (
            type(policy["warmup_steps"]) is not int
            or policy["warmup_steps"] != warmup
            or policy["min_ratio"] != 0.05
        ):
            raise ValueError("invalid planned learning-rate policy")
        completed = policy["step"] + 1
        if completed <= warmup:
            return policy["peak_lr"] * completed / warmup
        if horizon == 1 and completed == 1:
            return policy["peak_lr"]
        progress = min(completed - warmup, horizon - warmup) / (horizon - warmup)
    else:
        if (policy["kind"], policy["min_ratio"]) not in (
            ("cosine-2000-floor005-v1", 0.05),
            ("cosine-2000-v1", 0.2),
        ) or policy["total_steps"] != 2000:
            raise ValueError("invalid cosine learning-rate policy")
        progress = min(policy["step"], policy["total_steps"]) / policy["total_steps"]
    factor = (
        policy["min_ratio"]
        + (1.0 - policy["min_ratio"]) * (1.0 + math.cos(math.pi * progress)) / 2.0
    )
    return policy["peak_lr"] * factor


class MuonAdam(torch.optim.Optimizer):
    """One observed step updates disjoint Muon matrices and Adam auxiliaries."""

    def __init__(self, model: Lifter, lr: float):
        hidden = [p for p in model.blocks.parameters() if p.ndim == 2]
        hidden_ids = {id(p) for p in hidden}
        auxiliary = [p for p in model.parameters() if id(p) not in hidden_ids]
        qkv_ids = {
            id(p)
            for name, p in model.named_parameters()
            if name.startswith("blocks.") and name.endswith(".attention.qkv.weight")
        }
        self._qkv_indices = tuple(
            i for i, parameter in enumerate(hidden) if id(parameter) in qkv_ids
        )
        if len(self._qkv_indices) != model.config.depth:
            raise ValueError("each attention block must have one packed QKV matrix")
        gate_ids = {
            id(p)
            for name, p in model.named_parameters()
            if model.config.gated_ffn
            and name.startswith("blocks.")
            and name.endswith(".ffn.0.weight")
        }
        self._gate_indices = tuple(
            i for i, parameter in enumerate(hidden) if id(parameter) in gate_ids
        )
        if len(self._gate_indices) != (
            model.config.depth if model.config.gated_ffn else 0
        ):
            raise ValueError("each SwiGLU block must have one packed gate/value matrix")
        super().__init__(
            [
                {
                    "params": hidden,
                    "algorithm": "muon-fp16-split-qkv-gates",
                    "ns_precision": "cuda-fp16",
                    "qkv_indices": list(self._qkv_indices),
                    "gate_indices": list(self._gate_indices),
                    "lr": 1.5 * lr,
                    "lr_schedule": _cosine_policy(1.5 * lr),
                    "momentum": 0.98,
                    "nesterov": True,
                    "ns_coefficients": (3.4445, -4.775, 2.0315),
                    "ns_steps": 5,
                    "eps": 1e-7,
                    "adjust_lr_fn": "match_rms_adamw",
                },
                {
                    "params": auxiliary,
                    "algorithm": "adam",
                    "lr_schedule": _cosine_policy(lr),
                    "betas": (0.9, 0.98),
                    "eps": 1e-8,
                    "amsgrad": False,
                    "foreach": None,
                },
            ],
            {"lr": lr, "weight_decay": 0.0},
        )

        self._outward_radial_brake = model.config.outward_radial_brake
        if self._outward_radial_brake:
            self.param_groups[0]["outward_radial_brake"] = self._outward_radial_brake
        self._batched_muon = model.config.batched_muon
        if self._batched_muon:
            self.param_groups[0]["batched_muon"] = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            policy = group.get("lr_schedule")
            if policy is not None:
                group["lr"] = _cosine_rate(policy)
            if group["algorithm"] in (
                "muon-fp32",
                "muon-fp16-ns",
                "muon-fp16-split-qkv",
                "muon-fp16-split-qkv-gates",
            ):
                _muon_step(group, self.state)
            elif group["algorithm"] == "adam":
                _adam_step(group, self.state)
            else:
                raise ValueError("unknown optimizer parameter group")
            if policy is not None:
                policy["step"] += 1
        return loss

    def load_state_dict(self, state_dict):
        algorithms = [g.get("algorithm") for g in state_dict["param_groups"]]
        if algorithms not in (
            ["muon-fp32", "adam"],
            ["muon-fp16-ns", "adam"],
            ["muon-fp16-split-qkv", "adam"],
            ["muon-fp16-split-qkv-gates", "adam"],
        ):
            raise ValueError("incompatible Muon/Adam parameter groups")
        if (
            algorithms[0]
            in ("muon-fp16-ns", "muon-fp16-split-qkv", "muon-fp16-split-qkv-gates")
            and state_dict["param_groups"][0].get("ns_precision") != "cuda-fp16"
        ):
            raise ValueError("FP16 Muon checkpoint is missing its precision policy")
        batched = state_dict["param_groups"][0].get("batched_muon", False)
        if (
            type(batched) is not bool
            or batched != self._batched_muon
            or (batched and algorithms[0] != "muon-fp16-split-qkv-gates")
            or "batched_muon" in state_dict["param_groups"][1]
        ):
            raise ValueError("Muon checkpoint has incompatible batching policy")
        if algorithms[0] in (
            "muon-fp16-split-qkv",
            "muon-fp16-split-qkv-gates",
        ) and state_dict["param_groups"][0].get("qkv_indices") != list(
            self._qkv_indices
        ):
            raise ValueError("split Muon checkpoint has incompatible QKV roles")
        if algorithms[0] == "muon-fp16-split-qkv-gates" and state_dict["param_groups"][
            0
        ].get("gate_indices") != list(self._gate_indices):
            raise ValueError("split Muon checkpoint has incompatible gate/value roles")
        brake = state_dict["param_groups"][0].get("outward_radial_brake", 0.0)
        if (
            isinstance(brake, bool)
            or not isinstance(brake, (int, float))
            or not math.isfinite(brake)
            or not 0 <= brake <= 1
            or brake != self._outward_radial_brake
        ):
            raise ValueError("Muon checkpoint has incompatible radial braking")
        policies = [g.get("lr_schedule") for g in state_dict["param_groups"]]
        if any(policy is not None for policy in policies):
            if any(policy is None for policy in policies):
                raise ValueError("inconsistent optimizer learning-rate policies")
            for policy in policies:
                _cosine_rate(policy)
            if len({policy["step"] for policy in policies}) != 1:
                raise ValueError("optimizer learning-rate counters differ")
            if (
                len(
                    {
                        (
                            policy["kind"],
                            policy["total_steps"],
                            policy["min_ratio"],
                            policy.get("warmup_steps"),
                            policy.get("offset"),
                        )
                        for policy in policies
                    }
                )
                != 1
            ):
                raise ValueError("optimizer learning-rate horizons differ")
        # Old groups without a policy retain their saved constant rates.
        super().load_state_dict(state_dict)


class ScheduledAdamW(torch.optim.AdamW):
    """AdamW with a checkpointed planned cosine counter, one count per update."""

    def __init__(
        self, model: Lifter, lr: float, *, weight_decay=0.01, betas=(0.9, 0.999)
    ):
        super().__init__(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
        )
        for group in self.param_groups:
            group["algorithm"] = "adamw"
            group["lr_schedule"] = _cosine_policy(lr)

    def step(self, closure=None):
        for group in self.param_groups:
            group["lr"] = _cosine_rate(group["lr_schedule"])
        result = super().step(closure)
        for group in self.param_groups:
            group["lr_schedule"]["step"] += 1
        return result

    def load_state_dict(self, state_dict):
        groups = state_dict["param_groups"]
        if len(groups) != 1 or groups[0].get("algorithm") != "adamw":
            raise ValueError("incompatible AdamW parameter groups")
        _cosine_rate(groups[0].get("lr_schedule"))
        if (
            groups[0]["betas"] != self.param_groups[0]["betas"]
            or groups[0]["weight_decay"] != self.param_groups[0]["weight_decay"]
        ):
            raise ValueError("AdamW checkpoint has incompatible recipe")
        super().load_state_dict(state_dict)


def build_optimizer(
    model: Lifter,
    lr: float,
    saved_state=None,
    *,
    kind="muon-adam",
    weight_decay=0.01,
    betas=(0.9, 0.999),
):
    if saved_state is not None:
        algorithms = [g.get("algorithm") for g in saved_state["param_groups"]]
        if (kind == "adamw") != (algorithms == ["adamw"]):
            raise ValueError("saved optimizer differs from configured optimizer")
        if algorithms == ["adamw"]:
            return ScheduledAdamW(model, lr, weight_decay=weight_decay, betas=betas)
        # Existing Adam checkpoints keep their original update rule on resume.
        if not any(a is not None and a.startswith("muon-") for a in algorithms):
            if any(a not in (None, "adam") for a in algorithms):
                raise ValueError("unknown saved optimizer algorithm")
            return torch.optim.Adam(model.parameters(), lr=lr)
        return MuonAdam(model, lr)
    if kind == "adamw":
        return ScheduledAdamW(model, lr, weight_decay=weight_decay, betas=betas)
    if kind != "muon-adam":
        raise ValueError("unknown fresh optimizer")
    return MuonAdam(model, lr)


def _training_objective(state):
    """Use explicit metadata or verified historical source, never optimizer type."""
    if state is None:
        return "xyz-weighted-radial-v1"
    if "training_objective" in state:
        objective = state["training_objective"]
        if objective not in ("xyz-squared-v1", "xyz-weighted-radial-v1"):
            raise ValueError("unsupported checkpoint training objective")
        return objective
    sources = state.get("provenance", {}).get("source_files", {})
    fingerprint = (
        sources.get("tennis3d/lifting/train.py"),
        sources.get("tennis3d/lifting/losses.py"),
        sources.get("tennis3d/lifting/model.py"),
    )
    # Exact source bytes from main 7f664cb and the selected 6352fe2 recipe.
    # A revision alone can describe dirty experimental code, so it is insufficient.
    historical = {
        (
            "8f22eaad83de14813f43be55961eb345e1c777a2b9373b5957211db2087eff54",
            "82d4ab351ae28ad33021887d85c508ee086ce86cf7e2c3e833ed375ae0fae00d",
            "572e55eebcee2822e328ed0cbc8e9dc5d2cda606010947fce87770ef0ae26cb6",
        ): "xyz-squared-v1",
        (
            "6d8da83710ce1defe0c104c2b1fbe530d3feb95746230ce22e6c9a6d7b254342",
            "a5a4004548a25da81c45e924023a4604a0e35277bf12c220d06a53218a9a1a92",
            "9c90a9a7e4bd06f56c9edec4f06058f2c9b4a111624181856915844733ccf8cc",
        ): "xyz-weighted-radial-v1",
    }
    if fingerprint not in historical:
        raise ValueError(
            "checkpoint does not identify its training objective; "
            "cannot safely resume an ambiguous historical recipe"
        )
    return historical[fingerprint]


@torch.no_grad()
def optimizer_stability(optimizer):
    """Logging-only RMS of weights and first moments, in parameter-group order."""
    result = []
    for group in optimizer.param_groups:
        parameters = group["params"]
        moments = [
            moment
            for p in parameters
            if (moment := optimizer.state.get(p, {}).get("momentum_buffer")) is not None
            or (moment := optimizer.state.get(p, {}).get("exp_avg")) is not None
        ]

        def rms(tensors):
            if not tensors:
                return None
            squared = torch.stack([t.float().square().sum() for t in tensors]).sum()
            return float((squared / sum(t.numel() for t in tensors)).sqrt())

        result.append(
            {
                "algorithm": group.get("algorithm", "adam"),
                "weight_rms": rms(parameters),
                "first_moment_rms": rms(moments),
            }
        )
    return result


@dataclass
class TrainControl:
    """Optional caller limits for a fixed trial or graceful interruption.

    Limits can only shorten the configured budget. deadline is an absolute
    time.perf_counter deadline checked before every step. Callbacks get completed
    step diagnostics; candidate-independent hard process timeouts belong to the
    experiment search.
    """

    # Actual per-invocation wrapper work, charged before any new updates.
    setup_job_seconds: float = 0.0
    max_train_seconds: float | None = None
    stop_after_steps: int | None = None
    deadline: float | None = None
    stop_requested: bool = False
    use_step_limit: bool = True
    use_early_stopping: bool = True
    validate_during_training: bool = True
    evaluate_at_end: bool = True
    on_update: Callable[[dict], None] | None = None


def _prepare_validation_history(output, resume, run_id):
    """Carry the journal to a relocated resume without changing prior records.

    Recovery can replay updates newer than a checkpoint. Those validations append
    new events, even at the same step; invocation_id and resumed_from_step identify
    the attempt. A torn/invalid journal fails closed: preserve it separately and
    restore a complete copy before resuming, rather than silently dropping points.
    Legacy runs with no history start recording at their next validation.
    """
    target = output / "validation-history.jsonl"
    source = Path(resume).resolve().parent.parent / target.name if resume else target
    for existing in dict.fromkeys((target, source)):
        if not existing.exists():
            continue
        with existing.open(encoding="utf-8") as history:
            for line in history:
                try:
                    row = json.loads(line)
                    if not line.endswith("\n") or row["run_id"] != run_id:
                        raise ValueError("incomplete record or different run")
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(
                        f"invalid validation history {existing}; preserve it and "
                        "restore a complete copy before resuming"
                    ) from error
    if source != target and source.exists():
        if not target.exists():
            shutil.copyfile(source, target)
        else:
            original, relocated = source.read_bytes(), target.read_bytes()
            if original.startswith(relocated):
                # An interrupted relocation may have retained only a prefix.
                with target.open("ab") as history:
                    history.write(original[len(relocated) :])
            elif not relocated.startswith(original):
                raise ValueError(
                    "validation histories diverge; preserve both and resume into "
                    "a new output directory"
                )
    return target


def _unjournaled_best_record(best_path, history_path):
    """Check the journal against the selected best; return its lost row, if any.

    best.pt carries the exact history row of the validation that selected it
    and is published first, so a selection is never absent from the journal.
    Nothing is written here. Resume is refused when the record disagrees with
    its checkpoint, when the journal holds the same validation event twice with
    different content, or when the journal records a better score than the
    available best checkpoint (for example after an interrupted sync that
    copied the journal but not the checkpoint).
    """
    best = read_checkpoint(best_path) if best_path.exists() else None
    record = best.get("validation_record") if best else None
    if best is not None and record is not None:
        try:
            consistent = (
                record["run_id"] == best["run_id"]
                and record["step"] == best["step"]
                and record["modes"]["corrupted"]["val_xyz_m"] == best["best_score"]
                and isinstance(record["invocation_id"], str)
            )
        except (KeyError, TypeError):
            consistent = False
        if not consistent:
            raise ValueError("best checkpoint and its validation record disagree")
    events, scores = {}, []
    if history_path.exists():
        with history_path.open(encoding="utf-8") as history:
            for line in history:
                row = json.loads(line)
                event = (row.get("invocation_id"), row["step"])
                if events.setdefault(event, row) != row:
                    raise ValueError(
                        "validation history records one event twice with different "
                        "content; preserve it and restore a complete copy"
                    )
                score = row.get("modes", {}).get("corrupted", {}).get("val_xyz_m")
                if score is not None:
                    scores.append(score)
    lost = None
    if record is not None:
        journaled = events.get((record["invocation_id"], record["step"]))
        if journaled is None:
            lost = record
        elif journaled != record:
            raise ValueError(
                "validation history conflicts with the selected best "
                "checkpoint; preserve both and restore a complete copy"
            )
    if scores and best is None:
        raise ValueError("resume requires the original run's selected best.pt")
    if scores and best is not None and min(scores) < best["best_score"]:
        raise ValueError(
            "validation history records a better score than the available best "
            "checkpoint; preserve both and restore the matching checkpoint"
        )
    return lost


def train(
    config: TrainConfig,
    output,
    *,
    resume=None,
    fork=None,
    expected_plan=None,
    control: TrainControl | None = None,
) -> dict:
    """Train from random weights, restore a checkpoint, or fork a finished run.

    Returns paths and budget/step diagnostics. With evaluate_at_end, the final
    EMA is scored once unless the final step was already validated. `fork` is a
    frozen continuation plan (tennis3d.lifting.continuation): a new run in a new
    output that inherits the source checkpoint's complete state and global
    counters but none of its journals, budget or selection. On resume,
    `expected_plan` names the plan the resumed run must have been forked from.
    """
    if resume and fork:
        raise ValueError("resume and fork are mutually exclusive")
    if expected_plan and not resume:
        raise ValueError("a plan check applies to a resumed continuation only")
    with ExitStack() as datasets:
        return _train(
            config,
            output,
            resume=resume,
            fork=fork,
            expected_plan=expected_plan,
            control=control,
            datasets=datasets,
        )


def _validate_resume_data(state, config, datasets, *, fork=False):
    """Allow relocated sources, but bind every semantic input to the checkpoint.

    Old v1 checkpoints have only manifest hashes. Their original manifest bytes
    and the legacy sampler policy must still match; they cannot resume on v2.
    A fork may also restate the plateau stop rule and its artifact label, which
    change no update; later resumes of the fork must then keep them.
    """
    identities = {label: dataset.identity for label, dataset in datasets.items()}
    if "dataset_identity" in state:
        if state["dataset_identity"] != identities:
            raise ValueError("resume dataset identities differ from the checkpoint")
    elif any(dataset.format_version != 1 for dataset in datasets.values()) or state[
        "manifest_sha256"
    ] != {label: dataset.manifest_sha256 for label, dataset in datasets.items()}:
        raise ValueError("legacy resume requires the original v1 dataset manifests")
    mutable = {
        "max_steps",
        "max_train_seconds",
        "max_job_seconds",
        "device",
        "train_manifest",
        "val_manifest",
        "checkpoint_every",
    }
    if fork:
        mutable |= {"patience", "artifact_label"}
    # Normalization supplies defaults for fields added since a legacy checkpoint.
    previous = TrainConfig.model_validate(state["config"]).model_dump(mode="json")
    resolved = config.model_dump(mode="json")
    if {k: v for k, v in resolved.items() if k not in mutable} != {
        k: v for k, v in previous.items() if k not in mutable
    }:
        raise ValueError(
            "resume must preserve model, optimizer, data, sampling and evaluation config"
        )


def _train(config, output, *, resume, fork, expected_plan, control, datasets):
    started = time.perf_counter()
    control = control or TrainControl()
    if not math.isfinite(control.setup_job_seconds) or control.setup_job_seconds < 0:
        raise ValueError("setup job seconds must be finite and nonnegative")

    def update(phase, **values):
        if control.on_update:
            control.on_update(
                {
                    "phase": phase,
                    "elapsed_seconds": time.perf_counter() - started,
                    **values,
                }
            )

    update("setup")
    output = Path(output).resolve()
    state = read_checkpoint(resume) if resume else None
    continuation = state.get("continuation") if state else None
    if expected_plan and (
        continuation is None
        or load_plan(expected_plan)[1] != continuation["plan_sha256"]
    ):
        raise ValueError("resumed run was not forked from the expected plan")
    if fork:
        if output.exists() and any(output.iterdir()):
            raise ValueError(
                "fork requires a new empty output; resume the fork's own checkpoint"
            )
        plan, plan_sha256 = load_plan(fork)
        # The source is restored below exactly like a resume, under a new identity.
        inherited = read_checkpoint(plan.source.checkpoint)
        continuation = bind_source(plan, plan_sha256, inherited)
        run_id = plan.run_id
    else:
        inherited = state
        run_id = state["run_id"] if state else uuid.uuid4().hex
    if continuation and config.max_steps != continuation["final_step"]:
        raise ValueError("a continuation runs to its planned final step only")
    # Validation and recovery cadence count from where this run's updates began.
    origin = continuation["source"]["step"] if continuation else 0
    if output.exists() and any(output.iterdir()):
        identity = output / "run.json"
        if (
            not state
            or not identity.exists()
            or read_json(identity)["run_id"] != run_id
        ):
            raise ValueError(
                "output is nonempty and does not belong to this resumed run"
            )
        # The recorded lineage is never rewritten to match another checkpoint.
        if read_json(identity).get("continuation") != continuation:
            raise ValueError("output lineage differs from the resumed checkpoint")
        previous = output / "summary.json"
        if previous.exists() and read_json(previous)["step"] > state["step"]:
            raise ValueError("resume an older checkpoint into a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    history_path = _prepare_validation_history(output, resume, run_id)
    invocation_id = uuid.uuid4().hex
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    device = resolve_device(config.device)
    if control.max_train_seconds is not None and control.max_train_seconds <= 0:
        raise ValueError("external training limit must be positive")
    training_limit = min(
        config.max_train_seconds, control.max_train_seconds or float("inf")
    )
    train_data = datasets.enter_context(ViewDataset(config.train_manifest))
    val_data = datasets.enter_context(ViewDataset(config.val_manifest))
    validate_split_pair(train_data, val_data)
    sources = {"train": train_data, "val": val_data}
    manifest_hashes = {
        label: dataset.manifest_sha256 for label, dataset in sources.items()
    }
    dataset_identity = {label: dataset.identity for label, dataset in sources.items()}
    resolved = config.model_dump(mode="json")
    if inherited:
        _validate_resume_data(inherited, config, sources, fork=bool(fork))
    random.seed(config.seed)
    np.random.seed(config.seed % 2**32)
    torch.manual_seed(config.seed)
    model = build_lifter(config.model).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count >= 2_000_000:
        raise ValueError("initial lifter must have fewer than two million parameters")
    optimizer = build_optimizer(
        model,
        config.learning_rate,
        inherited["optimizer"] if inherited else None,
        kind=config.optimizer,
        weight_decay=config.adamw_weight_decay,
        betas=config.adamw_betas,
    )
    objective = _training_objective(inherited)
    legacy_squared = objective == "xyz-squared-v1"
    loss_fn = position_loss if legacy_squared else vertical_radial_position_loss
    loss_units = (
        "mean squared XYZ / (10m)^2"
        if legacy_squared
        else "0.1 * smooth weighted XYZ / 10m, Z gain 1.5, weighted-radius smoothing 0.1m"
    )
    if inherited is None:
        # An explicit plan survives the fixed-trial max_steps sentinel.
        # Resume always uses the saved policy, including legacy fixed horizons.
        for group in optimizer.param_groups:
            policy = group.get("lr_schedule")
            if policy is not None and policy["kind"] == "cosine-planned-warmup-v1":
                group["lr_schedule"] = _cosine_policy(
                    group["lr"], config.model.schedule_steps or config.max_steps
                )
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(config.ema_decay))
    ema.update_parameters(model)
    sampler = WindowSampler(train_data, config)
    step, training_seconds, job_seconds, best_score, stale = 0, 0.0, 0.0, None, 0
    best_step, last_validation_step = None, -1
    if inherited:
        model.load_state_dict(inherited["model"])
        ema.load_state_dict(inherited["ema"])
        optimizer.load_state_dict(inherited["optimizer"])
        sampler.load_state_dict(inherited["sampler"])
        restore_random_state(inherited["rng"], device)
        step = inherited["step"]
    if fork:
        assert continuation is not None
        # Moments and group settings stay as loaded; only the exhausted planned
        # schedule is replaced, starting at each group's saved final rate.
        for group in optimizer.param_groups:
            group["lr_schedule"] = tail_policy(
                group["lr"], continuation["additional_steps"], origin
            )
    if state:
        assert resume is not None
        training_seconds, job_seconds = (
            state["budget"]["train_seconds"],
            state["budget"]["job_seconds"],
        )
        best_score, stale = state["best_score"], state["stale_validations"]
        lost_record = _unjournaled_best_record(
            Path(resume).resolve().parent / "best.pt", history_path
        )
        best_score, best_step = carry_best_checkpoint(
            resume, state, checkpoint_dir / "best.pt"
        )
        if lost_record is not None:
            with history_path.open("a", encoding="utf-8") as history:
                history.write(json.dumps(lost_record, allow_nan=False) + "\n")
        last_validation_step = state.get("last_validation_step", -1)
        ledger_path = Path(resume).resolve().parent.parent / "budget.json"
        if ledger_path.exists():
            ledger = read_json(ledger_path)
            if ledger["run_id"] != run_id:
                raise ValueError("budget ledger belongs to another run")
            training_seconds = max(training_seconds, ledger["train_seconds"])
            job_seconds = max(job_seconds, ledger["job_seconds"])
        # The sidecar accounts for the final atomic checkpoint write itself.
        summary_path = Path(resume).resolve().parent.parent / "summary.json"
        if summary_path.exists():
            previous = read_json(summary_path)
            if previous.get("last_checkpoint_sha256") == file_sha256(resume):
                job_seconds = max(job_seconds, previous["job_seconds"])
                last_validation_step = max(
                    last_validation_step, previous.get("last_validation_step", -1)
                )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    config_hash = record_config(
        output / "config.resolved.json", config, overwrite=bool(state)
    )
    provenance = {
        "run_id": run_id,
        "schema_version": "1.0",
        **code_identity(),
        "manifest_sha256": manifest_hashes,
        "dataset_identity": dataset_identity,
        "config_sha256": config_hash,
        "environment": environment(device),
        "parameter_count": parameter_count,
        "training_objective": objective,
        **({"continuation": continuation} if continuation else {}),
        "execution_limits": {
            "max_train_seconds": training_limit,
            "max_steps": config.max_steps if control.use_step_limit else None,
            "patience": config.patience if control.use_early_stopping else None,
            "validation_every": config.validation_every
            if control.validate_during_training
            else None,
            "evaluate_at_end": control.evaluate_at_end,
            "checkpoint_every": config.checkpoint_every,
        },
    }
    write_json(output / "run.json", provenance, overwrite=bool(state))
    last_metrics = None
    initial_step = step

    # Resume already reconciles checkpoint, ledger and summary above. Add this
    # invocation's actual staging work once before checkpointing/enforcing limits.
    job_seconds += control.setup_job_seconds

    def elapsed_job():
        return job_seconds + time.perf_counter() - started

    if elapsed_job() < config.max_job_seconds:
        preflight(val_data)

    def record_budget():
        # Keep completed work charged even if a hard stop loses newer weights.
        write_json(
            output / "budget.json",
            {
                "run_id": run_id,
                "step": step,
                "train_seconds": training_seconds,
                "job_seconds": elapsed_job(),
            },
            overwrite=True,
        )

    def snapshot():
        for dataset in sources.values():
            dataset.verify_identity()
        return {
            "schema_version": CHECKPOINT_VERSION,
            "world_scale_m": WORLD_SCALE_M,
            "run_id": run_id,
            "config": resolved,
            "provenance": provenance,
            "manifest_sha256": manifest_hashes,
            "dataset_identity": dataset_identity,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None,
            "precision_scaler": None,
            "training_objective": objective,
            **({"continuation": continuation} if continuation else {}),
            "step": step,
            "sampler": sampler.state_dict(),
            "rng": random_state(),
            "best_score": best_score,
            "best_step": best_step,
            "stale_validations": stale,
            "last_validation_step": last_validation_step,
            "budget": {"train_seconds": training_seconds, "job_seconds": elapsed_job()},
        }

    def exposure():
        return added_exposure(
            continuation, step=step, batch_size=config.batch_size, sampler=sampler
        )

    def validate():
        nonlocal last_metrics, best_score, best_step, stale, last_validation_step
        if last_validation_step == step:
            return
        update("evaluation", step=step, train_seconds=training_seconds)
        last_metrics = evaluate_model(ema.module, val_data, device=device)
        if (
            last_metrics["manifest_sha256"] != manifest_hashes["val"]
            or last_metrics["source_identity"] != dataset_identity["val"]
        ):
            raise ValueError("validation manifest changed during training")
        score = last_metrics["modes"]["corrupted"]["val_xyz_m"]
        last_validation_step = step
        # One event per successful validation, independent of checkpoint cadence.
        # Metric suffixes _m/_px denote metres/pixels; fractions are unitless.
        history_row = {
            "schema_version": "1.0",
            "run_id": run_id,
            "invocation_id": invocation_id,
            "resumed_from_step": state["step"] if state else None,
            "step": step,
            "windows": step * config.batch_size,
            "sampled_views": sampler.draws,
            "rejected_crops": sampler.rejected,
            "epoch": sampler.epoch,
            "train_seconds": training_seconds,
            "job_seconds": elapsed_job(),
            "training_objective": objective,
            "loss_units": loss_units,
            **({"continuation": exposure()} if continuation else {}),
            "manifest_sha256": manifest_hashes["val"],
            "source_identity": dataset_identity["val"],
            "modes": {
                mode: {
                    k: v for k, v in metrics.items() if k not in ("views", "rallies")
                }
                for mode, metrics in last_metrics["modes"].items()
            },
        }
        if best_score is None or score < best_score:
            best_score, stale = score, 0
            best_step = step
            save_checkpoint(
                checkpoint_dir / "best.pt",
                {**snapshot(), "validation_record": history_row},
            )
        else:
            stale += 1
        write_json(
            output / "validation.json", {"step": step, **last_metrics}, overwrite=True
        )
        with history_path.open("a", encoding="utf-8") as history:
            history.write(json.dumps(history_row, allow_nan=False) + "\n")
        record_budget()

    # A crash before the first completed step still leaves a coherent restart.
    update("checkpoint", step=step, train_seconds=training_seconds)
    save_checkpoint(checkpoint_dir / "last.pt", snapshot())
    record_budget()
    if (
        continuation
        and step == origin
        and last_validation_step != step
        and control.validate_during_training
        and not control.stop_requested
        and elapsed_job() < config.max_job_seconds
    ):
        # The candidate's own measurement of the inherited weights. The source
        # run's score stays in the lineage record and is never copied here.
        validate()
        save_checkpoint(checkpoint_dir / "last.pt", snapshot())
    stop_reason = "max_steps"
    log_path = output / "loss.jsonl"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        while not control.use_step_limit or step < config.max_steps:
            now = time.perf_counter()
            if control.stop_requested:
                stop_reason = "requested"
                break
            if (
                control.use_early_stopping
                and config.patience is not None
                and stale >= config.patience
            ):
                stop_reason = "validation_plateau"
                break
            if control.deadline is not None and now >= control.deadline:
                stop_reason = "external_deadline"
                break
            if training_seconds >= training_limit:
                stop_reason = "train_budget"
                break
            if elapsed_job() >= config.max_job_seconds:
                stop_reason = "job_budget"
                break
            if (
                control.stop_after_steps is not None
                and step - initial_step >= control.stop_after_steps
            ):
                stop_reason = "external_step_limit"
                break
            update("training", step=step, train_seconds=training_seconds)
            synchronize(device)
            step_started = time.perf_counter()
            f, target, valid = sampler.next_batch()
            f, target, valid = f.to(device), target.to(device), valid.to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(f), target, valid, f.padding)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            ema.update_parameters(model)
            synchronize(device)
            duration = time.perf_counter() - step_started
            training_seconds += duration
            step += 1
            record_budget()
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "loss_units": loss_units,
                "gradient_norm": float(gradient_norm),
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                "step_seconds": duration,
                "train_seconds": training_seconds,
                "job_seconds": elapsed_job(),
                "epoch": sampler.epoch,
                "windows": step * config.batch_size,
                "sampled_views": sampler.draws,
                "rejected_crops": sampler.rejected,
            }
            if step % config.log_every == 0:
                row["optimizer_stability"] = optimizer_stability(optimizer)
                log.write(json.dumps(row, allow_nan=False) + "\n")
            update("training", **row)
            if control.stop_requested:
                stop_reason = "requested"
                break
            validated = False
            if (
                control.validate_during_training
                and (step - origin) % config.validation_every == 0
                and elapsed_job() < config.max_job_seconds
            ):
                validate()
                validated = True
            if validated or (
                config.checkpoint_every is not None
                and (step - origin) % config.checkpoint_every == 0
            ):
                update("checkpoint", step=step, train_seconds=training_seconds)
                save_checkpoint(checkpoint_dir / "last.pt", snapshot())
            if (
                validated
                and control.use_early_stopping
                and config.patience is not None
                and stale >= config.patience
            ):
                stop_reason = "validation_plateau"
                break
    if (
        control.evaluate_at_end
        and last_validation_step != step
        and elapsed_job() < config.max_job_seconds
        and not control.stop_requested
    ):
        validate()
    update("checkpoint", step=step, train_seconds=training_seconds)
    save_checkpoint(checkpoint_dir / "last.pt", snapshot())
    record_budget()
    summary = {
        **provenance,
        "step": step,
        "train_seconds": training_seconds,
        "train_budget_seconds": training_limit,
        "train_overrun_seconds": max(0.0, training_seconds - training_limit),
        "stop_reason": stop_reason,
        "best_score": best_score,
        "last_checkpoint": str(checkpoint_dir / "last.pt"),
        "best_checkpoint": str(checkpoint_dir / "best.pt")
        if (checkpoint_dir / "best.pt").exists()
        else None,
        "last_checkpoint_sha256": file_sha256(checkpoint_dir / "last.pt"),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if device == "cuda"
        else None,
        "last_validation_step": last_validation_step,
    }
    if continuation:
        summary["added_exposure"] = exposure()
    summary["job_seconds"] = elapsed_job()
    write_json(output / "summary.json", summary, overwrite=True)
    update(
        "complete", step=step, train_seconds=training_seconds, stop_reason=stop_reason
    )
    return summary
