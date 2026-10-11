"""Versioned main-producer identities, bounded pool reuse and view admission.

The legacy NPZ generator deliberately does not call this module. All proposal
seeds, including attempt zero, belong to this new SHA256 identity namespace.
"""

import fcntl
import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
from pathlib import Path

import numpy as np

from tennis3d.artifacts import ArtifactProvenance, file_sha256, json_sha256
from tennis3d.datasets.arrays import PHYSICAL, VIEW, RallyRecord, ViewRecord
from tennis3d.datasets.schema import Pool, RallyMetadata, Split, ViewMetadata
from tennis3d.evaluation.metrics import gap_durations
from tennis3d.inference.windows import make_windows
from tennis3d.synthetic import GENERATOR_VERSION, KINDS
from tennis3d.synthetic.cameras import observe
from tennis3d.synthetic.config import PhysicsConfig, SimulationConfig
from tennis3d.synthetic.generate import excerpt, family_physics, pool_settings
from tennis3d.synthetic.priors import RallyPriors, choose
from tennis3d.synthetic.rally import assemble
from tennis3d.synthetic.shot_pool import GenerationError, ShotPool, load_or_build
from tennis3d.synthetic.simulate import Simulator

IDENTITY_VERSION = "tennis3d-main-sha256-63-v1"
ADMISSION_VERSION = "supported-clean-corrupt-windows-250-125-v1"
MAX_PROPOSALS = 8
DEFAULT_MAX_RECORD_BYTES = 16 << 20
SPLITS: tuple[Split, ...] = ("train", "val", "test")


def identity_seed(domain: str, *parts: str | int) -> int:
    """Canonical domain-separated SHA256 truncated to signed-metadata width."""
    payload = json.dumps(
        [IDENTITY_VERSION, domain, *parts], separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def family_seed(seed: int, split: str, family_index: int) -> int:
    return identity_seed("family", seed, split, family_index)


def rally_seed(seed: int, split: str, rally_index: int) -> int:
    return identity_seed("rally", seed, split, rally_index)


def proposal_seed(seed: int, view_id: int, attempt: int) -> int:
    return identity_seed("view-proposal", seed, view_id, attempt)


@dataclass(frozen=True)
class FamilyPlan:
    split: Split
    family_index: int
    seed: int
    family_id: str
    physical: PhysicsConfig
    pool_config: dict
    cache_name: str


@dataclass(frozen=True)
class RallyPlan:
    split: Split
    rally_index: int
    family_index: int
    rally_id: str
    seed: int


def plan_family(
    cfg: SimulationConfig,
    split: Split,
    family_index: int,
    source_hash: str,
    runtimes: dict[str, str],
) -> FamilyPlan:
    if not 0 <= family_index < cfg.family_count(split):
        raise ValueError("family index is outside the recipe")
    seed = family_seed(cfg.seed, split, family_index)
    physical = family_physics(cfg, seed)
    settings = pool_settings(cfg, physical, seed, split, runtimes)
    return FamilyPlan(
        split,
        family_index,
        seed,
        f"{split}-family-{family_index:05d}-{seed}",
        physical,
        settings,
        f"{split}-{source_hash}-{json_sha256(settings)}.npz",
    )


def plan_families(
    cfg: SimulationConfig, source_hash: str, runtimes: dict[str, str]
) -> Iterator[FamilyPlan]:
    for split in SPLITS:
        for index in range(cfg.family_count(split)):
            yield plan_family(cfg, split, index, source_hash, runtimes)


def plan_rally(cfg: SimulationConfig, split: Split, rally_index: int) -> RallyPlan:
    if split not in SPLITS or not 0 <= rally_index < getattr(cfg, f"{split}_rallies"):
        raise ValueError("rally index is outside the recipe")
    return RallyPlan(
        split,
        rally_index,
        rally_index % cfg.family_count(split),
        f"{split}-rally-{rally_index:06d}",
        rally_seed(cfg.seed, split, rally_index),
    )


def plan_rallies(cfg: SimulationConfig, cohort_size: int = 16) -> Iterator[RallyPlan]:
    """Mix a fixed family cohort; order is recipe-owned, never worker-owned."""
    if cohort_size < 1:
        raise ValueError("cohort_size must be positive")
    for split in SPLITS:
        families = cfg.family_count(split)
        count = getattr(cfg, f"{split}_rallies")
        for first in range(0, families, cohort_size):
            for cycle in range((count + families - 1) // families):
                for family in range(first, min(first + cohort_size, families)):
                    index = cycle * families + family
                    if index < count:
                        yield plan_rally(cfg, split, index)


def _check_proposals(max_proposals: int) -> None:
    if not 1 <= max_proposals <= MAX_PROPOSALS:
        raise ValueError(f"max_proposals must be in [1, {MAX_PROPOSALS}]")


def audit_plan_seeds(cfg: SimulationConfig, max_proposals: int = 8) -> dict:
    """Audit all planned seeds, including unused retries, without Python sets.

    One uint64 buffer costs 28.6 MB for 210000 rallies, two views and eight
    proposals. In-place heapsort requires constant auxiliary storage. Seed
    derivation streams into the buffer; no rally plans or strings accumulate.
    """
    _check_proposals(max_proposals)
    families = sum(cfg.family_count(split) for split in SPLITS)
    rallies = sum(getattr(cfg, f"{split}_rallies") for split in SPLITS)
    proposals = rallies * cfg.views_per_rally * max_proposals
    values = np.empty(families + rallies + proposals, dtype=np.uint64)
    digest = hashlib.sha256()
    position = 0

    def add(value: int) -> None:
        nonlocal position
        values[position] = value
        position += 1
        digest.update(value.to_bytes(8, "big"))

    for split in SPLITS:
        for index in range(cfg.family_count(split)):
            add(family_seed(cfg.seed, split, index))
        for index in range(getattr(cfg, f"{split}_rallies")):
            seed = rally_seed(cfg.seed, split, index)
            add(seed)
            for view in range(cfg.views_per_rally):
                for attempt in range(max_proposals):
                    add(proposal_seed(seed, view, attempt))
    values.sort(kind="heapsort")
    # Bounded comparison chunks avoid another main-sized temporary array.
    for start in range(1, len(values), 65536):
        stop = min(start + 65536, len(values))
        if np.any(values[start:stop] == values[start - 1 : stop - 1]):
            raise ValueError("planned family/rally/proposal seed collision")
    return {
        "identity_version": IDENTITY_VERSION,
        "families": families,
        "rallies": rallies,
        "proposals": proposals,
        "unique_seeds": len(values),
        "ordered_seed_sha256": digest.hexdigest(),
        "audit_buffer_bytes": values.nbytes,
    }


@dataclass
class _CachedFamily:
    sim: Simulator
    pool: ShotPool
    descriptor: Pool
    file_identity: tuple[int, int, int, int]


_CACHE: OrderedDict[tuple[int, str], _CachedFamily] = OrderedDict()


def close_worker_cache() -> None:
    """Release process-owned MuJoCo scenes and numeric pool arrays."""
    _CACHE.clear()


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _get_family(
    cfg: SimulationConfig,
    plan: FamilyPlan,
    pool_cache: Path,
    source_hash: str,
    cache_size: int,
) -> tuple[_CachedFamily, dict]:
    if cache_size < 1:
        raise ValueError("cache_size must be positive")
    # A fork must not reuse its parent's compiled simulation or pool diagnostics.
    for key in list(_CACHE):
        if key[0] != os.getpid():
            del _CACHE[key]
    while len(_CACHE) > cache_size:
        _CACHE.popitem(last=False)
    path = Path(pool_cache).resolve() / plan.cache_name
    key = (os.getpid(), str(path))
    cached = _CACHE.get(key)
    if cached is not None:
        if _file_identity(path) != cached.file_identity:
            raise ValueError(f"immutable pool changed: {path}")
        _CACHE.move_to_end(key)
        return cached, {"cache_hit": True, "pool_built": False}
    path.parent.mkdir(parents=True, exist_ok=True)
    sim = Simulator(plan.physical)
    provenance = ArtifactProvenance(
        source_sha256=source_hash,
        config_sha256=json_sha256(plan.pool_config),
        code_revision=GENERATOR_VERSION,
    )
    # Every process locks the same exact pool identity before testing existence.
    with path.with_suffix(".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        built = not path.exists()
        pool = load_or_build(path, provenance, sim, cfg, plan.seed)
        descriptor = Pool(
            family_id=plan.family_id,
            split=plan.split,
            sha256=file_sha256(path),
            config=plan.pool_config,
            attempts=pool.attempts,
        )
        identity = _file_identity(path)
    for array in pool.states.values():
        array.flags.writeable = False
    cached = _CachedFamily(sim, pool, descriptor, identity)
    _CACHE[key] = cached
    while len(_CACHE) > cache_size:
        _CACHE.popitem(last=False)
    return cached, {
        "cache_hit": False,
        "pool_built": built,
        "pool_diagnostics": dict(pool.diagnostics),
    }


def prepare_family(
    cfg: SimulationConfig,
    split: Split,
    family_index: int,
    pool_cache: Path,
    source_hash: str,
    runtimes: dict[str, str],
    *,
    cache_size: int = 8,
) -> tuple[Pool, dict]:
    plan = plan_family(cfg, split, family_index, source_hash, runtimes)
    cached, diagnostic = _get_family(cfg, plan, pool_cache, source_hash, cache_size)
    return cached.descriptor, {"family_id": plan.family_id, **diagnostic}


def admission_failures(arrays: dict[str, np.ndarray]) -> dict:
    """Check unchanged shared windows, including short terminal intervals."""
    windows = make_windows(arrays["camera_valid"], max_frames=250, stride=125)
    rejected = []
    for index, window in enumerate(windows):
        reasons = []
        rows = slice(window.start, window.stop)
        if not arrays["ball_in_frame"][rows].any():
            reasons.append("no_clean_evidence")
        if not arrays["ball_observed"][rows].any():
            reasons.append("no_corrupted_evidence")
        if reasons:
            rejected.append(
                {
                    "index": index,
                    "start": window.start,
                    "stop": window.stop,
                    "reasons": reasons,
                }
            )
    supported = bool((arrays["camera_valid"] & arrays["label_valid"]).any())
    return {
        "accepted": supported and not rejected,
        "reasons": [] if supported else ["no_supported_labels"],
        "window_count": len(windows),
        "rejected_windows": rejected,
    }


def _numeric_summary(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    return {
        "count": len(values),
        "mean": float(values.mean()) if len(values) else None,
        "min": float(values.min()) if len(values) else None,
        "max": float(values.max()) if len(values) else None,
    }


def proposal_summary(
    arrays: dict[str, np.ndarray], cfg: SimulationConfig, seed: int
) -> dict:
    """Compact per-proposal denominators and geometry for conditioning audits."""
    rng = np.random.default_rng(seed)
    components = cfg.camera.components
    component = components[0] if len(components) == 1 else choose(rng, components)
    valid = arrays["camera_valid"]
    supported = valid & arrays["label_valid"]
    observed = arrays["ball_observed"]
    center = -np.einsum("tji,tj->ti", arrays["camera_R"], arrays["camera_t"])
    distance = np.linalg.norm(arrays["xyz_m"] - center, axis=-1)
    gap = gap_durations(arrays["t_s"], observed)
    starts = np.flatnonzero(np.diff(np.r_[False, ~observed].astype(int)) == 1)
    return {
        "camera_component": component.name,
        "frames": len(valid),
        "camera_valid": int(valid.sum()),
        "supported": int(supported.sum()),
        "in_frame": int(arrays["ball_in_frame"].sum()),
        "observed": int(observed.sum()),
        "missing_fraction": float((~observed).mean()),
        "missing_runs": _numeric_summary(gap[starts]),
        "camera_height_m": _numeric_summary(center[valid, 2]),
        "camera_distance_m": _numeric_summary(distance[valid]),
        "focal_px": _numeric_summary(arrays["camera_K"][valid, 0, 0]),
    }


class AdmissionError(GenerationError):
    """Exact planned identity plus bounded diagnostics, preserved across processes."""

    def __init__(self, rally_id: str, view_id: int, diagnostics: dict):
        super().__init__(rally_id, view_id, diagnostics)
        self.rally_id = rally_id
        self.view_id = view_id
        self.diagnostics = diagnostics

    def __str__(self) -> str:
        return f"Admission exhausted for {self.rally_id}/view-{self.view_id:02d}"


class RecordSizeError(GenerationError):
    """Small, pickleable failure instead of an oversized worker result."""

    def __init__(self, rally_id: str, limit: int, measured: int, stage: str):
        super().__init__(rally_id, limit, measured, stage)
        self.rally_id, self.limit, self.measured, self.stage = (
            rally_id,
            limit,
            measured,
            stage,
        )

    def __str__(self) -> str:
        return (
            f"record {self.rally_id} exceeds max_record_bytes={self.limit} "
            f"at {self.stage}: at least {self.measured} bytes"
        )


def recipe_frame_bytes(cfg: SimulationConfig) -> int:
    """Reserve frame arrays before physics using the five-second shot horizon.

    This covers the complete physical rally and full/middle-60-percent views.
    Events, metadata and diagnostics still need the actual serialized guard;
    this is not an absolute bound on all simulator allocations.
    """
    prior = cfg.rally_priors or RallyPriors()
    segments = len(KINDS) if cfg.rally_mode == "diagnostic" else cfg.max_returns + 2
    # Include the initial row, final endpoint and jitter/step overshoot.
    frames = segments * (5 * max(prior.cadence_fps) + 3)
    frame_keys = {
        "t_s",
        "xyz_m",
        "velocity_m_s",
        "spin_rad_s",
        "label_valid",
        "segment_id",
    }

    def width(layout, keys):
        return sum(
            np.dtype(layout[key][0]).itemsize
            * int(np.prod(layout[key][1], dtype=np.int64))
            for key in keys
        )

    physical_width = width(PHYSICAL, frame_keys)
    view_width = width(VIEW, frame_keys | (set(VIEW) - set(PHYSICAL) - {"image_size"}))
    view_frames = frames + (cfg.views_per_rally - 1) * ((3 * frames + 4) // 5)
    return physical_width * frames + view_width * view_frames


def _check_bytes(rally_id: str, limit: int, measured: int, stage: str) -> None:
    if measured > limit:
        raise RecordSizeError(rally_id, limit, measured, stage)


def _transport_bytes(value, rally_id: str, limit: int) -> int:
    """Count the actual multiprocessing pickle without retaining another copy."""

    class Counter:
        size = 0

        def write(self, payload):
            self.size += len(payload)
            _check_bytes(rally_id, limit, self.size, "serialized worker result")

    counter = Counter()
    ForkingPickler(counter).dump(value)
    return counter.size


def _check_worker_result(result: tuple[RallyRecord, dict], limit: int) -> int:
    return _transport_bytes(result, result[0].metadata.rally_id, limit)


def produce_record(
    cfg: SimulationConfig,
    split: Split,
    rally_index: int,
    pool: Pool,
    pool_cache: Path,
    source_hash: str,
    runtimes: dict[str, str],
    *,
    cache_size: int = 8,
    max_proposals: int = 8,
    max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES,
) -> tuple[RallyRecord, dict]:
    """Generate one fixed physical rally and admit its declared full/excerpt views."""
    _check_proposals(max_proposals)
    planned = plan_rally(cfg, split, rally_index)
    if (
        isinstance(max_record_bytes, bool)
        or not isinstance(max_record_bytes, int)
        or max_record_bytes < 1
    ):
        raise ValueError("max_record_bytes must be a positive integer")
    _check_bytes(
        planned.rally_id,
        max_record_bytes,
        recipe_frame_bytes(cfg),
        "recipe frame reservation",
    )
    family = plan_family(cfg, split, planned.family_index, source_hash, runtimes)
    cached, _ = _get_family(cfg, family, pool_cache, source_hash, cache_size)
    if cached.descriptor != pool:
        raise ValueError(f"planned pool descriptor mismatch for {planned.rally_id}")
    before = cached.pool.diagnostics.copy()
    try:
        rally = assemble(cached.sim, cached.pool, cfg, planned.seed)
    except GenerationError as error:
        raise GenerationError(f"{planned.rally_id}: {error}") from error
    numeric_bytes = sum(array.nbytes for array in rally.arrays.values())
    _check_bytes(planned.rally_id, max_record_bytes, numeric_bytes, "physical arrays")
    for array in rally.arrays.values():
        array.flags.writeable = False
    diagnostic = {
        "identity_version": IDENTITY_VERSION,
        "admission_version": ADMISSION_VERSION,
        "split": split,
        "rally_id": planned.rally_id,
        "family_id": family.family_id,
        "family_seed": family.seed,
        "rally_seed": planned.seed,
        "physics_diagnostics": dict(cached.pool.diagnostics - before),
        "views": [],
    }
    views = []
    for view_id in range(cfg.views_per_rally):
        arrays, offset = rally.arrays, 0.0
        if view_id:
            length = len(arrays["t_s"])
            arrays, offset = excerpt(arrays, length // 5, length * 4 // 5)
            for array in arrays.values():
                array.flags.writeable = False
        view_diagnostic = {
            "view_id": view_id,
            "physical_time_origin_s": offset,
            "original_seed": proposal_seed(planned.seed, view_id, 0),
            "accepted_seed": None,
            "attempt_count": 0,
            "proposals": [],
        }
        diagnostic["views"].append(view_diagnostic)
        for attempt in range(max_proposals):
            seed = proposal_seed(planned.seed, view_id, attempt)
            observed = observe(arrays, cfg, seed)
            proposed_bytes = sum(array.nbytes for array in observed.values())
            _check_bytes(
                planned.rally_id,
                max_record_bytes,
                numeric_bytes + proposed_bytes,
                "view arrays",
            )
            checked = admission_failures(observed)
            view_diagnostic["proposals"].append(
                {
                    "attempt": attempt,
                    "seed": seed,
                    **checked,
                    "summary": proposal_summary(observed, cfg, seed),
                }
            )
            view_diagnostic["attempt_count"] += 1
            if checked["accepted"]:
                view_diagnostic["accepted_seed"] = seed
                views.append(
                    ViewRecord(
                        ViewMetadata(
                            view_id=view_id,
                            view_seed=seed,
                            physical_time_origin_s=offset,
                        ),
                        observed,
                    )
                )
                numeric_bytes += proposed_bytes
                break
        else:
            error = AdmissionError(planned.rally_id, view_id, diagnostic)
            _transport_bytes(error, planned.rally_id, max_record_bytes)
            raise error
    record = RallyRecord(
        RallyMetadata(
            rally_id=planned.rally_id,
            family_id=family.family_id,
            pool_sha256=pool.sha256,
            split=split,
            seed=planned.seed,
            fps=rally.fps,
            attempts=rally.attempts,
        ),
        rally.arrays,
        views,
    )
    result = record, diagnostic
    _check_worker_result(result, max_record_bytes)
    return result
