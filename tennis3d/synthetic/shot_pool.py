"""Split-owned cached valid initial states and bounded nearest-prior selection."""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree  # ty: ignore[unresolved-import]

from tennis3d.artifacts import ArtifactProvenance, read_npz, write_npz
from tennis3d.synthetic import KINDS
from tennis3d.synthetic.config import SimulationConfig
from tennis3d.synthetic.shots import (
    InvalidShot,
    Segment,
    proposal,
    rotate,
    run_shot,
)
from tennis3d.synthetic.simulate import Simulator, State


class GenerationError(ValueError):
    pass


@dataclass
class ShotPool:
    states: dict[str, np.ndarray]
    attempts: dict[str, int]
    # Run-local diagnostics do not alter the public numeric artifact contract.
    diagnostics: Counter = field(default_factory=Counter)

    def stitch(
        self,
        sim: Simulator,
        position: np.ndarray,
        kind: str,
        cfg: SimulationConfig,
        *,
        fps: float,
        end_at_volley: bool = False,
        end_at_bounce: bool = False,
        sample_seed: int = 0,
    ) -> tuple[Segment, int]:
        side = 1 if position[1] < 0 else -1
        canonical_position = position * np.array([side, side, 1])
        states = self.states[kind]
        _, nearest = cKDTree(states[:, :3]).query(
            canonical_position, k=min(cfg.nearest_attempts, len(states))
        )
        indices = np.atleast_1d(nearest)
        failures = Counter()
        self.diagnostics[f"stitch_calls/{kind}"] += 1
        for index in indices:
            self.diagnostics[f"stitch_candidates/{kind}"] += 1
            candidate = states[index]
            state = rotate(State(candidate[:3], candidate[3:6], candidate[6:9]), side)
            state.xyz = position.copy()
            try:
                return run_shot(
                    sim,
                    state,
                    kind,
                    fps=fps,
                    end_at_volley=end_at_volley,
                    end_at_bounce=end_at_bounce,
                    sample_jitter_s=cfg.timestamp_jitter_s,
                    sample_seed=sample_seed,
                ), int(index)
            except InvalidShot as error:
                failures[str(error)] += 1
                self.diagnostics[f"stitch_rejected/{kind}/{error}"] += 1
        self.diagnostics[f"stitch_failed/{kind}"] += 1
        raise GenerationError(
            f"No valid {kind} from endpoint {position.tolist()} after {len(indices)} distinct nearest priors: {dict(failures)}. Increase pool_size/nearest_attempts or revise shot priors."
        )


def load_or_build(
    path: Path,
    provenance: ArtifactProvenance,
    sim: Simulator,
    cfg: SimulationConfig,
    seed: int,
) -> ShotPool:
    if path.exists():
        arrays, _ = read_npz(path, expected=provenance)
        for kind in KINDS:
            if (
                arrays[kind].shape != (cfg.pool_size, 9)
                or not np.isfinite(arrays[kind]).all()
            ):
                raise GenerationError(f"Invalid cached {kind} pool at {path}")
        return ShotPool(
            {k: arrays[k] for k in KINDS}, dict(zip(KINDS, arrays["attempts"].tolist()))
        )
    rng = np.random.default_rng(seed)
    states, attempts = {}, {}
    diagnostics = Counter()
    for kind in KINDS:
        valid, failures = [], Counter()
        for attempt in range(cfg.pool_attempts):
            initial = proposal(rng, kind, cfg.shot_priors)
            diagnostics[f"pool_proposals/{kind}"] += 1
            try:
                run_shot(sim, initial, kind)
                diagnostics[f"pool_accepted/{kind}"] += 1
                valid.append(
                    np.concatenate([initial.xyz, initial.velocity, initial.spin])
                )
            except InvalidShot as error:
                failures[str(error)] += 1
                diagnostics[f"pool_rejected/{kind}/{error}"] += 1
            if len(valid) == cfg.pool_size:
                break
        if len(valid) != cfg.pool_size:
            raise GenerationError(
                f"Pool {path.name}/{kind}: only {len(valid)}/{cfg.pool_size} valid states after {cfg.pool_attempts} attempts; rejections={dict(failures)}. Increase pool_attempts or revise shot priors."
            )
        states[kind], attempts[kind] = np.array(valid), attempt + 1
    write_npz(
        path,
        {**states, "attempts": np.array(list(attempts.values()), dtype=np.int64)},
        provenance,
    )
    return ShotPool(states, attempts, diagnostics)
