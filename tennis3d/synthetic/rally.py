"""Stitch complete tennis points with exact position continuity at racket hits."""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from tennis3d.synthetic import EVENT_CODES, KINDS
from tennis3d.synthetic.config import SimulationConfig
from tennis3d.synthetic.priors import RallyPriors
from tennis3d.synthetic.shot_pool import GenerationError, ShotPool
from tennis3d.synthetic.shots import InvalidShot, Segment, run_shot
from tennis3d.synthetic.simulate import Event, Simulator, State


@dataclass
class Rally:
    arrays: dict[str, np.ndarray]
    attempts: int
    fps: int


def assemble(sim: Simulator, pool: ShotPool, cfg: SimulationConfig, seed: int) -> Rally:
    rng = np.random.default_rng(seed)
    failures = Counter()
    prior = cfg.rally_priors or RallyPriors()
    # Without configured priors, keep the legacy unweighted cadence draw.
    fps = int(
        rng.choice(prior.cadence_fps)
        if cfg.rally_priors is None
        else rng.choice(prior.cadence_fps, p=prior.cadence_weights)
    )
    if cfg.rally_mode == "diagnostic":
        sequence = list(KINDS)
    else:
        count = int(rng.integers(cfg.min_returns, cfg.max_returns + 1))
        shots = rng.choice(prior.return_kinds, count, p=prior.return_weights).tolist()
        # A service receiver cannot volley the serve.
        if shots[0] == "volley":
            shots[0] = "return"
        if rng.random() < prior.terminal_out_probability:
            shots[-1] = "out"
        sequence = ["toss", "serve", *shots]
    pool.diagnostics["rally_calls"] += 1
    for kind in sequence:
        pool.diagnostics[f"rally_requested/{kind}"] += 1
    for attempt in range(cfg.rally_attempts):
        pool.diagnostics["rally_attempts"] += 1
        row = int(rng.integers(len(pool.states["toss"])))
        initial = pool.states["toss"][row]
        try:
            toss = run_shot(
                sim,
                State(initial[:3], initial[3:6], initial[6:]),
                "toss",
                fps=fps,
                toss_offset=float(
                    rng.uniform(prior.toss_offset_s[0], prior.toss_offset_s[1])
                ),
                sample_jitter_s=cfg.timestamp_jitter_s,
                sample_seed=seed,
            )
            segments, ids = [toss], [row]
            for shot_index, kind in enumerate(sequence[1:], start=1):
                segment, prior_id = pool.stitch(
                    sim,
                    segments[-1].end.xyz,
                    kind,
                    cfg,
                    fps=fps,
                    end_at_volley=shot_index + 1 < len(sequence)
                    and sequence[shot_index + 1] == "volley",
                    end_at_bounce=shot_index == len(sequence) - 1,
                    sample_seed=seed + shot_index,
                )
                segments.append(segment)
                ids.append(prior_id)
            pool.diagnostics["rally_accepted"] += 1
            return Rally(combine(segments, ids), attempt + 1, fps)
        except (GenerationError, InvalidShot) as error:
            reason = str(error).split(" from endpoint ")[0]
            failures[reason] += 1
            pool.diagnostics[f"rally_rejected/{reason}"] += 1
    raise GenerationError(
        f"Rally seed {seed} failed {cfg.rally_attempts} attempts: {dict(failures)}. Increase pool_size/nearest_attempts or revise priors."
    )


def combine(segments: list[Segment], ids: list[int]) -> dict[str, np.ndarray]:
    times, xyz, velocity, spin, segment_ids = [], [], [], [], []
    events: list[Event] = []
    hit_before, hit_after = [], []
    offset = 0.0
    for i, segment in enumerate(segments):
        if i:
            previous = segments[i - 1]
            if not np.array_equal(previous.end.xyz, segment.xyz[0]):
                raise ValueError("stitch position discontinuity")
            events.append(Event(offset, "hit", segment.xyz[0].copy(), 0.0))
            hit_before.append(
                np.concatenate(
                    [previous.end.xyz, previous.end.velocity, previous.end.spin]
                )
            )
            hit_after.append(
                np.concatenate([segment.xyz[0], segment.velocity[0], segment.spin[0]])
            )
        # Endpoint belongs to the outgoing segment. Keep one timestamp with post-hit state.
        stop = -1 if i < len(segments) - 1 else None
        times.extend(segment.times[:stop] + offset)
        xyz.extend(segment.xyz[:stop])
        velocity.extend(segment.velocity[:stop])
        spin.extend(segment.spin[:stop])
        segment_ids.extend([i] * len(segment.times[:stop]))
        events.extend(
            Event(e.time + offset, e.kind, e.xyz, e.precision) for e in segment.events
        )
        offset += segment.times[-1]
    events.sort(key=lambda e: e.time)
    return {
        "t_s": np.array(times),
        "xyz_m": np.array(xyz),
        "velocity_m_s": np.array(velocity),
        "spin_rad_s": np.array(spin),
        "label_valid": np.ones(len(times), dtype=bool),
        "segment_id": np.array(segment_ids, dtype=np.int64),
        "prior_id": np.array(ids, dtype=np.int64),
        "segment_kind": np.array(
            [KINDS.index(segment.kind) for segment in segments], dtype=np.int64
        ),
        "event_t_s": np.array([e.time for e in events]),
        "event_code": np.array([EVENT_CODES[e.kind] for e in events], dtype=np.int64),
        "event_xyz_m": np.array([e.xyz for e in events]),
        "event_precision_s": np.array([e.precision for e in events]),
        "hit_before": np.array(hit_before),
        "hit_after": np.array(hit_after),
    }
