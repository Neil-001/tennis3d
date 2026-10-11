"""Tennis-specific shot proposals and acceptance, separate from engine stepping."""

from dataclasses import dataclass

import numpy as np

from tennis3d.court.model import LENGTH_M, SERVICE_LINE_DISTANCE_M, SINGLES_WIDTH_M
from tennis3d.synthetic.priors import ShotPriors, choose
from tennis3d.synthetic.simulate import Event, Simulator, State


@dataclass
class Segment:
    times: np.ndarray
    xyz: np.ndarray
    velocity: np.ndarray
    spin: np.ndarray
    events: list[Event]
    end: State
    kind: str


class InvalidShot(ValueError):
    pass


# (low, high) forward and upward speed in m/s, then X spin in rad/s (negative is topspin).
RETURN_PRIOR = ((17, 29), (4, 8), (-230, -60))
SHOT_PRIORS = {
    "lob": ((11, 17), (10, 14), (-90, 0)),
    "slice": ((16, 24), (3, 6), (60, 140)),
    "volley": ((12, 22), (0, 4), (-40, 80)),
    "out": ((25, 35), (6, 10), (-80, 20)),
}


def proposal(
    rng: np.random.Generator, kind: str, priors: ShotPriors | None = None
) -> State:
    """Canonical near-to-far priors. Opposite-side shots rotate about world Z."""
    if priors is not None and kind in priors.proposals:
        component = choose(rng, priors.proposals[kind].components)
        p, v, w = (
            rng.uniform(*np.transpose(bounds))
            for bounds in (
                component.position_m,
                component.velocity_m_s,
                component.spin_rad_s,
            )
        )
        v[0] -= p[0] * component.aim_x_gain_s_inv
        return State(p, v, w)
    if kind in ("toss", "serve"):
        p = np.array(
            [rng.uniform(-2.5, -0.4), rng.uniform(-12.5, -12.0), rng.uniform(2.3, 3.3)]
        )
        if kind == "toss":
            p[2] = rng.uniform(1.3, 1.6)
            v = [rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(4.5, 5.8)]
            return State(p, np.array(v), np.zeros(3))
        v = np.array([rng.uniform(3, 9), rng.uniform(32, 48), rng.uniform(-2.5, 0.5)])
        w = np.array(
            [rng.uniform(-220, -30), rng.uniform(-40, 40), rng.uniform(-30, 30)]
        )
        return State(p, v, w)
    p = np.array([rng.uniform(-3.5, 3.5), rng.uniform(-13, -5), rng.uniform(0.75, 1.5)])
    if kind == "volley":
        p[1] = rng.uniform(-6, -3)
        p[2] = rng.uniform(1, 2.7)
    # Always draw the return prior first so every kind keeps the same RNG stream.
    speed, vz, spin = (rng.uniform(*bounds) for bounds in RETURN_PRIOR)
    if kind in SHOT_PRIORS:
        speed, vz, spin = (rng.uniform(*bounds) for bounds in SHOT_PRIORS[kind])
    v = np.array([rng.uniform(-2.5, 2.5) - p[0] * 0.15, speed, vz])
    w = np.array([spin, rng.uniform(-25, 25), rng.uniform(-25, 25)])
    return State(p, v, w)


def rotate(state: State, side: int) -> State:
    rotation = np.array([side, side, 1])
    return State(state.xyz * rotation, state.velocity * rotation, state.spin * rotation)


def run_shot(
    sim: Simulator,
    initial: State,
    kind: str,
    *,
    fps: float = 60,
    end_at_volley: bool = False,
    end_at_bounce: bool = False,
    toss_offset: float = 0,
    sample_jitter_s: float = 0,
    sample_seed: int = 0,
) -> Segment:
    """Sample actual integration times, never interpolate through a hit or contact.

    A return endpoint is an opponent racket opportunity after exactly one legal
    bounce, or before any bounce when end_at_volley is requested. Out shots end
    at their first out bounce. A toss ends around the apex, before the serve.
    """
    sim.reset(initial)
    side = 1 if initial.xyz[1] < 0 else -1
    times, states, events = [0.0], [initial], []
    sample_rng = np.random.default_rng(sample_seed)
    sample_index = 1
    next_sample = 1 / fps + sample_rng.uniform(-sample_jitter_s, sample_jitter_s)
    crossed, bounces, separated = False, 0, False
    apex_time = None
    previous = initial.xyz.copy()
    while sim.data.time < 5.0:
        dt, onset, separation = sim.step()
        t = float(sim.data.time)
        p, v = sim.data.qpos[:3], sim.data.qvel[:3]
        if (
            not np.isfinite(p).all()
            or p[2] < sim.config.radius_m - sim.config.max_penetration_m
        ):
            raise InvalidShot("excessive contact penetration or nonfinite state")
        if p[2] > 15 or abs(p[0]) > 15 or abs(p[1]) > 28:
            raise InvalidShot("outside bounded tennis scene")
        if previous[1] * side < 0 <= p[1] * side:
            # Substep crossing, not a sparse-frame threshold. Net sag is approximated
            # by a straight slope from 0.914m centre to 1.07m singles posts.
            a = -previous[1] / (p[1] - previous[1])
            cross = previous + a * (p - previous)
            height = 0.914 + 0.156 * min(
                abs(cross[0]) / (SINGLES_WIDTH_M / 2 + 0.914), 1
            )
            if (
                abs(cross[0]) > SINGLES_WIDTH_M / 2 + 0.914
                or cross[2] - sim.config.radius_m <= height
            ):
                raise InvalidShot("net strike or outside supported net span")
            crossed = True
            events.append(Event(t - dt + a * dt, "net_crossing", cross, dt))
        if onset:
            bounces += 1
            events.append(Event(t, "bounce", p.copy(), dt))
            if not crossed or p[1] * side <= 0 or bounces > 1:
                raise InvalidShot("own-side or second bounce")
            inside = abs(p[0]) <= SINGLES_WIDTH_M / 2 and abs(p[1]) <= LENGTH_M / 2
            if kind == "serve":
                inside = (
                    inside
                    and abs(p[1]) <= SERVICE_LINE_DISTANCE_M
                    and p[0] * initial.xyz[0] < 0
                )
            if kind == "out":
                if inside:
                    raise InvalidShot("requested out shot landed in court")
                events.append(Event(t, "out", p.copy(), dt))
            elif not inside:
                raise InvalidShot("bounce outside target court/service box")
        separated = separated or separation
        done = False
        if kind == "toss":
            if v[2] <= max(0, -toss_offset * 9.81) and apex_time is None:
                apex_time = t
            done = apex_time is not None and t >= apex_time + max(0, toss_offset)
            if done and not 2.2 <= p[2] <= 3.5:
                raise InvalidShot("toss contact outside serve height range")
        elif kind == "out" or end_at_bounce:
            done = bounces == 1
        elif end_at_volley:
            if bounces:
                raise InvalidShot("volley feed bounced before receiver")
            done = crossed and p[1] * side >= 4.5 and 0.8 <= p[2] <= 3.5
        else:
            done = separated and p[2] >= 0.8 and p[1] * side >= 3
            if separated and v[2] < 0 and p[2] < 0.8:
                raise InvalidShot("bounce too low for supported racket contact")
        if t + 1e-12 >= next_sample or done:
            times.append(t)
            states.append(sim.state())
            sample_index += 1
            next_sample = sample_index / fps + sample_rng.uniform(
                -sample_jitter_s, sample_jitter_s
            )
        previous = p.copy()
        if done:
            return Segment(
                np.array(times),
                np.array([s.xyz for s in states]),
                np.array([s.velocity for s in states]),
                np.array([s.spin for s in states]),
                events,
                sim.state(),
                kind,
            )
    raise InvalidShot("no valid endpoint within 5 seconds")
