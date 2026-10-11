"""Prior boundaries, joint sampling, cache isolation and stable family identities."""

import numpy as np
import pytest
from pydantic import ValidationError

pytest.importorskip("mujoco")
pytest.importorskip("scipy")

from tennis3d.artifacts import ArtifactProvenance, json_sha256
from tennis3d.synthetic import KINDS
from tennis3d.synthetic.config import SimulationConfig
from tennis3d.synthetic.generate import family_physics, family_seed_for, pool_settings
from tennis3d.synthetic.priors import FamilyPriors, RallyPriors, ShotPriors
from tennis3d.synthetic.rally import assemble
from tennis3d.synthetic.shot_pool import load_or_build
from tennis3d.synthetic.shots import proposal
from tennis3d.synthetic.simulate import Simulator, State


def component(name="flat", weight=1.0, speed=(40.0, 45.0), spin=(-150.0, -100.0)):
    return {
        "name": name,
        "weight": weight,
        "position_m": [[-2.0, -1.0], [-12.5, -12.0], [2.5, 3.0]],
        "velocity_m_s": [[3.0, 6.0], list(speed), [-2.0, 0.0]],
        "spin_rad_s": [list(spin), [-20.0, 20.0], [-20.0, 20.0]],
    }


def test_joint_shot_components_bounds_replay_and_roundtrip():
    priors = ShotPriors.model_validate(
        {
            "proposals": {
                "serve": {
                    "components": [
                        component(weight=0.5),
                        component("kick", 0.5, (30.0, 35.0), (-360.0, -300.0)),
                    ]
                }
            }
        }
    )
    cfg = SimulationConfig(shot_priors=priors)
    assert SimulationConfig.model_validate_json(cfg.model_dump_json()) == cfg
    rng = np.random.default_rng(53)
    draws = [proposal(rng, "serve", priors) for _ in range(100)]
    assert any(s.velocity[1] > 40 for s in draws)
    assert any(s.velocity[1] < 35 for s in draws)
    for state in draws:
        assert -2 <= state.xyz[0] <= -1 and 2.5 <= state.xyz[2] <= 3
        if state.velocity[1] > 40:
            assert -150 <= state.spin[0] <= -100
        else:
            assert 30 <= state.velocity[1] <= 35 and -360 <= state.spin[0] <= -300
    replay = np.random.default_rng(53)
    for expected in draws:
        actual = proposal(replay, "serve", priors)
        np.testing.assert_array_equal(expected.velocity, actual.velocity)
        np.testing.assert_array_equal(expected.spin, actual.spin)
    for kind in KINDS:
        old = proposal(np.random.default_rng(7), kind)
        fallback = proposal(np.random.default_rng(7), kind, ShotPriors())
        np.testing.assert_array_equal(old.xyz, fallback.xyz)
        np.testing.assert_array_equal(old.velocity, fallback.velocity)
        np.testing.assert_array_equal(old.spin, fallback.spin)


@pytest.mark.parametrize(
    "change",
    [
        {"weight": 0.7},
        {"position_m": [[0.0, 1.0], [-1.0, -2.0], [1.0, 2.0]]},
        {"velocity_m_s": [[0.0, float("inf")], [1.0, 2.0], [1.0, 2.0]]},
        {"spin_rad_s": [[0.0, float("nan")], [1.0, 2.0], [1.0, 2.0]]},
        {"spin_rpm": [[0.0, 1.0]] * 3},
    ],
)
def test_invalid_shot_priors_rejected(change):
    with pytest.raises(ValidationError):
        ShotPriors.model_validate(
            {"proposals": {"serve": {"components": [{**component(), **change}]}}}
        )


@pytest.mark.parametrize(
    "values",
    [
        {"cadence_fps": [30]},
        {"cadence_weights": [0.2, 0.2, 0.2, 0.2]},
        {"return_kinds": ["serve"]},
        {"return_weights": [float("nan"), 0.0, 0.0, 1.0]},
        {"toss_offset_s": [0.1, -0.1]},
        {"terminal_out_probability": 1.1},
    ],
)
def test_invalid_rally_priors_rejected(values):
    with pytest.raises(ValidationError):
        RallyPriors.model_validate(values)


def test_split_family_counts_and_order_independent_seeds():
    cfg = SimulationConfig(train_rallies=20, train_families=10)
    assert [cfg.family_count(s) for s in ("train", "val", "test")] == [10, 1, 1]
    with pytest.raises(ValidationError, match="families cannot exceed"):
        SimulationConfig(train_families=5)
    with pytest.raises(ValidationError, match="half the shortest"):
        SimulationConfig(
            timestamp_jitter_s=0.003,
            rally_priors=RallyPriors(cadence_fps=[240], cadence_weights=[1.0]),
        )
    first = np.random.SeedSequence(cfg.seed).spawn(3)
    for split in range(3):
        assert family_seed_for(cfg.seed, split, 0) == int(
            first[split].generate_state(1)[0]
        )
    identities = [(split, family) for split in range(3) for family in range(10)]
    forward = {
        identity: family_seed_for(cfg.seed, *identity) for identity in identities
    }
    backward = {
        identity: family_seed_for(cfg.seed, *identity)
        for identity in reversed(identities)
    }
    assert forward == backward and len(set(forward.values())) == 30
    # A different split count does not enter a surviving family's physical draw.
    other = cfg.model_copy(update={"train_families": 2})
    for identity in identities:
        seed = forward[identity]
        assert family_physics(cfg, seed) == family_physics(other, seed)


def test_ball_family_bounds_and_effective_drag():
    priors = FamilyPriors.model_validate(
        {
            "components": [
                {
                    "name": "new_ball",
                    "weight": 1.0,
                    "mass_kg": [0.056, 0.0594],
                    "radius_m": [0.0327, 0.0343],
                    "blunt_drag": [0.25, 0.28],
                    "magnus": [0.75, 0.85],
                    "density": [1.15, 1.25],
                }
            ]
        }
    )
    cfg = SimulationConfig(family_priors=priors)
    for seed in range(5):
        physics = family_physics(cfg, seed)
        assert 0.056 <= physics.mass_kg <= 0.0594
        assert 0.0327 <= physics.radius_m <= 0.0343
        assert 0.50 <= 2 * physics.blunt_drag <= 0.56
        assert physics == family_physics(cfg, seed)
        sim = Simulator(physics)
        sim.reset(
            State(np.array([0.0, 0.0, 2.0]), np.array([0.0, 25.0, 0.0]), np.zeros(3))
        )
        # MuJoCo's blunt coefficient is half the conventional drag coefficient.
        expected = (
            -physics.density
            * physics.blunt_drag
            * np.pi
            * physics.radius_m**2
            * 25**2
            / physics.mass_kg
        )
        assert sim.data.qacc[1] == pytest.approx(expected, rel=0.01)
        assert sim.data.qacc[2] == pytest.approx(-9.81)
    for bad in (
        {"mass_kg": [0.01, 0.06]},
        {"radius_m": [0.03, 0.04]},
        {"magnus": [-1.0, 1.0]},
        {"density": [1.0, float("inf")]},
    ):
        with pytest.raises(ValidationError):
            FamilyPriors.model_validate(
                {"components": [{"name": "bad", "weight": 1.0, **bad}]}
            )


def test_pool_cache_identity_includes_priors_and_rejects_stale(tmp_path):
    cfg = SimulationConfig(pool_size=2)
    physical = family_physics(cfg, 0)
    settings = pool_settings(cfg, physical, 0, "train", {})
    identity = json_sha256(settings)
    changed = cfg.model_copy(
        update={
            "shot_priors": ShotPriors.model_validate(
                {"proposals": {"serve": {"components": [component()]}}}
            )
        }
    )
    revised_identity = json_sha256(pool_settings(changed, physical, 0, "train", {}))
    assert revised_identity != identity
    assert json_sha256(pool_settings(cfg, physical, 0, "val", {})) != identity
    assert (
        json_sha256(
            pool_settings(
                cfg.model_copy(update={"noise_px": 8.0}), physical, 0, "train", {}
            )
        )
        == identity
    )
    provenance = ArtifactProvenance(
        source_sha256="0" * 64, config_sha256=identity, code_revision="test"
    )
    path = tmp_path / "pool.npz"
    sim = Simulator(physical)
    pool = load_or_build(path, provenance, sim, cfg, 0)
    assert sum(pool.attempts.values()) == sum(
        v for k, v in pool.diagnostics.items() if k.startswith("pool_proposals/")
    )
    assert sum(
        v for k, v in pool.diagnostics.items() if k.startswith("pool_accepted/")
    ) == 2 * len(KINDS)
    cached = load_or_build(path, provenance, sim, cfg, 0)
    for kind in KINDS:
        np.testing.assert_array_equal(pool.states[kind], cached.states[kind])
    with pytest.raises(ValueError, match="provenance"):
        load_or_build(
            path,
            provenance.model_copy(update={"config_sha256": revised_identity}),
            sim,
            changed,
            0,
        )


def test_rally_mixture_controls_cadence_and_terminal_shots(tmp_path):
    cfg = SimulationConfig(
        pool_size=24,
        min_returns=1,
        max_returns=1,
        rally_priors=RallyPriors(
            cadence_fps=[50],
            cadence_weights=[1.0],
            return_kinds=["slice"],
            return_weights=[1.0],
            terminal_out_probability=0.0,
        ),
    )
    sim = Simulator(cfg.physics)
    provenance = ArtifactProvenance(
        source_sha256="0" * 64, config_sha256="1" * 64, code_revision="test"
    )
    pool = load_or_build(tmp_path / "pool.npz", provenance, sim, cfg, 42)
    first = assemble(sim, pool, cfg, 0)
    second = assemble(sim, pool, cfg, 0)
    assert first.fps == 50
    assert first.arrays["segment_kind"].tolist() == [
        KINDS.index(k) for k in ("toss", "serve", "slice")
    ]
    np.testing.assert_array_equal(first.arrays["xyz_m"], second.arrays["xyz_m"])
    assert pool.diagnostics["rally_accepted"] == 2
    assert pool.diagnostics["stitch_candidates/serve"] >= 2
    assert first.arrays["event_code"][-1] == 1


@pytest.mark.parametrize(
    "weights", [[0.6, 0.6], [0.0, 1.0], [1.0], [float("nan"), 0.5]]
)
def test_empirical_run_weights_reject_invalid_lists(weights):
    with pytest.raises(ValidationError):
        SimulationConfig(
            visible_run_lengths_s=[0.1, 0.2],
            missing_run_lengths_s=[0.1],
            visible_run_weights=weights,
        )


def test_empirical_run_weights_need_durations_and_roundtrip():
    with pytest.raises(ValidationError, match="must match"):
        SimulationConfig(missing_run_weights=[1.0])
    cfg = SimulationConfig(
        visible_run_lengths_s=[0.1, 0.2],
        missing_run_lengths_s=[0.1],
        visible_run_weights=[0.4, 0.6],
    )
    assert SimulationConfig.model_validate_json(cfg.model_dump_json()) == cfg
