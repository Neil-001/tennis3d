"""Physical scale/sign checks against the adapted simulator."""

import numpy as np
import pytest

pytest.importorskip("mujoco")

from tennis3d.synthetic.config import PhysicsConfig
from tennis3d.synthetic.simulate import Simulator, State


def state(p=(0, 0, 2), v=(0, 0, 0), w=(0, 0, 0)):
    return State(
        np.array(p, dtype=float), np.array(v, dtype=float), np.array(w, dtype=float)
    )


def test_gravity_drag_spin_and_reset():
    vacuum = Simulator(PhysicsConfig(density=0, viscosity=0))
    vacuum.reset(state(v=(0, 30, 0)))
    for _ in range(500):
        vacuum.step()
    np.testing.assert_allclose(
        vacuum.state().xyz, [0, 15, 2 - 0.5 * 9.81 * 0.5**2], atol=0.0025
    )
    np.testing.assert_allclose(vacuum.state().velocity, [0, 30, -4.905], atol=1e-10)
    sim = Simulator(PhysicsConfig())
    accel = []
    for spin in (0, -200, 200):
        sim.reset(state(v=(0, 30, 0), w=(spin, 0, 0)))
        accel.append(sim.data.qacc[:3].copy())
    assert -21 < accel[0][1] < -20
    assert accel[1][2] < accel[0][2] < accel[2][2]
    np.testing.assert_allclose(accel[0][2], -9.81, atol=1e-6)
    first = state(v=(1, 25, 2), w=(-150, 20, 0))
    outcomes = []
    for _ in range(2):
        sim.reset(first)
        for _ in range(100):
            sim.step()
        outcomes.append(sim.state())
    np.testing.assert_array_equal(outcomes[0].xyz, outcomes[1].xyz)
    # Spin is world angular velocity, not rotating free-joint local coordinates.
    np.testing.assert_allclose(
        outcomes[0].spin / np.linalg.norm(outcomes[0].spin),
        first.spin / np.linalg.norm(first.spin),
        atol=0.001,
    )


def rebound(step, velocity=(0, 0, 0), height=2.5735):
    cfg = PhysicsConfig(flight_step_s=min(0.001, step * 16), contact_step_s=step)
    sim = Simulator(cfg)
    sim.reset(state(p=(0, 0, height), v=velocity))
    seen, peak, low, episodes = False, 0, height, 0
    for _ in range(100000):
        _, onset, _ = sim.step()
        episodes += onset
        seen |= onset
        low = min(low, sim.data.qpos[2])
        if seen:
            peak = max(peak, sim.data.qpos[2] - cfg.radius_m)
        if seen and sim.data.qvel[2] < 0 and sim.data.qpos[2] > 0.1:
            break
    return peak, low, episodes, sim.state()


def test_itf_drop_and_contact_convergence():
    coarse = rebound(0.0000625)
    fine = rebound(0.00003125)
    assert 1.35 < coarse[0] < 1.47
    assert abs(coarse[0] - fine[0]) < 0.015
    assert coarse[2] == fine[2] == 1
    assert coarse[1] > 0.02


def test_tennis_speed_oblique_bounce_convergence():
    coarse = rebound(0.0000625, velocity=(0, 40, -12), height=0.3)
    fine = rebound(0.00003125, velocity=(0, 40, -12), height=0.3)
    assert coarse[1] > 0.01
    assert coarse[2] == fine[2] == 1
    assert 0 < coarse[3].velocity[1] < 40
    assert coarse[3].spin[0] < 0
    assert abs(coarse[0] - fine[0]) < 0.04
    assert np.linalg.norm(coarse[3].xyz - fine[3].xyz) < 0.15
