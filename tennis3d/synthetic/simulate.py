# SPDX-License-Identifier: GPL-3.0-only
# Adapted from KieDani/UpliftingTableTennis, revision 1a64bf6488f3a142f733e73e8de00bfecf044b23.
# See licenses/Uplifting-GPL-3.0.txt and licenses/README.md for provenance.
"""Native MuJoCo stepping with explicit world spin and contact episodes."""

from dataclasses import dataclass

import mujoco
import numpy as np

from tennis3d.synthetic.config import PhysicsConfig


@dataclass
class State:
    xyz: np.ndarray
    velocity: np.ndarray
    spin: np.ndarray


@dataclass
class Event:
    time: float
    kind: str
    xyz: np.ndarray
    precision: float


class Simulator:
    """Reuse one compiled sphere/plane scene; no graphics or custom force law."""

    def __init__(self, config: PhysicsConfig):
        self.config = config
        inertia = config.inertia_factor * config.mass_kg * config.radius_m**2
        self.model = mujoco.MjModel.from_xml_string(f"""
<mujoco>
  <option cone="elliptic" gravity="0 0 -9.81" integrator="implicit"
    timestep="{config.flight_step_s}" density="{config.density}"
    viscosity="{config.viscosity}" iterations="100" tolerance="1e-10"/>
  <worldbody>
    <geom name="ground" type="plane" size="100 100 .1" contype="0" conaffinity="0"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <inertial pos="0 0 0" mass="{config.mass_kg}" diaginertia="{inertia} {inertia} {inertia}"/>
      <geom name="ball_geom" type="sphere" size="{config.radius_m}"
        contype="0" conaffinity="0" fluidshape="ellipsoid"
        fluidcoef="{config.blunt_drag} 0 0 0 {config.magnus}"/>
    </body>
  </worldbody>
  <contact>
    <pair geom1="ball_geom" geom2="ground" condim="3"
      solref="-{config.stiffness} -{config.damping}"
      friction="{config.friction} {config.friction} 0 0 0"
      solimp=".98 .99 .001 .5 2"/>
  </contact>
</mujoco>""")
        self.data = mujoco.MjData(self.model)
        self._velocity = np.empty(6)
        self.reset(State(np.array([0.0, 0.0, 1.0]), np.zeros(3), np.zeros(3)))

    def reset(self, state: State):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = state.xyz
        self.data.qvel[:3] = state.velocity
        # Reset uses identity quaternion, so local and world angular axes agree.
        self.data.qvel[3:] = state.spin
        mujoco.mj_forward(self.model, self.data)
        self.in_contact = bool(self.data.ncon)

    def state(self) -> State:
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, 1, self._velocity, 0
        )
        return State(
            self.data.qpos[:3].copy(),
            self.data.qvel[:3].copy(),
            self._velocity[:3].copy(),
        )

    def step(self) -> tuple[float, bool, bool]:
        cfg, data = self.config, self.data
        # Enter the fine region before even a 75 m/s downward shot reaches ground.
        near = data.qpos[2] - cfg.radius_m < max(
            0.1, abs(data.qvel[2]) * cfg.flight_step_s * 3
        )
        dt = cfg.contact_step_s if near else cfg.flight_step_s
        self.model.opt.timestep = dt
        mujoco.mj_step(self.model, data)
        mujoco.mj_forward(self.model, data)
        contact = bool(data.ncon)  # The scene contains only the named ball-ground pair.
        onset = contact and not self.in_contact
        separation = self.in_contact and not contact
        self.in_contact = contact
        return dt, onset, separation
