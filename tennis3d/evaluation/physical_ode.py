"""SI tennis flight ODE, checked against the generator's MuJoCo sphere forces."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


@dataclass(frozen=True)
class FlightParameters:
    mass_kg: float = 0.057
    radius_m: float = 0.0335
    density_kg_m3: float = 1.225
    viscosity_pa_s: float = 0.000018
    blunt_drag: float = 0.30
    magnus: float = 0.80
    inertia_factor: float = 0.535
    gravity_m_s2: float = 9.81

    def __post_init__(self):
        values = asdict(self)
        if any(not np.isfinite(v) or v < 0 for v in values.values()) or any(
            values[k] == 0 for k in ("mass_kg", "radius_m", "inertia_factor")
        ):
            raise ValueError(
                "flight parameters must be finite with positive mass/radius/inertia"
            )

    @property
    def coefficients(self) -> dict:
        r, rho, mu = self.radius_m, self.density_kg_m3, self.viscosity_pa_s
        return {
            "quadratic_drag_kg_m": rho * np.pi * r**2 * self.blunt_drag,
            "linear_drag_kg_s": 6 * np.pi * mu * r,
            "magnus_kg": rho * (4 / 3) * np.pi * r**3 * (self.magnus - 0.5),
            "spin_decay_s_inv": 8
            * np.pi
            * mu
            * r
            / (self.inertia_factor * self.mass_kg),
        }


@dataclass(frozen=True)
class FitConfig:
    min_frames: int = 8
    min_observed_frames: int = 4
    min_duration_s: float = 0.12
    max_time_gap_s: float = 0.15
    contact_guard_s: float = 0.01
    loss_scale_m: float = 0.03
    max_position_m: float = 1000.0
    max_velocity_m_s: float = 100.0
    max_spin_rad_s: float = 1500.0
    max_nfev: int = 200
    rtol: float = 1e-8
    atol: float = 1e-10
    max_step_s: float = 0.05
    # Segmentation settings, not residual acceptance thresholds.
    automatic_jump_m_s: float = 8.0
    automatic_merge_s: float = 0.08

    def __post_init__(self):
        values = asdict(self)
        if any(not np.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError("fit settings must be positive and finite")
        if any(
            type(values[k]) is not int
            for k in ("min_frames", "min_observed_frames", "max_nfev")
        ):
            raise ValueError("fit counts must be integers")
        if self.min_frames < 4 or self.min_observed_frames > self.min_frames:
            raise ValueError(
                "need at least four frames and observed minimum <= frame minimum"
            )


DEFAULT_PHYSICS = FlightParameters()
DEFAULT_CONFIG = FitConfig()


def flight_rhs(state, physics: FlightParameters) -> np.ndarray:
    """World XYZ up; spin world rad/s; no wind, bounce or racket force."""
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (9,) or not np.isfinite(state).all():
        raise ValueError("initial flight state must be nine finite SI values")
    return _derivative(state, physics, physics.coefficients)


def _derivative(state, physics, coefficients):
    v, w = state[3:6], state[6:9]
    c = coefficients
    acceleration = (
        -(c["quadratic_drag_kg_m"] * np.linalg.norm(v) + c["linear_drag_kg_s"]) * v
        + c["magnus_kg"] * np.cross(w, v)
    ) / physics.mass_kg
    acceleration[2] -= physics.gravity_m_s2
    return np.r_[v, acceleration, -c["spin_decay_s_inv"] * w]


def integrate_flight(t_s, state, physics=DEFAULT_PHYSICS, config=DEFAULT_CONFIG):
    """Integrate at relative seconds from state origin, never round to frame cadence."""
    times = np.asarray(t_s, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    flight_rhs(state, physics)  # Validate even zero-duration requests.
    if (
        times.ndim != 1
        or not len(times)
        or not np.isfinite(times).all()
        or times[0] < 0
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("flight times must be increasing finite nonnegative seconds")
    if times[-1] == 0:
        return state[None, :].copy()
    coefficients = physics.coefficients
    solution = solve_ivp(
        lambda _, y: _derivative(y, physics, coefficients),
        (0.0, float(times[-1])),
        state,
        t_eval=times,
        method="DOP853",
        rtol=config.rtol,
        atol=config.atol,
        max_step=config.max_step_s,
    )
    if (
        not solution.success
        or solution.y.shape != (9, len(times))
        or not np.isfinite(solution.y).all()
    ):
        raise RuntimeError(f"flight integration failed: {solution.message}")
    return solution.y.T


def fit_flight(t_s, xyz_m, physics=DEFAULT_PHYSICS, config=DEFAULT_CONFIG) -> dict:
    """Robust XYZ-only fit of p0/v0/world spin. Truth and UV never enter the objective."""
    times = np.asarray(t_s, dtype=np.float64)
    xyz = np.asarray(xyz_m, dtype=np.float64)
    if (
        times.ndim != 1
        or len(times) < 4
        or xyz.shape != (len(times), 3)
        or not np.isfinite(xyz).all()
    ):
        raise ValueError("fit needs at least four finite XYZ samples")
    times = times - times[0]
    # Enforce the same time validation as the public integrator.
    integrate_flight(times, np.r_[xyz[0], np.zeros(6)], physics, config)
    design = np.stack([np.ones(len(times)), times, times**2 / 2], axis=1)
    p, v, a = np.linalg.lstsq(design, xyz, rcond=None)[0]
    c = physics.coefficients
    aero = (
        a
        + [0, 0, physics.gravity_m_s2]
        + (c["quadratic_drag_kg_m"] * np.linalg.norm(v) + c["linear_drag_kg_s"])
        * v
        / physics.mass_kg
    )
    spin = np.zeros(3)
    if abs(c["magnus_kg"]) > 1e-15 and np.dot(v, v) > 1e-8:
        spin = physics.mass_kg * np.cross(v, aero) / (c["magnus_kg"] * np.dot(v, v))
    bounds = np.repeat(
        [config.max_position_m, config.max_velocity_m_s, config.max_spin_rad_s], 3
    )
    scale = np.repeat([10.0, 30.0, 200.0], 3)
    starts = [np.r_[p, v, np.zeros(3)], np.r_[p, v, spin]]
    if np.linalg.norm(spin) < 1e-8:
        starts = starts[:1]
    candidates, errors = [], []
    for start in starts:
        try:
            candidate = least_squares(
                lambda x: (
                    integrate_flight(times, x, physics, config)[:, :3] - xyz
                ).ravel(),
                np.clip(start, -bounds * 0.999, bounds * 0.999),
                bounds=(-bounds, bounds),
                x_scale=scale,
                loss="soft_l1",
                f_scale=config.loss_scale_m,
                max_nfev=config.max_nfev,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
            candidates.append(candidate)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            errors.append(str(exc))
    converged = [c for c in candidates if c.success]
    if not converged:
        return {
            "status": "failed",
            "reason": "optimizer_failed",
            "attempt_messages": errors + [str(c.message) for c in candidates],
        }
    best = min(converged, key=lambda c: c.cost)
    fitted = integrate_flight(times, best.x, physics, config)[:, :3]
    error = np.linalg.norm(fitted - xyz, axis=1)
    singular = np.linalg.svd(best.jac * scale, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-6)) if singular[0] > 0 else 0
    return {
        "status": "fitted",
        "reason": None,
        "initial_state": {
            "position_m": best.x[:3].tolist(),
            "velocity_m_s": best.x[3:6].tolist(),
            "spin_rad_s": best.x[6:9].tolist(),
        },
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "max_discrepancy_m": float(error.max()),
        "fitted_xyz_m": fitted.tolist(),
        "optimizer": {
            "nfev": int(best.nfev),
            "message": str(best.message),
            "robust_cost_m2": float(best.cost),
            "active_bounds": best.active_mask.tolist(),
            "attempts": len(starts),
            "converged_attempts": len(converged),
            "scaled_robust_jacobian_rank": rank,
            "scaled_robust_jacobian_singular_values": singular.tolist(),
        },
        "limits": {
            "bound_reached": bool(best.active_mask.any()),
            "state_weakly_identified": rank < 9,
        },
    }
