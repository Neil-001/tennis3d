"""Explicit engineering proposal mixtures. All bounds use the named SI units.

Components select position, velocity and spin together; coordinates are uniform
within a component. Mixture weights describe proposals, before physical rejection.
"""

from math import isclose
from typing import Annotated, Literal

import numpy as np
from pydantic import AfterValidator, Field, model_validator

from tennis3d.schemas import StrictModel


def ordered(value: list[float]) -> list[float]:
    if value[0] > value[1]:
        raise ValueError("range lower bound must be <= upper bound")
    return value


Bounds = Annotated[
    list[float], Field(min_length=2, max_length=2), AfterValidator(ordered)
]
AxesBounds = Annotated[list[Bounds], Field(min_length=3, max_length=3)]
ShotKind = Literal["toss", "serve", "return", "lob", "slice", "volley", "out"]
ReturnKind = Literal["return", "lob", "slice", "volley"]


def default_return_kinds() -> list[ReturnKind]:
    return ["return", "lob", "slice", "volley"]


def mixture_weights(weights: list[float]) -> None:
    if (
        not weights
        or any(weight < 0 for weight in weights)
        or not isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-9)
    ):
        raise ValueError("mixture weights must be nonnegative and sum to one")


def check_components(components) -> None:
    mixture_weights([component.weight for component in components])
    if len({component.name for component in components}) != len(components):
        raise ValueError("component names must be unique within a mixture")


def choose(rng: np.random.Generator, components):
    return components[
        int(rng.choice(len(components), p=[c.weight for c in components]))
    ]


class ShotComponent(StrictModel):
    name: str = Field(min_length=1)
    weight: float = Field(gt=0, le=1)
    position_m: AxesBounds
    velocity_m_s: AxesBounds
    spin_rad_s: AxesBounds
    aim_x_gain_s_inv: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def canonical_position(self):
        if self.position_m[1][1] >= 0 or self.position_m[2][0] <= 0:
            raise ValueError("canonical proposals require y < 0 and z > 0 metres")
        return self


class ShotMixture(StrictModel):
    components: list[ShotComponent] = Field(min_length=1)

    @model_validator(mode="after")
    def mixture(self):
        check_components(self.components)
        return self


class ShotPriors(StrictModel):
    # Omitted kinds retain the smoke proposals and their existing RNG draws.
    proposals: dict[ShotKind, ShotMixture] = Field(default_factory=dict)


class RallyPriors(StrictModel):
    cadence_fps: list[Annotated[int, Field(ge=1, le=240)]] = Field(
        default_factory=lambda: [25, 30, 50, 60], min_length=1
    )
    cadence_weights: list[Annotated[float, Field(ge=0, le=1)]] = Field(
        default_factory=lambda: [0.25, 0.25, 0.25, 0.25], min_length=1
    )
    return_kinds: list[ReturnKind] = Field(
        default_factory=default_return_kinds, min_length=1
    )
    return_weights: list[Annotated[float, Field(ge=0, le=1)]] = Field(
        default_factory=lambda: [0.55, 0.15, 0.2, 0.1], min_length=1
    )
    terminal_out_probability: float = Field(default=0.3, ge=0, le=1)
    toss_offset_s: Bounds = Field(default_factory=lambda: [-0.07, 0.07])

    @model_validator(mode="after")
    def mixtures(self):
        for values, weights in (
            (self.cadence_fps, self.cadence_weights),
            (self.return_kinds, self.return_weights),
        ):
            if len(values) != len(weights) or len(set(values)) != len(values):
                raise ValueError("mixture values must be unique and match weight count")
            mixture_weights(weights)
        if self.toss_offset_s[0] < -0.2 or self.toss_offset_s[1] > 0.2:
            raise ValueError("toss_offset_s must lie within [-0.2, 0.2] seconds")
        return self


class FamilyComponent(StrictModel):
    name: str = Field(min_length=1)
    weight: float = Field(gt=0, le=1)
    mass_kg: Bounds | None = None
    radius_m: Bounds | None = None
    inertia_factor: Bounds | None = None
    density: Bounds | None = None  # kg/m^3, same parameter name as PhysicsConfig
    blunt_drag: Bounds | None = None  # MuJoCo coefficient, conventional C_D / 2
    magnus: Bounds | None = None
    friction: Bounds | None = None
    damping: Bounds | None = None

    @model_validator(mode="after")
    def physical_bounds(self):
        for name in ("mass_kg", "radius_m", "inertia_factor", "damping"):
            value = getattr(self, name)
            if value is not None and value[0] <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("density", "blunt_drag", "magnus", "friction"):
            value = getattr(self, name)
            if value is not None and value[0] < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name, lower, upper in (
            ("mass_kg", 0.056, 0.0594),
            ("radius_m", 0.0327, 0.0343),
        ):
            value = getattr(self, name)
            if value is not None and (value[0] < lower or value[1] > upper):
                raise ValueError(f"{name} family bounds must satisfy ITF Type 2 limits")
        return self


class FamilyPriors(StrictModel):
    components: list[FamilyComponent] = Field(min_length=1)

    @model_validator(mode="after")
    def mixture(self):
        check_components(self.components)
        return self
