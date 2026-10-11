"""Generation settings with small defaults and bounded search attempts, in SI units."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from tennis3d.court.model import BALL_RADIUS_M
from tennis3d.schemas import StrictModel
from tennis3d.synthetic.observation_config import CameraConfig, CorruptionConfig
from tennis3d.synthetic.priors import (
    FamilyPriors,
    RallyPriors,
    ShotPriors,
    mixture_weights,
)

RunLengths = Annotated[list[Annotated[float, Field(gt=0)]], Field(min_length=1)]
RunWeights = Annotated[list[Annotated[float, Field(gt=0, le=1)]], Field(min_length=1)]


class PhysicsConfig(StrictModel):
    mass_kg: float = Field(default=0.057, gt=0)
    radius_m: float = Field(default=BALL_RADIUS_M, gt=0)
    inertia_factor: float = Field(default=0.535, gt=0)
    density: float = Field(default=1.225, ge=0)
    viscosity: float = Field(default=0.000018, ge=0)
    blunt_drag: float = Field(default=0.30, ge=0)
    magnus: float = Field(default=0.80, ge=0)
    friction: float = Field(default=0.6, ge=0)
    stiffness: float = Field(default=1_000_000.0, gt=0)
    damping: float = Field(default=170.0, gt=0)
    flight_step_s: float = Field(default=0.001, gt=0, le=0.001)
    contact_step_s: float = Field(default=0.0000625, gt=0, le=0.000125)
    max_penetration_m: float = Field(default=0.02, gt=0)

    @model_validator(mode="after")
    def step_order(self):
        if self.max_penetration_m >= self.radius_m:
            raise ValueError("max_penetration_m must be less than radius_m")
        if self.contact_step_s > self.flight_step_s:
            raise ValueError("contact_step_s must be <= flight_step_s")
        return self


class SimulationConfig(StrictModel):
    seed: int = Field(default=42, ge=0)
    train_rallies: int = Field(default=4, ge=1)
    val_rallies: int = Field(default=1, ge=1)
    test_rallies: int = Field(default=1, ge=1)
    families_per_split: int = Field(default=1, ge=1)
    train_families: int | None = Field(default=None, ge=1)
    val_families: int | None = Field(default=None, ge=1)
    test_families: int | None = Field(default=None, ge=1)
    shot_priors: ShotPriors | None = None
    rally_priors: RallyPriors | None = None
    family_priors: FamilyPriors | None = None
    camera: CameraConfig = Field(default_factory=CameraConfig)
    corruption: CorruptionConfig = Field(default_factory=CorruptionConfig)
    pool_size: int = Field(default=24, ge=1)
    pool_attempts: int = Field(default=1500, ge=1)
    rally_attempts: int = Field(default=40, ge=1)
    nearest_attempts: int = Field(default=24, ge=1)
    rally_mode: Literal["varied", "diagnostic"] = "varied"
    min_returns: int = Field(default=1, ge=1)
    max_returns: int = Field(default=8, ge=1)
    physics_variation: float = Field(default=0.05, ge=0, le=0.2)
    views_per_rally: int = Field(default=1, ge=1)
    image_width: int = Field(default=1280, ge=64)
    image_height: int = Field(default=720, ge=64)
    timestamp_jitter_s: float = Field(default=0.001, ge=0, le=0.005)
    visible_run_lengths_s: RunLengths | None = None
    missing_run_lengths_s: RunLengths | None = None
    visible_run_weights: RunWeights | None = None
    missing_run_weights: RunWeights | None = None
    noise_px: float = Field(default=1.5, ge=0)
    missing_probability: float = Field(default=0.08, ge=0, le=0.8)
    camera_noise_px: float = Field(default=1.0, ge=0)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)

    def family_count(self, split: str) -> int:
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        return getattr(self, f"{split}_families") or self.families_per_split

    @model_validator(mode="after")
    def ranges(self):
        for split in ("train", "val", "test"):
            if self.family_count(split) > getattr(self, f"{split}_rallies"):
                raise ValueError(f"{split} families cannot exceed its rally count")
        if self.family_priors is not None:
            for component in self.family_priors.components:
                if (
                    component.radius_m is not None
                    and component.radius_m[0] <= self.physics.max_penetration_m
                ):
                    raise ValueError("family radius must exceed max_penetration_m")
        if self.rally_priors is not None and 2 * self.timestamp_jitter_s >= 1 / max(
            self.rally_priors.cadence_fps
        ):
            raise ValueError(
                "timestamp jitter must be less than half the shortest frame interval"
            )
        if (self.visible_run_lengths_s is None) != (self.missing_run_lengths_s is None):
            raise ValueError(
                "provide both visible_run_lengths_s and missing_run_lengths_s"
            )
        for state in ("visible", "missing"):
            weights = getattr(self, f"{state}_run_weights")
            lengths = getattr(self, f"{state}_run_lengths_s")
            if weights is not None:
                if lengths is None or len(weights) != len(lengths):
                    raise ValueError(
                        f"{state}_run_weights must match its duration list"
                    )
                mixture_weights(weights)
        if self.max_returns < self.min_returns:
            raise ValueError("max_returns must be >= min_returns")
        return self
