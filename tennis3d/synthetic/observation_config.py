"""Explicit engineering priors for broadcast geometry and observation errors."""

from typing import Annotated

from pydantic import Field, model_validator

from tennis3d.schemas import StrictModel
from tennis3d.synthetic.priors import Bounds, check_components

Probability = Annotated[float, Field(ge=0, le=1)]


class CameraComponent(StrictModel):
    name: str = "elevated_baseline"
    weight: float = Field(default=1, gt=0, le=1)
    x_m: Bounds = Field(default_factory=lambda: [-2.0, 2.0])
    y_m: Bounds = Field(default_factory=lambda: [-31.0, -27.0])
    z_m: Bounds = Field(default_factory=lambda: [12.0, 16.0])
    focal_over_width: Bounds = Field(default_factory=lambda: [0.85, 1.05])
    target_y_m: float = 1.0
    target_z_m: float = 0.4
    angular_frequency_rad_s: float = Field(default=0.5, ge=0)
    x_motion_m: float = Field(default=0.06, ge=0)
    z_motion_m: float = Field(default=0.03, ge=0)
    target_x_motion_m: float = Field(default=0.18, ge=0)
    zoom_fraction: float = Field(default=0.015, ge=0, lt=1)

    @model_validator(mode="after")
    def geometry(self):
        if self.focal_over_width[0] <= 0:
            raise ValueError("focal_over_width must be positive")
        if self.z_m[0] <= self.z_motion_m:
            raise ValueError("camera must remain above ground throughout motion")
        if self.y_m[0] <= self.target_y_m <= self.y_m[1]:
            raise ValueError(
                "camera Y range must exclude target Y to avoid singular look-at"
            )
        return self


class CameraConfig(StrictModel):
    components: Annotated[list[CameraComponent], Field(min_length=1)] = Field(
        default_factory=lambda: [CameraComponent()]
    )
    principal_point_jitter_fraction: float = Field(default=0.15, ge=0)
    # A bounded per-view intrinsic error, not measured calibration error.
    focal_error_fraction: float = Field(default=0, ge=0, lt=1)

    @model_validator(mode="after")
    def mixture(self):
        check_components(self.components)
        return self


class CorruptionConfig(StrictModel):
    initial_missing_probability: Probability = 0.0
    occlusion_start_fraction: Bounds = Field(default_factory=lambda: [0.2, 0.7])
    occlusion_duration_s: Bounds = Field(default_factory=lambda: [0.12, 0.35])
    false_candidate_probability: Probability = 0.02
    false_candidate_noise_px: float = Field(default=35, ge=0)
    motion_bias_fraction: Bounds = Field(default_factory=lambda: [0.0, 0.15])
    motion_clip_px: float = Field(default=10, ge=0)
    confidence: Bounds = Field(default_factory=lambda: [0.65, 1.0])
    localization_correlation_s: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def ranges(self):
        for key in (
            "occlusion_start_fraction",
            "occlusion_duration_s",
            "motion_bias_fraction",
            "confidence",
        ):
            if getattr(self, key)[0] < 0:
                raise ValueError(f"{key} must be nonnegative")
        for key in ("occlusion_start_fraction", "confidence", "motion_bias_fraction"):
            if getattr(self, key)[1] > 1:
                raise ValueError(f"{key} must lie in [0, 1]")
        return self
