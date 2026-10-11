"""Resolved stage options. Relative paths use the process working directory."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from tennis3d.schemas import StrictModel


class DetectConfig(StrictModel):
    assets_manifest: Path = Path("configs/assets.yaml")
    # auto never selects intel-gpu: it needs the optional intel extra and drivers.
    device: Literal["auto", "cpu", "cuda", "intel-gpu"] = "auto"
    threads: int = Field(default=2, ge=1)
    save_heatmaps: bool = True
    overlay_every: int = Field(default=30, ge=0)
    duplicate_ssim: float = Field(default=0.9995, gt=0, le=1)
    duplicate_motion_error: float = Field(default=0.7, gt=0)
    cut_mean_difference: float = Field(default=55.0, gt=0, le=255)
    candidate_merge_px: float = Field(default=8.0, gt=0)


class CalibrationConfig(StrictModel):
    mode: Literal["moving", "static"] = "moving"
    min_landmarks: int = Field(default=6, ge=6, le=14)
    ransac_diagonal: float = Field(default=0.008, gt=0)
    median_diagonal: float = Field(default=0.003, gt=0)
    p90_diagonal: float = Field(default=0.008, gt=0)
    focal_min_width: float = Field(default=0.3, gt=0)
    focal_max_width: float = Field(default=12.0, gt=0)
    ambiguity_slack_px: float = Field(default=1.0, ge=0)
    ambiguity_height_fraction: float = Field(default=0.35, gt=0)
    ambiguity_ray_degrees: float = Field(default=2.0, gt=0)
    max_gap_s: float = Field(default=0.5, ge=0)
    innovation_gate: float = Field(default=36.0, gt=0)
    reset_after_rejections: int = Field(default=4, ge=1)

    @model_validator(mode="after")
    def ordered_focal_bounds(self):
        if self.focal_min_width >= self.focal_max_width:
            raise ValueError("focal_min_width must be below focal_max_width")
        return self
