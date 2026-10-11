"""Local reconstruction options; paths are relative to the command working directory."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from tennis3d.inference.quality import QualityConfig
from tennis3d.perception.config import CalibrationConfig, DetectConfig
from tennis3d.schemas import StrictModel


def lifting_device(device):
    """Only detection runs on Intel graphics; the small lifter stays on the CPU."""
    return "cpu" if device == "intel-gpu" else device


class LiftConfig(StrictModel):
    checkpoint: Path = Path("checkpoints/best.pt")
    device: Literal["auto", "cpu", "cuda"] = "auto"
    model_label: str = Field(default="Unvalidated lifter", min_length=1, max_length=200)
    threads: int = Field(default=2, ge=1)
    max_frames: int = Field(default=250, ge=1, le=250)
    stride: int = Field(default=125, ge=1, le=250)

    @model_validator(mode="after")
    def window_order(self):
        if self.stride > self.max_frames:
            raise ValueError("stride cannot exceed max_frames")
        return self


class ReconstructionConfig(StrictModel):
    detection: DetectConfig = Field(default_factory=DetectConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    lifting: LiftConfig = Field(default_factory=LiftConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
