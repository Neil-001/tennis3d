"""Resolved trainer/evaluator settings. Paths are relative to the working directory."""

from typing import Literal

from pydantic import Field, field_validator, model_validator

from tennis3d.lifting.model import ModelConfig
from tennis3d.schemas import StrictModel


class TrainConfig(StrictModel):
    train_manifest: str = "data/synthetic/train.manifest.json"
    val_manifest: str = "data/synthetic/val.manifest.json"
    model: ModelConfig = Field(default_factory=ModelConfig)
    seed: int = Field(default=42, ge=0)
    batch_size: int = Field(default=2, ge=1)
    min_frames: int = Field(default=20, ge=1, le=250)
    max_frames: int = Field(default=250, ge=1, le=250)
    event_crop_probability: float = Field(default=0.25, ge=0, le=1)
    optimizer: Literal["muon-adam", "adamw"] = "muon-adam"
    adamw_weight_decay: float = Field(default=0.01, ge=0)
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    learning_rate: float = Field(default=0.0001, gt=0)
    ema_decay: float = Field(default=0.999, ge=0, lt=1)
    gradient_clip: float = Field(default=5.0, gt=0)
    max_steps: int = Field(default=1000, ge=1)
    max_train_seconds: float = Field(default=300.0, gt=0)
    max_job_seconds: float = Field(default=172800.0, gt=0, le=172800)
    validation_every: int = Field(default=100, ge=1)
    # None keeps the original validation-triggered recovery checkpoints.
    checkpoint_every: int | None = Field(default=None, ge=1)
    log_every: int = Field(default=1, ge=1)
    patience: int | None = Field(default=None, ge=1)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    artifact_label: str = "development training; accuracy not established"

    @field_validator("adamw_betas", mode="before")
    @classmethod
    def json_betas(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def lengths(self):
        if any(not 0 <= beta < 1 for beta in self.adamw_betas):
            raise ValueError("AdamW betas must be in [0, 1)")
        if (
            self.model.architecture_version == "stage4-lstm-v1"
            and self.optimizer != "adamw"
        ):
            raise ValueError("LSTM training requires explicit AdamW")
        if self.min_frames > self.max_frames:
            raise ValueError("min_frames must not exceed max_frames")
        return self


def _evaluation_modes() -> list[Literal["corrupted", "clean"]]:
    return ["corrupted", "clean"]


class EvaluationConfig(StrictModel):
    manifest: str = "data/synthetic/val.manifest.json"
    modes: list[Literal["corrupted", "clean"]] = Field(
        default_factory=_evaluation_modes
    )
    device: Literal["auto", "cpu", "cuda"] = "auto"

    @model_validator(mode="after")
    def unique_modes(self):
        if not self.modes or len(set(self.modes)) != len(self.modes):
            raise ValueError("evaluation modes must be nonempty and unique")
        return self
