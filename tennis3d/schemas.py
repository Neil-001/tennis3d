"""Strict public result contract, schema 1.0. Times are seconds, XYZ metres."""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tennis3d.court.model import (
    DOUBLES_WIDTH_M,
    LENGTH_M,
    SERVICE_LINE_DISTANCE_M,
    SINGLES_WIDTH_M,
)

SCHEMA_VERSION = "1.0"
SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonnegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
Fraction = Annotated[float, Field(ge=0, le=1)]
Observation = Literal["detected", "missing", "out_of_frame", "ambiguous", "duplicate"]
Reconstruction = Literal["estimated", "inferred", "unavailable"]
ReasonCode = Literal[
    "long_gap",
    "invalid_camera",
    "ambiguous_track",
    "window_disagreement",
    "implausible_prediction",
    "unsupported_shot",
    "decoder_error",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        frozen=True,
    )


class Source(StrictModel):
    name: Annotated[str, Field(min_length=1)]
    sha256: SHA256
    width: PositiveInt
    height: PositiveInt
    frame_count: NonnegativeInt
    time_origin_s: float
    # Original decoder time base, numerator/denominator. Required even for fixtures.
    time_base: Annotated[list[PositiveInt], Field(min_length=2, max_length=2)]


class Axes(StrictModel):
    x: Literal["across_court"] = "across_court"
    y: Literal["along_court"] = "along_court"
    z: Literal["up"] = "up"


class Coordinates(StrictModel):
    units: Literal["m"] = "m"
    origin: Literal["court_center_ground"] = "court_center_ground"
    axes: Axes = Field(default_factory=Axes)
    near_baseline: Literal["negative_y"] = "negative_y"
    point: Literal["ball_center"] = "ball_center"


def _fixed(value: float):
    return Field(default=value, ge=value, le=value)


class Court(StrictModel):
    length_m: float = _fixed(LENGTH_M)
    singles_width_m: float = _fixed(SINGLES_WIDTH_M)
    doubles_width_m: float = _fixed(DOUBLES_WIDTH_M)
    service_line_distance_m: float = _fixed(SERVICE_LINE_DISTANCE_M)


class Frame(StrictModel):
    frame_index: NonnegativeInt
    t_s: Annotated[float, Field(ge=0)]
    xyz_m: Annotated[list[float], Field(min_length=3, max_length=3)] | None
    observation: Observation
    reconstruction: Reconstruction
    camera_valid: bool
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    duplicate_of: NonnegativeInt | None = None

    @model_validator(mode="after")
    def consistent_state(self) -> Self:
        unavailable = self.reconstruction == "unavailable"
        if unavailable != (self.xyz_m is None):
            raise ValueError(
                "unavailable requires null XYZ; available requires three finite coordinates"
            )
        if not self.camera_valid and (
            not unavailable or "invalid_camera" not in self.reason_codes
        ):
            raise ValueError(
                "invalid camera requires unavailable XYZ and invalid_camera reason"
            )
        if self.camera_valid and "invalid_camera" in self.reason_codes:
            raise ValueError("invalid_camera contradicts camera_valid")
        if unavailable != bool(self.reason_codes):
            raise ValueError("reason_codes must be nonempty exactly when unavailable")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("reason_codes must be unique")
        if not unavailable and (
            (self.observation == "detected") != (self.reconstruction == "estimated")
        ):
            raise ValueError(
                "estimated requires detected; inferred requires a non-detected observation"
            )
        if (self.observation == "duplicate") != (self.duplicate_of is not None):
            raise ValueError(
                "duplicate observations require duplicate_of; other observations forbid it"
            )
        if self.duplicate_of is not None and self.duplicate_of >= self.frame_index:
            raise ValueError("duplicate_of must reference an earlier source frame")
        return self


class Summary(StrictModel):
    valid_frame_fraction: Fraction
    detected_frame_fraction: Fraction


class Provenance(StrictModel):
    model_label: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    lifter_checkpoint_sha256: SHA256 | None
    config_sha256: SHA256
    calibration_mode: Literal[
        "estimated_static", "estimated_per_frame", "synthetic", "unavailable"
    ]
    calibration_assumptions: list[str]
    trajectory_kind: Literal["model_estimate", "synthetic_ground_truth", "fixture"] = (
        "model_estimate"
    )
    code_revision: str | None = None


PixelPoint = Annotated[list[float], Field(min_length=2, max_length=2)]


class ProjectionFrame(StrictModel):
    frame_index: NonnegativeInt
    t_s: Annotated[float, Field(ge=0)]
    ball_clean_uv: PixelPoint | None
    ball_observed_uv: PixelPoint | None
    court_uv: Annotated[list[PixelPoint | None], Field(min_length=14, max_length=14)]


class SyntheticEvent(StrictModel):
    kind: Literal["hit", "bounce", "net", "out"]
    t_s: Annotated[float, Field(ge=0)]
    xyz_m: Annotated[list[float], Field(min_length=3, max_length=3)]


class SyntheticDiagnostics(StrictModel):
    kind: Literal["synthetic"] = "synthetic"
    frames: list[ProjectionFrame]
    events: list[SyntheticEvent] = Field(default_factory=list)


class Result(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    status: Literal["ok", "partial", "failed"]
    source: Source
    coordinates: Coordinates = Field(default_factory=Coordinates)
    court: Court = Field(default_factory=Court)
    frames: list[Frame]
    summary: Summary
    provenance: Provenance
    # Input-contract failures can invalidate a run despite locally available XYZ.
    failure_reasons: list[ReasonCode] = Field(default_factory=list)
    synthetic_diagnostics: SyntheticDiagnostics | None = None

    @model_validator(mode="after")
    def consistent_timeline(self) -> Self:
        n = self.source.frame_count
        if len(self.frames) != n:
            raise ValueError("one output record is required per declared source frame")
        for i, frame in enumerate(self.frames):
            if frame.frame_index != i:
                raise ValueError(
                    "frame indices must be contiguous in source order from zero"
                )
            if i == 0 and frame.t_s != 0:
                raise ValueError("first normalized timestamp must be zero")
            if i and frame.t_s <= self.frames[i - 1].t_s:
                raise ValueError("timestamps must be strictly increasing")
        valid = sum(f.xyz_m is not None for f in self.frames)
        detected = sum(f.observation == "detected" for f in self.frames)
        expected = (valid / n, detected / n) if n else (0.0, 0.0)
        actual = (
            self.summary.valid_frame_fraction,
            self.summary.detected_frame_fraction,
        )
        if any(abs(a - e) > 1e-9 for a, e in zip(actual, expected)):
            raise ValueError(
                "summary fractions must use the declared source-frame denominator"
            )
        if self.failure_reasons or not valid:
            status = "failed"
        else:
            status = "ok" if valid == n else "partial"
        if self.status != status:
            raise ValueError(
                f"status must be {status} for these frames/failure reasons"
            )
        return self

    @model_validator(mode="after")
    def consistent_diagnostics(self) -> Self:
        diagnostics = self.synthetic_diagnostics
        if diagnostics is None:
            return self
        if self.provenance.calibration_mode != "synthetic" or (
            self.provenance.trajectory_kind not in ("synthetic_ground_truth", "fixture")
        ):
            raise ValueError("synthetic diagnostics require synthetic provenance")
        if len(diagnostics.frames) != len(self.frames) or any(
            (a.frame_index, a.t_s) != (b.frame_index, b.t_s)
            for a, b in zip(diagnostics.frames, self.frames)
        ):
            raise ValueError("diagnostic frames must exactly match the result timeline")
        previous = 0.0
        for event in diagnostics.events:
            if not self.frames or not previous <= event.t_s <= self.frames[-1].t_s:
                raise ValueError("events must be ordered within the result timeline")
            previous = event.t_s
        return self


def read_result(path: str | Path) -> Result:
    from tennis3d.artifacts import read_json

    return Result.model_validate(read_json(path))


def write_result(path: str | Path, result: Result, *, overwrite: bool = False) -> None:
    from tennis3d.artifacts import write_json

    # Revalidate in case a caller mutated a nested list after construction.
    checked = Result.model_validate(result.model_dump())
    write_json(path, checked.model_dump(mode="json"), overwrite=overwrite)
