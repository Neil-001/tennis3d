"""Freeze source-only clip identities before any final-model output inspection."""

from fractions import Fraction
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from tennis3d.artifacts import file_sha256, read_json
from tennis3d.schemas import SHA256, PositiveInt, StrictModel


class Clip(StrictModel):
    clip_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    recording_group: str = Field(min_length=1)
    recording_evidence: str = Field(min_length=1)
    distinct_recording_verified: bool
    source_url: str
    source_revision: str | None = None
    source_sha256: SHA256 | None = None
    source_bytes: PositiveInt | None = None
    source_path: str | None = None
    original_source_sha256: SHA256 | None = None
    original_source_bytes: PositiveInt | None = None
    original_source_frame_start: int | None = Field(default=None, ge=0)
    original_source_frame_end_exclusive: PositiveInt | None = None
    preparation_note: str = "Unchanged uploaded input"
    source_frame_start: int = Field(default=0, ge=0)
    frame_count: PositiveInt | None = None
    width: PositiveInt | None = None
    height: PositiveInt | None = None
    time_basis: Literal["source_pts", "unavailable"]
    timeline_path: str | None = None
    timeline_sha256: SHA256 | None = None
    timing_note: str
    local_use: Literal["permitted", "unknown", "unavailable"]
    redistribution: Literal["permitted", "restricted", "unknown"]
    rights_evidence: str
    labels_path: str | None = None
    labels_sha256: SHA256 | None = None
    label_status: Literal["normalized_source_pts", "unavailable"]
    detector_training_independence: Literal["verified", "unknown", "overlap"]
    categories: list[Literal["static", "modest_motion", "occlusion", "serve", "lob"]]
    category_evidence: dict[str, str] = Field(default_factory=dict)
    eligible: bool
    barriers: list[str]

    @model_validator(mode="after")
    def eligibility(self):
        if (self.labels_path is None) != (self.labels_sha256 is None):
            raise ValueError("label file and hash must be supplied together")
        if (self.label_status == "unavailable") != (self.labels_path is None):
            raise ValueError("label status and availability disagree")
        if self.eligible and (
            self.barriers
            or self.local_use != "permitted"
            or not self.distinct_recording_verified
            or self.time_basis != "source_pts"
            or any(
                value is None
                for value in (
                    self.source_sha256,
                    self.source_bytes,
                    self.source_path,
                    self.frame_count,
                    self.width,
                    self.height,
                    self.timeline_path,
                    self.timeline_sha256,
                )
            )
        ):
            raise ValueError(
                "eligible clip requires audited access, recording, identity and source PTS"
            )
        if not self.eligible and not self.barriers:
            raise ValueError("ineligible clips require concrete barriers")
        if set(self.category_evidence) != set(self.categories) or any(
            not note for note in self.category_evidence.values()
        ):
            raise ValueError("each source-reviewed category requires its evidence")
        originals = [
            self.original_source_sha256,
            self.original_source_bytes,
            self.original_source_frame_start,
            self.original_source_frame_end_exclusive,
        ]
        if any(value is not None for value in originals) and not all(
            value is not None for value in originals
        ):
            raise ValueError(
                "prepared inputs require complete original source identity and interval"
            )
        if (
            self.original_source_frame_start is not None
            and self.original_source_frame_end_exclusive is not None
            and self.original_source_frame_end_exclusive
            <= self.original_source_frame_start
        ):
            raise ValueError("original source interval must be nonempty")
        if len(self.categories) != len(set(self.categories)):
            raise ValueError("duplicate source-reviewed category")
        return self


class Inventory(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    frozen_utc: str
    intended_role: Literal["resolution_stress", "representative_end_to_end"]
    selection_rule: str
    output_inspected_before_freeze: Literal[False] = False
    excluded_development_recording_groups: list[str]
    excluded_development_source_sha256: list[SHA256]
    exclusion_evidence: str
    clips: list[Clip]
    limitations: list[str]

    @model_validator(mode="after")
    def separation(self):
        if len({clip.clip_id for clip in self.clips}) != len(self.clips):
            raise ValueError("clip IDs must be unique")
        for clip in self.clips:
            if clip.eligible and (
                clip.recording_group in self.excluded_development_recording_groups
                or clip.source_sha256 in self.excluded_development_source_sha256
            ):
                raise ValueError("development recordings cannot enter final evaluation")
        return self

    def summary(self):
        eligible = [clip for clip in self.clips if clip.eligible]
        groups = {clip.recording_group for clip in eligible}
        categories = {category for clip in eligible for category in clip.categories}
        missing = sorted(
            {"static", "modest_motion", "occlusion", "serve", "lob"} - categories
        )
        return {
            "preselected_records": len(self.clips),
            "eligible_clips": len(eligible),
            "distinct_recordings": len(groups),
            "missing_categories": missing,
            "intended_role": self.intended_role,
            "required_gallery_ready": self.intended_role == "representative_end_to_end"
            and len(eligible) >= 20
            and len(groups) >= 3
            and not missing,
            "barriers": {
                clip.clip_id: clip.barriers for clip in self.clips if clip.barriers
            },
        }


def read_inventory(path):
    return Inventory.model_validate(read_json(path))


def verify_clip(clip: Clip):
    """Check input bytes and exact integer PTS mapping, never model outputs."""
    if not clip.eligible:
        raise ValueError("clip is not eligible")
    assert clip.source_path is not None and clip.timeline_path is not None
    source = Path(clip.source_path)
    if (
        source.stat().st_size != clip.source_bytes
        or file_sha256(source) != clip.source_sha256
    ):
        raise ValueError("source video identity changed")
    path = Path(clip.timeline_path)
    if file_sha256(path) != clip.timeline_sha256:
        raise ValueError("source timeline changed")
    timeline = read_json(path)
    if timeline["source_sha256"] != clip.source_sha256:
        raise ValueError("timeline belongs to another source")
    rows = timeline["frames"]
    if len(rows) != clip.frame_count or timeline["image_size"] != [
        clip.width,
        clip.height,
    ]:
        raise ValueError("source timeline dimensions/count differ")
    previous = None
    for index, row in enumerate(rows):
        numerator, denominator = row["time_base"]
        if (
            type(row["pts"]) is not int
            or type(numerator) is not int
            or numerator <= 0
            or type(denominator) is not int
            or denominator <= 0
            or type(row["frame_index"]) is not int
            or row["frame_index"] != index + clip.source_frame_start
        ):
            raise ValueError(
                "timeline requires exact source indices and integer PTS/time bases"
            )
        current = Fraction(row["pts"] * numerator, denominator)
        if previous is not None and current <= previous:
            raise ValueError("source PTS must strictly increase")
        previous = current
    if (
        clip.labels_path is not None
        and file_sha256(clip.labels_path) != clip.labels_sha256
    ):
        raise ValueError("normalized annotation manifest changed")
    return timeline
