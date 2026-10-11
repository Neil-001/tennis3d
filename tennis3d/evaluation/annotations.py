"""Normalized labels and adapter for the audited TrackNet five-column CSV."""

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from tennis3d.artifacts import file_sha256
from tennis3d.schemas import SHA256, NonnegativeInt, PositiveInt, StrictModel


class Annotation(StrictModel):
    frame_index: NonnegativeInt
    image_name: str
    t_s: Annotated[float, Field(ge=0)] | None
    visibility: Literal[0, 1, 2, 3]
    uv: Annotated[list[float], Field(min_length=2, max_length=2)] | None
    event: Literal["flight", "hit", "bounce"] | None = None

    @model_validator(mode="after")
    def location(self) -> Self:
        if (self.visibility == 0) != (self.uv is None):
            raise ValueError("visibility 0 has no location; classes 1..3 require one")
        return self


class AnnotationManifest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    source_sha256: SHA256
    labels_sha256: SHA256
    clip_id: str
    recording_group: str
    split: Literal["development", "validation", "test"]
    width: PositiveInt
    height: PositiveInt
    source_frame_count: PositiveInt
    time_basis: Literal["source_pts", "declared_nominal", "unavailable"]
    timing_note: str
    pixel_convention: str
    rights_and_split_note: str
    annotations: list[Annotation]

    @model_validator(mode="after")
    def mapping(self) -> Self:
        previous_index = -1
        previous_time = -1.0
        for row in self.annotations:
            if not previous_index < row.frame_index < self.source_frame_count:
                raise ValueError(
                    "annotations must have unique ascending mapped source indices"
                )
            previous_index = row.frame_index
            if (self.time_basis == "unavailable") != (row.t_s is None):
                raise ValueError("annotation times must agree with declared time_basis")
            if row.t_s is not None:
                if row.t_s <= previous_time:
                    raise ValueError("mapped annotation times must strictly increase")
                previous_time = row.t_s
            if row.uv is not None and not (
                0 <= row.uv[0] < self.width and 0 <= row.uv[1] < self.height
            ):
                raise ValueError("annotation pixels must be inside the original image")
        return self


def read_tracknet_csv(
    path: Path, *, image_mapping: Mapping[str, tuple[int, float | None]], metadata: dict
) -> AnnotationManifest:
    """Require an explicit image basename -> (decoded index, time) mapping.

    CSV order never defines source order. Nominal cadence must be labelled as
    such in metadata, not passed off as recovered PTS. Repeated event rows stay
    separate; the caller must declare any later event-grouping convention.
    """
    expected = ["file name", "visibility", "x-coordinate", "y-coordinate", "status"]
    rows = []
    seen = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected:
            raise ValueError(
                "unsupported TrackNet CSV header; audit this release first"
            )
        for item in reader:
            name = item["file name"]
            if name in seen or name not in image_mapping:
                raise ValueError(f"duplicate or unmapped annotation image: {name}")
            seen.add(name)
            visibility = int(item["visibility"])
            xy = [item["x-coordinate"].strip(), item["y-coordinate"].strip()]
            uv = None if not any(xy) else [float(value) for value in xy]
            status = item["status"].strip()
            events = {"": None, "0": "flight", "1": "hit", "2": "bounce"}
            if status not in events:
                raise ValueError(f"unknown TrackNet event: {status}")
            index, time = image_mapping[name]
            rows.append(
                Annotation.model_validate(
                    {
                        "frame_index": index,
                        "image_name": name,
                        "t_s": time,
                        "visibility": visibility,
                        "uv": uv,
                        "event": events[status],
                    }
                )
            )
    return AnnotationManifest(
        **metadata,
        labels_sha256=file_sha256(path),
        annotations=sorted(rows, key=lambda row: row.frame_index),
    )
