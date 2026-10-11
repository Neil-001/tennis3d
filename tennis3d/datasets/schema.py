"""Small strict metadata models for numeric dataset format 2.0."""

from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from tennis3d.artifacts import json_sha256
from tennis3d.schemas import SHA256, NonnegativeInt, PositiveInt, StrictModel

Split = Literal["train", "val", "test"]
Identity = Annotated[str, Field(min_length=1)]
Seed = Annotated[int, Field(ge=0, le=2**63 - 1)]


class Artifact(StrictModel):
    path: str
    sha256: SHA256
    size_bytes: PositiveInt

    @field_validator("path")
    @classmethod
    def relative_path(cls, value: str) -> str:
        p = PurePosixPath(value)
        if (
            not value
            or p.is_absolute()
            or ".." in p.parts
            or "\\" in value
            or ":" in value
            or p.as_posix() != value
            or value == "."
        ):
            raise ValueError("artifact path must be a normalized relative POSIX path")
        return value


class Pool(StrictModel):
    family_id: Identity
    split: Split
    sha256: SHA256
    config: dict
    attempts: dict[str, int]


class Provenance(StrictModel):
    generator_version: Identity
    generator_sha256: SHA256
    code_revision: Identity
    runtime_versions: dict[str, str]
    config_sha256: SHA256
    config: dict
    seed: Seed
    pools: list[Pool]

    @model_validator(mode="after")
    def check_config(self) -> Self:
        if json_sha256(self.config) != self.config_sha256:
            raise ValueError("resolved config checksum mismatch")
        if not self.runtime_versions:
            raise ValueError("runtime versions are required")
        families = [p.family_id for p in self.pools]
        if len(families) != len(set(families)):
            raise ValueError("duplicate family provenance")
        return self

    def check_rally(self, rally: "RallyMetadata") -> None:
        pool = next((p for p in self.pools if p.family_id == rally.family_id), None)
        if pool is None or (pool.split, pool.sha256) != (
            rally.split,
            rally.pool_sha256,
        ):
            raise ValueError("rally family/pool provenance mismatch")


class RallyMetadata(StrictModel):
    rally_id: Identity
    family_id: Identity
    pool_sha256: SHA256
    split: Split
    seed: Seed
    fps: PositiveInt
    attempts: PositiveInt


class ViewMetadata(StrictModel):
    view_id: NonnegativeInt
    view_seed: Seed
    physical_time_origin_s: Annotated[float, Field(ge=0)]


class ViewIndex(ViewMetadata):
    frames: Annotated[int, Field(ge=2)]


class RallyIndex(RallyMetadata):
    views: Annotated[list[ViewIndex], Field(min_length=1)]

    @model_validator(mode="after")
    def unique_views(self) -> Self:
        ids = [v.view_id for v in self.views]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate view identity")
        return self


class Counts(StrictModel):
    rallies: NonnegativeInt
    views: NonnegativeInt
    frames: NonnegativeInt


class Shard(Artifact):
    row_group_rallies: PositiveInt
    records: Annotated[list[RallyIndex], Field(min_length=1)]


class Manifest(StrictModel):
    schema_version: Literal["2.0"] = "2.0"
    state: Literal["incomplete", "complete"] = "incomplete"
    provenance: Provenance
    shards: list[Shard]
    expected: dict[Split, Counts]

    @model_validator(mode="after")
    def membership(self) -> Self:
        if set(self.expected) != {"train", "val", "test"}:
            raise ValueError("expected counts must include every split")
        paths = [s.path for s in self.shards]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate shard path")
        identities: dict[str, dict[str, str]] = {
            key: {} for key in ("rally_id", "family_id", "pool_sha256")
        }
        pools = {p.family_id: (p.split, p.sha256) for p in self.provenance.pools}
        for shard in self.shards:
            for row in shard.records:
                for key, seen in identities.items():
                    value = getattr(row, key)
                    if value in seen and (
                        key == "rally_id" or seen[value] != row.split
                    ):
                        raise ValueError(f"duplicate or cross-split {key}")
                    seen[value] = row.split
                if pools.get(row.family_id) != (row.split, row.pool_sha256):
                    raise ValueError("rally family/pool provenance mismatch")
        counts = count_records(self.shards)
        for split, expected in self.expected.items():
            actual = counts[split]
            for key in ("rallies", "views", "frames"):
                if getattr(actual, key) > getattr(expected, key):
                    raise ValueError("dataset exceeds expected membership counts")
            if self.state == "complete" and actual != expected:
                raise ValueError("complete dataset does not match expected counts")
        return self


def count_records(shards: list[Shard]) -> dict[Split, Counts]:
    result: dict[Split, dict[str, int]] = {
        s: {"rallies": 0, "views": 0, "frames": 0} for s in ("train", "val", "test")
    }
    for shard in shards:
        for row in shard.records:
            result[row.split]["rallies"] += 1
            result[row.split]["views"] += len(row.views)
            result[row.split]["frames"] += sum(v.frames for v in row.views)
    return {s: Counts(**v) for s, v in result.items()}


def local_path(root: Path, relative: str) -> Path:
    Artifact.relative_path(relative)
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact symlink escapes dataset root")
    return path
