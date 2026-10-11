"""Execution limits are separate from the frozen numeric generation recipe."""

from typing import Literal

from pydantic import Field, model_validator

from tennis3d.schemas import StrictModel


class ProducerConfig(StrictModel):
    workers: int = Field(default=2, ge=1, le=4)
    inflight: int = Field(default=4, ge=1, le=64)
    worker_cache_families: int = Field(default=8, ge=1, le=64)
    family_cohort: int = Field(default=16, ge=1)
    shard_rallies: int = Field(default=32, ge=1)
    row_group_rallies: int = Field(default=1, ge=1)
    max_proposals: int = Field(default=8, ge=1, le=8)
    max_record_bytes: int = Field(default=16 << 20, ge=1024)
    memory_bytes: int = Field(default=6 << 30, ge=1 << 28)
    controller_reserve_bytes: int = Field(default=1 << 30, ge=1 << 20)
    worker_reserve_bytes: int = Field(default=256 << 20, ge=1 << 20)
    finalization_reserve_bytes: int = Field(default=5 << 30, ge=1 << 20)
    staging_bytes: int = Field(default=4 << 30, ge=1 << 20)
    metadata_reserve_bytes: int = Field(default=1 << 30, ge=1 << 20)
    sdk_reserve_bytes: int = Field(default=256 << 20, ge=1 << 20)
    max_failures: int = Field(default=3, ge=1, le=10)
    drain_seconds: float = Field(default=30, gt=0, le=300)
    recovery_bytes: int = Field(default=4 << 30, ge=1 << 20)
    repo_id: str | None = None
    revision: str | None = None
    prefix: str = "dataset"
    evict: bool = False
    subset_policy: str | None = None
    purpose: Literal["main-candidate", "qualification"] = "main-candidate"

    @model_validator(mode="after")
    def limits(self):
        from tennis3d.datasets.schema import Artifact

        Artifact.relative_path(self.prefix)
        if self.row_group_rallies > self.shard_rallies:
            raise ValueError("row groups cannot exceed shard membership")
        if self.inflight < self.workers:
            raise ValueError("inflight must include every worker")
        # IPC transports can retain both pickled and decoded results. Arrow's
        # nested Python representation costs much more than the numeric buffers.
        required = (
            self.controller_reserve_bytes
            + self.workers * self.worker_reserve_bytes
            + self.max_record_bytes * (4 * self.inflight + 16 * self.row_group_rallies)
        )
        if (
            required > self.memory_bytes
            or self.controller_reserve_bytes + self.finalization_reserve_bytes
            > self.memory_bytes
        ):
            raise ValueError("execution reservations exceed the memory budget")
        if bool(self.repo_id) != bool(self.revision) or (
            self.evict and not self.repo_id
        ):
            raise ValueError(
                "publication requires repo_id and revision; eviction requires publication"
            )
        if self.shard_reservation() >= self.staging_bytes:
            raise ValueError("staging cannot reserve one complete shard transaction")
        return self

    def recipe_reservation(self, cfg) -> None:
        from tennis3d.synthetic import KINDS
        from tennis3d.synthetic.producer_records import recipe_frame_bytes

        if recipe_frame_bytes(cfg) > self.max_record_bytes:
            raise ValueError("recipe frame reservation exceeds max_record_bytes")
        pool_bytes = cfg.pool_size * len(KINDS) * 9 * 8
        cache_bytes = self.worker_cache_families * (8 * pool_bytes + (4 << 20))
        if cache_bytes + (128 << 20) > self.worker_reserve_bytes:
            raise ValueError(
                "worker pool/cache reservation exceeds worker_reserve_bytes"
            )
        families = sum(cfg.family_count(s) for s in ("train", "val", "test"))
        if 2 * pool_bytes * families + self.shard_reservation() > self.staging_bytes:
            raise ValueError(
                "all planned pools plus a shard exceed staging reservation"
            )

    def shard_reservation(self) -> int:
        # Payload plus writer compression/metadata, resolver copy, SDK and the
        # old/new durable metadata generations. Pool cache is charged separately.
        return (
            2 * self.shard_rallies * self.max_record_bytes
            + self.metadata_reserve_bytes
            + self.sdk_reserve_bytes
        )

    def layout(self) -> dict:
        return self.model_dump(
            include={
                "family_cohort",
                "shard_rallies",
                "row_group_rallies",
                "max_proposals",
                "prefix",
                "purpose",
            }
        )
