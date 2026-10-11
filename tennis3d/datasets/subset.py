"""Declared subset selection, audit-pass extraction, and offline verification.

The producer supplies catalog strata from frozen definitions. This module has no
production quota defaults and never uses model scores to choose records.
"""

import shutil
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Annotated, Literal, Protocol, Self

import numpy as np
from pydantic import Field, model_validator

from tennis3d.artifacts import (
    atomic_file,
    file_sha256,
    json_sha256,
    read_json,
    write_json,
)
from tennis3d.datasets.arrays import RallyRecord
from tennis3d.datasets.io import (
    DEFAULT_MAX_GROUP_BYTES,
    ShardReader,
    ShardWriter,
    read_manifest,
    write_manifest,
)
from tennis3d.datasets.schema import (
    Artifact,
    Manifest,
    RallyIndex,
    Seed,
    Shard,
    count_records,
    local_path,
)
from tennis3d.evaluation.metrics import strata as evaluation_strata
from tennis3d.schemas import SHA256, NonnegativeInt, StrictModel

Commit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Name = Annotated[str, Field(min_length=1)]
GAP_BINS = ("gap_up_to_0.15s", "gap_0.15_to_0.5s", "gap_over_0.5s")
DISTANCE_BINS = ("distance_under_20m", "distance_20_to_40m", "distance_40m_plus")


class ViewCounts(StrictModel):
    view_id: NonnegativeInt
    supported: NonnegativeInt
    supported_occluded: NonnegativeInt
    gap_bins: dict[str, NonnegativeInt]
    distance_bins: dict[str, NonnegativeInt]

    @model_validator(mode="after")
    def valid_counts(self) -> Self:
        if self.supported_occluded > self.supported:
            raise ValueError("occluded count exceeds supported count")
        if set(self.gap_bins) != set(GAP_BINS) or set(self.distance_bins) != set(
            DISTANCE_BINS
        ):
            raise ValueError("catalog coverage bins differ from evaluator definitions")
        if (
            sum(self.gap_bins.values()) != self.supported_occluded
            or sum(self.distance_bins.values()) != self.supported
        ):
            raise ValueError("catalog coverage bins do not partition supported counts")
        return self


def supported_counts(record: RallyRecord) -> list[ViewCounts]:
    result = []
    for view in record.views:
        supported = view.arrays["label_valid"] & view.arrays["camera_valid"]
        masks = evaluation_strata(view.arrays, view.arrays["ball_observed"])
        result.append(
            ViewCounts(
                view_id=view.metadata.view_id,
                supported=int(np.count_nonzero(supported)),
                supported_occluded=int(
                    np.count_nonzero(supported & ~view.arrays["ball_observed"])
                ),
                gap_bins={
                    k: int(np.count_nonzero(supported & masks[k])) for k in GAP_BINS
                },
                distance_bins={
                    k: int(np.count_nonzero(supported & masks[k]))
                    for k in DISTANCE_BINS
                },
            )
        )
    return result


class CatalogRally(StrictModel):
    shard_path: str
    shard_sha256: SHA256
    row_index: NonnegativeInt
    record: RallyIndex
    counts: list[ViewCounts]
    physical_duration_s: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    shot_kind_bits: Annotated[int, Field(gt=0, le=127)]
    recipe_category: Name | None = None
    # Labels are supplied by the producer using catalog.definitions.
    strata: dict[str, str]

    @model_validator(mode="after")
    def valid_row(self) -> Self:
        Artifact.relative_path(self.shard_path)
        if [v.view_id for v in self.counts] != [v.view_id for v in self.record.views]:
            raise ValueError("catalog counts must include every view in source order")
        if any(c.supported > v.frames for c, v in zip(self.counts, self.record.views)):
            raise ValueError("supported count exceeds view frames")
        return self


def _check_coverage(record: RallyRecord, row: CatalogRally) -> None:
    if supported_counts(record) != row.counts:
        raise ValueError("numeric supported counts differ from catalog")
    duration = float(record.physical["t_s"][-1] - record.physical["t_s"][0])
    bits = sum(1 << int(kind) for kind in np.unique(record.physical["segment_kind"]))
    if duration != row.physical_duration_s or bits != row.shot_kind_bits:
        raise ValueError("numeric physical coverage differs from catalog")


class CoverageCatalog(StrictModel):
    schema_version: Literal["subset-catalog-1"] = "subset-catalog-1"
    source_manifest_sha256: SHA256
    definitions: dict[str, str]
    # Frozen producer recipe/source identity for the catalog definitions.
    definitions_source_sha256: SHA256
    records: list[CatalogRally]

    @model_validator(mode="after")
    def valid_catalog(self) -> Self:
        ids = [row.record.rally_id for row in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate catalog rally")
        if any(set(row.strata) != set(self.definitions) for row in self.records):
            raise ValueError("catalog strata differ from declared definitions")
        if any(
            not name or not definition for name, definition in self.definitions.items()
        ):
            raise ValueError("catalog definitions must be nonempty")
        return self


class StratumQuota(StrictModel):
    name: Name
    split: Literal["train", "val"]
    equals: dict[str, str]
    rallies: Annotated[int, Field(gt=0)]


class SelectionPolicy(StrictModel):
    algorithm: Literal["sha256-rally-rank-v1"] = "sha256-rally-rank-v1"
    seed: Seed
    strata: Annotated[list[StratumQuota], Field(min_length=1)]
    reserved_rally_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_names(self) -> Self:
        keys = [(s.split, s.name) for s in self.strata]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate stratum name within split")
        if len(self.reserved_rally_ids) != len(set(self.reserved_rally_ids)):
            raise ValueError("duplicate reserved rally")
        return self


class Selection(StrictModel):
    source_manifest_sha256: SHA256
    catalog_sha256: SHA256
    catalog_schema: Literal["subset-catalog-1"] = "subset-catalog-1"
    catalog_definitions: dict[str, str]
    catalog_definitions_source_sha256: SHA256
    policy: SelectionPolicy
    records: list[CatalogRally]


def _validate_catalog(manifest: Manifest, catalog: CoverageCatalog) -> None:
    if json_sha256(manifest.model_dump()) != catalog.source_manifest_sha256:
        raise ValueError("catalog source manifest mismatch")
    original = {
        (shard.path, i): (shard.sha256, row)
        for shard in manifest.shards
        for i, row in enumerate(shard.records)
    }
    rows = {(row.shard_path, row.row_index): row for row in catalog.records}
    if set(rows) != set(original) or len(rows) != len(catalog.records):
        raise ValueError("catalog must cover the entire source manifest")
    for key, row in rows.items():
        if original[key] != (row.shard_sha256, row.record):
            raise ValueError("catalog source row mismatch")


def _stratum_lookup(policy: SelectionPolicy):
    # Build per validation, since callers can mutate policy models and dictionaries.
    lookup = {}
    for quota in policy.strata:
        keys = tuple(sorted(quota.equals))
        values = tuple(quota.equals[k] for k in keys)
        lookup.setdefault(quota.split, {}).setdefault(keys, {}).setdefault(
            values, []
        ).append(quota)
    return lookup


def _matching_strata(row: CatalogRally, lookup):
    return [
        quota
        for keys, values in lookup.get(row.record.split, {}).items()
        for quota in values.get(tuple(row.strata[k] for k in keys), [])
    ]


def select_subset(
    manifest: Manifest, catalog: CoverageCatalog, policy: SelectionPolicy
) -> Selection:
    """Rank all eligible IDs per disjoint declared stratum; reject sparse quotas."""
    manifest = Manifest.model_validate(manifest.model_dump())
    catalog = CoverageCatalog.model_validate(catalog.model_dump())
    policy = SelectionPolicy.model_validate(policy.model_dump())
    if manifest.state != "complete":
        raise ValueError("selection requires the prospective complete source manifest")
    _validate_catalog(manifest, catalog)
    reserved = set(policy.reserved_rally_ids)
    if not reserved <= {row.record.rally_id for row in catalog.records}:
        raise ValueError("reserved rally is absent from catalog")
    buckets: dict[tuple[str, str], list[CatalogRally]] = {
        (s.split, s.name): [] for s in policy.strata
    }
    for quota in policy.strata:
        if not set(quota.equals) <= set(catalog.definitions):
            raise ValueError("stratum uses an undeclared catalog definition")
    lookup = _stratum_lookup(policy)
    for row in catalog.records:
        if row.record.split == "test" or row.record.rally_id in reserved:
            continue
        matching = _matching_strata(row, lookup)
        if len(matching) > 1:
            raise ValueError("selection strata overlap")
        if matching:
            quota = matching[0]
            buckets[quota.split, quota.name].append(row)
    selected = []
    for quota in sorted(policy.strata, key=lambda s: (s.split, s.name)):
        candidates = buckets[quota.split, quota.name]
        if len(candidates) < quota.rallies:
            raise ValueError(f"infeasible quota for {quota.split}/{quota.name}")
        ranked = sorted(
            candidates,
            key=lambda row: (
                json_sha256(
                    [
                        policy.algorithm,
                        policy.seed,
                        quota.split,
                        quota.name,
                        row.record.rally_id,
                    ]
                ),
                row.record.rally_id,
            ),
        )
        selected.extend(ranked[: quota.rallies])
    return Selection(
        source_manifest_sha256=catalog.source_manifest_sha256,
        catalog_sha256=json_sha256(catalog.model_dump()),
        catalog_definitions=catalog.definitions,
        catalog_definitions_source_sha256=catalog.definitions_source_sha256,
        policy=policy,
        records=selected,
    )


class AuditedRecordSource(Protocol):
    """Structural producer callback contract; temporary resolver paths are excluded."""

    @property
    def shard_path(self) -> str: ...
    @property
    def shard_sha256(self) -> str: ...
    @property
    def row_index(self) -> int: ...
    @property
    def split(self) -> str: ...
    @property
    def rally_id(self) -> str: ...
    @property
    def view_ids(self) -> tuple[int, ...]: ...


class ExtractedSubset(StrictModel):
    source_provenance_sha256: SHA256
    selection: Selection
    manifest: Artifact
    shards: list[Shard]


class SubsetExtractor:
    """Callable visitor for the producer's existing global numeric audit.

    Use as a context manager. Call finish only after that audit returns normally,
    including all global/catalog checks. An exception or absent finish aborts the
    owned directory. Never invoke a second source scan to feed this visitor.
    """

    def __init__(
        self,
        output_dir: str | Path,
        manifest: Manifest,
        catalog: CoverageCatalog,
        selection: Selection,
        *,
        row_group_rallies: int = 16,
        target_bytes: int = 32 * 1024**2,
    ):
        if row_group_rallies <= 0 or target_bytes <= 0:
            raise ValueError("writer bounds must be positive")
        checked = select_subset(manifest, catalog, selection.policy)
        if checked != selection:
            raise ValueError("selection differs from declared deterministic policy")
        self.selection = checked
        self.source = Manifest.model_validate(manifest.model_dump())
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=False)
        self.row_group_rallies, self.target_bytes = row_group_rallies, target_bytes
        self._selected = {r.record.rally_id: r for r in checked.records}
        self._seen: set[str] = set()
        self._audit_shard = 0
        self._audit_row = 0
        self._writer: ShardWriter | None = None
        self._shards: list[Shard] = []
        self._finished = False
        self._aborted = False

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if kind is not None or not self._finished:
            self.abort()

    def abort(self) -> None:
        if self._writer is not None:
            self._writer.abort()
            self._writer = None
        if not self._aborted:
            shutil.rmtree(self.root)
        self._aborted = True
        self._finished = False

    def __call__(self, record: RallyRecord, source: AuditedRecordSource) -> None:
        try:
            self._visit(record, source)
        except BaseException:
            self.abort()
            raise

    def _visit(self, record: RallyRecord, source: AuditedRecordSource) -> None:
        if self._finished or self._aborted:
            raise ValueError("subset extraction is closed")
        if self._audit_shard >= len(self.source.shards):
            raise ValueError("duplicate or excess source audit callback")
        shard = self.source.shards[self._audit_shard]
        original = shard.records[self._audit_row]
        if (
            source.shard_path,
            source.shard_sha256,
            source.row_index,
            source.split,
            source.rally_id,
            tuple(source.view_ids),
        ) != (
            shard.path,
            shard.sha256,
            self._audit_row,
            original.split,
            original.rally_id,
            tuple(v.view_id for v in original.views),
        ) or record.index() != original:
            raise ValueError(
                "source audit callbacks differ from original shard/row order"
            )
        self._audit_row += 1
        if self._audit_row == len(shard.records):
            self._audit_row = 0
            self._audit_shard += 1
        # The catalog covers each unique original row once, so a selected rally
        # matched at this audit position is its selected source row.
        selected = self._selected.get(source.rally_id)
        if selected is None:
            return
        _check_coverage(record, selected)
        if self._writer is None:
            name = f"part-{len(self._shards):05d}.parquet"
            self._writer = ShardWriter(
                self.root / name,
                self.source.provenance,
                row_group_rallies=self.row_group_rallies,
                target_bytes=self.target_bytes,
            )
        self._writer.append(record)
        self._seen.add(source.rally_id)
        if self._writer.target_reached:
            self._shards.append(self._writer.close())
            self._writer = None

    def finish(self, *, source_audit_report: dict) -> ExtractedSubset:
        """Commit only after the caller's full source audit succeeds.

        The report is the trusted audit return value, not an untrusted receipt.
        The source manifest must already describe its full expected membership.
        """
        try:
            return self._finish(source_audit_report)
        except BaseException:
            self.abort()
            raise

    def _finish(self, source_audit_report: dict) -> ExtractedSubset:
        if self._finished or self._aborted:
            raise ValueError("subset extraction is closed")
        Manifest.model_validate({**self.source.model_dump(), "state": "complete"})
        expected = {
            "schema_version": "2.0",
            "shards": len(self.source.shards),
            "counts": {
                s: c.model_dump() for s, c in count_records(self.source.shards).items()
            },
        }
        if source_audit_report != expected:
            raise ValueError("successful complete source audit report required")
        if self._audit_shard != len(self.source.shards):
            raise ValueError("source audit did not visit every original row")
        if self._seen != set(self._selected):
            raise ValueError("source audit did not supply every selected rally")
        if self._writer is not None:
            self._shards.append(self._writer.close())
            self._writer = None
        manifest = Manifest(
            state="complete",
            provenance=self.source.provenance,
            shards=self._shards,
            expected=count_records(self._shards),
        )
        path = self.root / "manifest.json"
        write_manifest(path, manifest)
        result = ExtractedSubset(
            source_provenance_sha256=json_sha256(self.source.provenance.model_dump()),
            selection=self.selection,
            manifest=_artifact(path, "manifest.json"),
            shards=self._shards,
        )
        self._finished = True
        return result


def _artifact(path: Path, relative: str) -> Artifact:
    return Artifact(
        path=relative, sha256=file_sha256(path), size_bytes=path.stat().st_size
    )


class SubsetDescriptor(StrictModel):
    schema_version: Literal["compact-subset-1"] = "compact-subset-1"
    main_repo_id: Name
    main_repo_type: Literal["dataset"] = "dataset"
    main_commit: Commit
    main_manifest: Artifact
    main_catalog: Artifact
    # These original source artifacts and identities are independent of storage
    # paths/commit used to publish this descriptor and its compact shards.
    subset: ExtractedSubset

    @model_validator(mode="after")
    def membership(self) -> Self:
        rows = self.subset.selection.records
        original = {r.record.rally_id: r.record for r in rows}
        copied = {r.rally_id: r for s in self.subset.shards for r in s.records}
        if (
            len(original) != len(rows)
            or original != copied
            or len(copied) != sum(len(s.records) for s in self.subset.shards)
        ):
            raise ValueError("subset artifact membership differs from selection")
        if any(r.split == "test" for r in original.values()):
            raise ValueError("test rallies cannot enter search subset")
        if set(original) & set(self.subset.selection.policy.reserved_rally_ids):
            raise ValueError("reserved rally in subset")
        selection = self.subset.selection
        assigned = {(s.split, s.name): 0 for s in selection.policy.strata}
        for quota in selection.policy.strata:
            if not set(quota.equals) <= set(selection.catalog_definitions):
                raise ValueError("selection uses undefined catalog strata")
        lookup = _stratum_lookup(selection.policy)
        for row in rows:
            if set(row.strata) != set(selection.catalog_definitions):
                raise ValueError("selected catalog definitions differ")
            matches = _matching_strata(row, lookup)
            if len(matches) != 1:
                raise ValueError(
                    "selected rally must match exactly one declared stratum"
                )
            assigned[matches[0].split, matches[0].name] += 1
        if any(assigned[s.split, s.name] != s.rallies for s in selection.policy.strata):
            raise ValueError("selected membership differs from declared quotas")
        for key in ("family_id", "pool_sha256"):
            seen = {}
            for row in original.values():
                identity = getattr(row, key)
                if identity in seen and seen[identity] != row.split:
                    raise ValueError("cross-split selected family/pool")
                seen[identity] = row.split
        paths = [self.subset.manifest.path, *[s.path for s in self.subset.shards]]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate compact artifact path")
        return self


def bind_release(
    subset: ExtractedSubset,
    *,
    main_repo_id: str,
    main_commit: str,
    main_manifest: Artifact,
    main_catalog: Artifact,
    main_manifest_path: str | Path,
    main_catalog_path: str | Path,
) -> SubsetDescriptor:
    """Bind verified source metadata to the publisher's immutable main commit.

    Local files must match both published byte identities and the original
    canonical selection inputs. This reads metadata only, never main shards.
    The publisher remains responsible for authenticating the remote commit.
    """
    subset = ExtractedSubset.model_validate(subset.model_dump())
    manifest_path, catalog_path = Path(main_manifest_path), Path(main_catalog_path)
    _verify_artifact(manifest_path, main_manifest)
    _verify_artifact(catalog_path, main_catalog)
    manifest = read_manifest(manifest_path)
    catalog = CoverageCatalog.model_validate(read_json(catalog_path))
    if (
        json_sha256(manifest.model_dump()) != subset.selection.source_manifest_sha256
        or json_sha256(catalog.model_dump()) != subset.selection.catalog_sha256
    ):
        raise ValueError("published source metadata differs from original selection")
    if select_subset(manifest, catalog, subset.selection.policy) != subset.selection:
        raise ValueError("published source selection differs from extracted membership")
    if json_sha256(manifest.provenance.model_dump()) != subset.source_provenance_sha256:
        raise ValueError("subset provenance differs from original main manifest")
    # Refuse metadata changed while it was parsed and checked.
    _verify_artifact(manifest_path, main_manifest)
    _verify_artifact(catalog_path, main_catalog)
    return SubsetDescriptor(
        main_repo_id=main_repo_id,
        main_commit=main_commit,
        main_manifest=main_manifest,
        main_catalog=main_catalog,
        subset=subset,
    )


def write_descriptor(path: str | Path, descriptor: SubsetDescriptor) -> str:
    checked = SubsetDescriptor.model_validate(descriptor.model_dump())
    payload = checked.model_dump()
    digest = json_sha256(payload)
    write_json(path, {"descriptor": payload, "sha256": digest})
    return digest


def read_descriptor(path: str | Path, *, expected_sha256: str) -> SubsetDescriptor:
    raw = read_json(path)
    if (
        set(raw) != {"descriptor", "sha256"}
        or raw["sha256"] != expected_sha256
        or json_sha256(raw["descriptor"]) != expected_sha256
    ):
        raise ValueError("subset descriptor checksum mismatch")
    return SubsetDescriptor.model_validate(raw["descriptor"])


def validate_source_subset(
    root: str | Path,
    expected_subset_sha256: str,
    manifest_artifact: Artifact,
    repo_id: str | None,
    revision: str | None,
) -> SubsetDescriptor:
    """Check frozen source metadata at open without another numeric scan."""
    root = Path(root)
    descriptor = read_descriptor(
        root / "subset.json", expected_sha256=expected_subset_sha256
    )
    if descriptor.subset.manifest != manifest_artifact:
        raise ValueError("source compact manifest differs from subset descriptor")
    if (repo_id, revision) != (descriptor.main_repo_id, descriptor.main_commit):
        raise ValueError("source main publication differs from subset descriptor")
    return descriptor


def _verify_artifact(path: Path, artifact: Artifact) -> None:
    if (
        path.stat().st_size != artifact.size_bytes
        or file_sha256(path) != artifact.sha256
    ):
        raise ValueError("compact artifact size/checksum mismatch")


def verify_materialized(
    root: str | Path,
    *,
    expected_descriptor_sha256: str,
    max_group_bytes: int = DEFAULT_MAX_GROUP_BYTES,
) -> SubsetDescriptor:
    """Verify local numeric membership/counts. No network or fallback exists."""
    if max_group_bytes <= 0:
        raise ValueError("decoded group byte bound must be positive")
    root = Path(root)
    descriptor = read_descriptor(
        root / "subset.json", expected_sha256=expected_descriptor_sha256
    )
    compact = descriptor.subset
    path = local_path(root, compact.manifest.path)
    _verify_artifact(path, compact.manifest)
    manifest = read_manifest(path)
    if (
        json_sha256(manifest.provenance.model_dump())
        != compact.source_provenance_sha256
    ):
        raise ValueError("compact manifest provenance differs from original source")
    if manifest.shards != compact.shards:
        raise ValueError("compact manifest shards differ from descriptor")
    selected = {row.record.rally_id: row for row in compact.selection.records}
    for shard in manifest.shards:
        with ShardReader(
            local_path(root, shard.path),
            shard,
            provenance=manifest.provenance,
            max_group_bytes=max_group_bytes,
        ) as reader:
            for i in range(len(shard.records)):
                record = reader.read(i)
                _check_coverage(record, selected[record.metadata.rally_id])
    return descriptor


def materialize_subset(
    descriptor: SubsetDescriptor,
    output_dir: str | Path,
    *,
    resolver: Callable[[Artifact], AbstractContextManager[Path]],
    expected_descriptor_sha256: str,
    max_group_bytes: int = DEFAULT_MAX_GROUP_BYTES,
) -> Path:
    """Copy compact artifacts before timed trials, then verify entirely locally.

    Resolver scope belongs to the separately pinned compact publication. It is
    called only for compact files, never original main shards or the main catalog.
    The output directory must be new; failures remove only this transaction.
    """
    if max_group_bytes <= 0:
        raise ValueError("decoded group byte bound must be positive")
    descriptor = SubsetDescriptor.model_validate(descriptor.model_dump())
    if json_sha256(descriptor.model_dump()) != expected_descriptor_sha256:
        raise ValueError("subset descriptor checksum mismatch")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    try:
        for artifact in [descriptor.subset.manifest, *descriptor.subset.shards]:
            with resolver(artifact) as source:
                destination = local_path(root, artifact.path)
                with (
                    atomic_file(destination) as stream,
                    Path(source).open("rb") as incoming,
                ):
                    remaining = artifact.size_bytes
                    while remaining:
                        chunk = incoming.read(min(remaining, 1024 * 1024))
                        if not chunk:
                            raise ValueError(
                                "compact artifact is shorter than declared size"
                            )
                        stream.write(chunk)
                        remaining -= len(chunk)
                    if incoming.read(1):
                        raise ValueError("compact artifact exceeds declared size")
                _verify_artifact(destination, artifact)
        write_descriptor(root / "subset.json", descriptor)
        # SourceSpec defaults control runtime/cache settings. These descriptors
        # bind local storage to the original immutable publication provenance.
        for split in ("train", "val"):
            if any(
                r.record.split == split for r in descriptor.subset.selection.records
            ):
                write_json(
                    root / f"{split}.source.json",
                    {
                        "source_version": 1,
                        "manifest": descriptor.subset.manifest.model_dump(),
                        "split": split,
                        "root": str(root.resolve()),
                        "repo_id": descriptor.main_repo_id,
                        "revision": descriptor.main_commit,
                        "subset_sha256": expected_descriptor_sha256,
                        "max_group_bytes": max_group_bytes,
                    },
                )
        verify_materialized(
            root,
            expected_descriptor_sha256=expected_descriptor_sha256,
            max_group_bytes=max_group_bytes,
        )
    except BaseException:
        shutil.rmtree(root)
        raise
    return root / descriptor.subset.manifest.path
