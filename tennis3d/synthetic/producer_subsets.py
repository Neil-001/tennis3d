"""Catalog joins and owned compact extraction in the producer's one audit."""

import shutil
from pathlib import Path

from tennis3d.artifacts import json_sha256
from tennis3d.datasets.io import read_manifest, write_manifest
from tennis3d.datasets.schema import Artifact, Manifest
from tennis3d.datasets.subset import (
    CatalogRally,
    CoverageCatalog,
    ExtractedSubset,
    SelectionPolicy,
    SubsetExtractor,
    ViewCounts,
    select_subset,
)
from tennis3d.synthetic import KINDS
from tennis3d.synthetic.producer_catalog import (
    DISTANCES,
    GAPS,
    STRATA_DEFINITIONS,
    definition_hash,
    stratum_labels,
)
from tennis3d.synthetic.producer_state import (
    artifact,
    fsync_dir,
    load,
    save,
    usage,
    verify,
)


def build_catalog(files, manifest, commits) -> CoverageCatalog:
    """Join verified ordered generation parts to every original manifest row.

    No numeric shard is opened. The complete manifest is prospective until the
    parent's final audit succeeds. Commits may omit their already-parsed shard.
    This internal join accepts the coordinator's validated typed ledger metadata.
    Fresh outer validation checks membership while sharing its frozen indexes;
    The subset tooling independently copies and validates the catalog before selection.
    """
    manifest = Manifest(
        state=manifest.state,
        provenance=manifest.provenance,
        shards=manifest.shards,
        expected=manifest.expected,
    )
    if manifest.state != "complete":
        raise ValueError("catalog requires prospective complete manifest")
    if len(commits) != len(manifest.shards):
        raise ValueError("catalog ledger differs from manifest shards")
    rows = []
    for shard, commit in zip(manifest.shards, commits):
        if "shard" in commit and commit["shard"] != shard.model_dump():
            raise ValueError("catalog ledger shard order differs from manifest")
        spec = Artifact.model_validate(commit["catalog"])
        part = load(verify(Path(files), spec))
        if set(part) != {"records"} or len(part["records"]) != len(shard.records):
            raise ValueError("catalog part must cover its entire shard")
        for row_index, (raw, original) in enumerate(
            zip(part["records"], shard.records)
        ):
            if raw["index"] != original.model_dump():
                raise ValueError("catalog source row differs from manifest")
            index = original
            presence = raw["shot_kind_bits"]
            if len(presence) != len(KINDS) or any(
                type(v) is not bool for v in presence
            ):
                raise ValueError("catalog shot presence differs from KINDS")
            bits = sum(1 << i for i, present in enumerate(presence) if present)
            counts = []
            for view in raw["views"]:
                if len(view["gap_counts"]) != len(GAPS) or len(
                    view["distance_counts"]
                ) != len(DISTANCES):
                    raise ValueError("catalog coverage bins differ from evaluator")
                counts.append(
                    ViewCounts(
                        view_id=view["view_id"],
                        supported=view["supported"],
                        supported_occluded=view["occluded"],
                        gap_bins=dict(zip(GAPS, view["gap_counts"])),
                        distance_bins=dict(zip(DISTANCES, view["distance_counts"])),
                    )
                )
            duration = raw["duration_s"]
            rows.append(
                CatalogRally(
                    shard_path=shard.path,
                    shard_sha256=shard.sha256,
                    row_index=row_index,
                    record=index,
                    counts=counts,
                    physical_duration_s=duration,
                    shot_kind_bits=bits,
                    strata=stratum_labels(index, duration, bits, counts),
                )
            )
    return CoverageCatalog(
        source_manifest_sha256=json_sha256(manifest.model_dump()),
        definitions=STRATA_DEFINITIONS,
        definitions_source_sha256=definition_hash(),
        records=rows,
    )


def rebase_extracted(output, extracted, prefix) -> ExtractedSubset:
    """Qualify compact metadata paths while keeping files under that repo prefix.

    The subset tooling writes files inside output/prefix with basename paths.
    Rewrite its small manifest and byte identity so all returned artifacts resolve
    from output. Call exactly once on an unqualified, newly finished extraction.
    """
    Artifact.relative_path(prefix)
    output = Path(output)
    compact = output / prefix
    relative_paths = [extracted.manifest.path, *[s.path for s in extracted.shards]]
    if any(Path(path).name != path for path in relative_paths):
        raise ValueError("extraction paths must be unqualified basenames")
    manifest_path = compact / extracted.manifest.path
    manifest = read_manifest(manifest_path)
    if manifest.shards != extracted.shards:
        raise ValueError("compact manifest differs from extracted shards")
    shards = [
        shard.model_copy(update={"path": f"{prefix}/{shard.path}"})
        for shard in extracted.shards
    ]
    manifest = Manifest.model_validate(
        {**manifest.model_dump(), "shards": [s.model_dump() for s in shards]}
    )
    write_manifest(manifest_path, manifest, overwrite=True)
    return ExtractedSubset(
        source_provenance_sha256=extracted.source_provenance_sha256,
        selection=extracted.selection,
        manifest=artifact(output, f"{prefix}/{extracted.manifest.path}"),
        shards=shards,
    )


class ExtractionTransaction:
    """Audit visitor owning a NEW repository root, with no implicit quotas.

    The parent journals and later moves that root.
    """

    def __init__(self, output, manifest, catalog, policy, execution):
        policy = SelectionPolicy.model_validate(policy.model_dump())
        selection = select_subset(manifest, catalog, policy)
        self.root = Path(output)
        self.execution = execution
        self.selected_ids = frozenset(row.record.rally_id for row in selection.records)
        self.prefix = f"{execution.prefix}/search"
        Artifact.relative_path(self.prefix)
        self.root.mkdir(parents=True, exist_ok=False)
        self._finished = False
        self._aborted = False
        self._extractor = None
        try:
            self._extractor = SubsetExtractor(
                self.root / self.prefix,
                manifest,
                catalog,
                selection,
                row_group_rallies=execution.row_group_rallies,
                target_bytes=execution.shard_rallies * execution.max_record_bytes,
            )
        except BaseException:
            self.abort()
            raise

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if kind is not None or not self._finished:
            self.abort()

    def abort(self):
        if self._aborted:
            return
        if self._extractor is not None:
            self._extractor.abort()
        shutil.rmtree(self.root)
        self._aborted = True
        self._finished = False

    def __call__(self, record, source):
        try:
            if self._finished or self._aborted:
                raise ValueError("subset extraction transaction is closed")
            if (
                source.rally_id in self.selected_ids
                and usage(self.root.parent) + self.execution.shard_reservation()
                > self.execution.staging_bytes
            ):
                raise ValueError(
                    "staging cannot reserve selected subset copy transaction"
                )
            assert self._extractor is not None
            self._extractor(record, source)
        except BaseException:
            self.abort()
            raise

    def finish(self, report) -> ExtractedSubset:
        """Accept only the trusted return report of the complete parent audit."""
        try:
            if self._finished or self._aborted:
                raise ValueError("subset extraction transaction is closed")
            assert self._extractor is not None
            extracted = self._extractor.finish(source_audit_report=report)
            extracted = rebase_extracted(self.root, extracted, self.prefix)
            save(self.root / "selection.json", extracted.selection.model_dump())
            save(self.root / "extracted.json", extracted.model_dump())
            fsync_dir(self.root / self.prefix)
            fsync_dir(self.root)
            self._finished = True
            return extracted
        except BaseException:
            self.abort()
            raise
