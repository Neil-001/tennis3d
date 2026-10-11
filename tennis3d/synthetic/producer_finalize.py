"""One durable numeric audit, including optional compact extraction."""

import gc
import shutil
import time
from contextlib import nullcontext

from tennis3d.artifacts import json_sha256, read_json, write_json
from tennis3d.datasets.audit import finalize_manifest
from tennis3d.datasets.io import read_manifest
from tennis3d.datasets.schema import Artifact, Manifest
from tennis3d.synthetic.producer_catalog import summarize
from tennis3d.synthetic.producer_parallel import Stopped
from tennis3d.synthetic.producer_state import (
    artifact,
    fsync_dir,
    load,
    save,
    usage,
    verify,
)
from tennis3d.synthetic.producer_subsets import ExtractionTransaction, build_catalog
from tennis3d.synthetic.producer_summary import NumericSummary


def recover_extraction(output, control):
    """Only a journal authorizes removal of interrupted owned compact files."""
    journal = control / "subset-active.json"
    if not journal.exists():
        return
    if load(journal) != {"roots": ["subset-transaction", "subsets"]}:
        raise ValueError("invalid extraction ownership journal")
    if not (control / "finalized.json").exists():
        for name in ("subset-transaction", "subsets"):
            root = output / name
            if root.is_symlink():
                raise ValueError("interrupted extraction root is a symlink")
            if root.exists():
                shutil.rmtree(root)
        fsync_dir(output)
    journal.unlink()
    fsync_dir(control)


def finalization(
    output,
    files,
    control,
    manifest,
    commits,
    plan,
    execution,
    publication,
    stop,
    policy,
    fault,
):
    """Checkpoint a successful audit before any complete publication.

    Resume verifies its exact metadata and compact bytes without another numeric
    scan. A crash before this checkpoint repeats the required global audit.
    """
    checkpoint = control / "finalized.json"
    if checkpoint.exists():
        saved = load(checkpoint)
        if saved["plan_sha256"] != json_sha256(plan):
            raise ValueError("finalized checkpoint differs from the producer plan")
        for raw in saved["artifacts"]:
            verify(files, Artifact.model_validate(raw))
        for raw in saved["subset_artifacts"]:
            verify(output / "subsets", Artifact.model_validate(raw))
        complete = read_manifest(files / saved["manifest"]["path"])
        if complete.model_dump() != {**manifest.model_dump(), "state": "complete"}:
            raise ValueError("finalized manifest differs from committed ledger")
        return complete, read_json(files / saved["audit_path"]), saved, None

    prefix = execution.prefix
    if (
        usage(output) + execution.metadata_reserve_bytes + execution.sdk_reserve_bytes
        > execution.staging_bytes
    ):
        raise ValueError("finalization metadata staging reservation unavailable")
    # Revalidate complete membership with a fresh outer Manifest. Nested typed
    # indexes already came from the hash-verified ledger and remain shared.
    complete = Manifest(
        provenance=manifest.provenance,
        shards=manifest.shards,
        expected=manifest.expected,
        state="complete",
    )
    catalog = build_catalog(files, complete, commits)
    catalog_path = f"{prefix}/catalog.json"
    # This is a plain catalog, independently byte-hashed for source binding.
    write_json(files / catalog_path, catalog.model_dump(), overwrite=True)
    fsync_dir((files / catalog_path).parent)
    part_rows = iter(
        row for c in commits for row in load(files / c["catalog"]["path"])["records"]
    )
    admission_rows = iter(
        row for c in commits for row in load(files / c["admission"]["path"])["records"]
    )
    summary = NumericSummary(execution.family_cohort)
    report = {}
    manifest_path = f"{prefix}/manifest.json"
    subset_artifacts = []
    if policy is not None:
        save(
            control / "subset-active.json", {"roots": ["subset-transaction", "subsets"]}
        )
        transaction = ExtractionTransaction(
            output / "subset-transaction", complete, catalog, policy, execution
        )
    else:
        transaction = None
    # Selection retains only its compact exact identities, so the full catalog
    # can be released before numeric/Arrow buffers are admitted.
    del catalog
    gc.collect()

    def visitor(record, source):
        if stop.requested:
            raise Stopped("stopped during numeric audit")
        summary.add(record, next(admission_rows))
        if summarize(record) != next(part_rows):
            raise ValueError("generation catalog differs from audited numeric data")
        if transaction is not None:
            transaction(record, source)

    def audited(audit_report):
        if next(part_rows, None) is not None or next(admission_rows, None) is not None:
            raise ValueError("catalog/admission has unaudited records")
        if stop.requested:
            raise Stopped("stopped after numeric audit")
        if transaction is not None:
            transaction.finish(audit_report)
        report.update(audit_report)

    started = time.monotonic()
    with transaction if transaction is not None else nullcontext():
        complete = finalize_manifest(
            files / manifest_path,
            manifest,
            root=files,
            resolver=publication.resolve,
            on_record=visitor,
            on_audit_complete=audited,
            overwrite=(files / manifest_path).exists(),
        )
        fault("audit")
    seconds = time.monotonic() - started
    if transaction is not None:
        (output / "subset-transaction").rename(output / "subsets")
        fsync_dir(output)
        extracted = load(output / "subsets" / "extracted.json")
        subset_paths = [
            "selection.json",
            "extracted.json",
            extracted["manifest"]["path"],
            *[s["path"] for s in extracted["shards"]],
        ]
        subset_artifacts = [
            artifact(output / "subsets", p).model_dump() for p in subset_paths
        ]
        fault("subset")
    summary_path, audit_path = f"{prefix}/summary.json", f"{prefix}/audit.json"
    save(files / summary_path, summary.finish(), overwrite=True)
    write_json(files / audit_path, report, overwrite=True)
    metadata = [
        artifact(files, p).model_dump()
        for p in (catalog_path, manifest_path, summary_path, audit_path)
    ]
    saved = {
        "plan_sha256": json_sha256(plan),
        "manifest": artifact(files, manifest_path).model_dump(),
        "audit_path": audit_path,
        "catalog_path": catalog_path,
        "summary_path": summary_path,
        "artifacts": metadata,
        "subset_artifacts": subset_artifacts,
    }
    save(checkpoint, saved)
    fault("finalized")
    recover_extraction(output, control)
    return complete, report, saved, seconds
