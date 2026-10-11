"""Publish an audit-extracted compact copy after its immutable main release."""

import gc
from pathlib import Path
from typing import cast

from tennis3d.artifacts import json_sha256, read_json, write_json
from tennis3d.datasets.io import read_manifest
from tennis3d.datasets.publication import (
    ArtifactLike,
    BoundedStaging,
    HuggingFaceBackend,
    Publisher,
)
from tennis3d.datasets.schema import Artifact, count_records
from tennis3d.datasets.subset import (
    ExtractedSubset,
    bind_release,
    read_descriptor,
    write_descriptor,
)
from tennis3d.synthetic.producer_state import artifact, load, save_once, usage, verify


def _chunks(path):
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            yield chunk


def publish_subset(
    output,
    execution,
    main_commit,
    main_manifest,
    main_catalog,
    main_files,
    audit_report,
    *,
    backend=None,
    check_stop=None,
):
    """Publish compact artifacts last-manifest, using the trusted main audit.

    The parent persists main_commit before invoking this helper. A local-only
    main release returns pending binding; no substitute commit is manufactured.
    This checks metadata and artifact bytes, never opens numeric source shards.
    """
    check_stop = check_stop or (lambda: None)
    check_stop()
    output, main_files = Path(output), Path(main_files)
    root = output / "subsets"
    extracted = ExtractedSubset.model_validate(load(root / "extracted.json"))
    if load(root / "selection.json") != extracted.selection.model_dump():
        raise ValueError("compact selection checkpoint differs from extraction")
    main_manifest = Artifact.model_validate(main_manifest)
    main_catalog = Artifact.model_validate(main_catalog)
    source_path = verify(main_files, main_manifest)
    catalog_path = verify(main_files, main_catalog)
    source = read_manifest(source_path)
    expected_report = {
        "schema_version": "2.0",
        "shards": len(source.shards),
        "counts": {s: c.model_dump() for s, c in count_records(source.shards).items()},
    }
    if source.state != "complete" or audit_report != expected_report:
        raise ValueError(
            "successful complete main audit required for compact publication"
        )
    if (
        json_sha256(source.model_dump()) != extracted.selection.source_manifest_sha256
        or json_sha256(read_json(catalog_path)) != extracted.selection.catalog_sha256
        or json_sha256(source.provenance.model_dump())
        != extracted.source_provenance_sha256
    ):
        raise ValueError("compact extraction differs from main source metadata")
    compact = read_manifest(verify(root, extracted.manifest))
    if (
        compact.state != "complete"
        or compact.shards != extracted.shards
        or compact.provenance != source.provenance
        or compact.expected != count_records(extracted.shards)
    ):
        raise ValueError("compact manifest differs from finished extraction")
    # bind_release reads and validates its own full source manifest/catalog.
    # Release this copy before that allocation while the caller keeps its main.
    del source
    gc.collect()
    prefix = f"{execution.prefix}/search"
    if extracted.manifest.path != f"{prefix}/manifest.json" or any(
        not s.path.startswith(f"{prefix}/") for s in extracted.shards
    ):
        raise ValueError("compact artifacts differ from execution repository prefix")
    for shard in extracted.shards:
        verify(root, shard)
    if main_commit is None:
        return {
            "state": "extracted",
            "needs_binding": True,
            "manifest": extracted.manifest.model_dump(),
            "selection_sha256": json_sha256(extracted.selection.model_dump()),
            "extracted_sha256": json_sha256(extracted.model_dump()),
        }
    if not execution.repo_id or not execution.revision:
        raise ValueError("compact publication requires the main repository destination")
    check_stop()
    descriptor = bind_release(
        extracted,
        main_repo_id=execution.repo_id,
        main_commit=main_commit,
        main_manifest=main_manifest,
        main_catalog=main_catalog,
        main_manifest_path=source_path,
        main_catalog_path=catalog_path,
    )
    descriptor_relative = f"{prefix}/subset.json"
    descriptor_path = root / descriptor_relative
    descriptor_sha256 = json_sha256(descriptor.model_dump())
    if descriptor_path.exists():
        if (
            read_descriptor(descriptor_path, expected_sha256=descriptor_sha256)
            != descriptor
        ):
            raise ValueError("existing compact descriptor differs from release")
    else:
        write_descriptor(descriptor_path, descriptor)
    descriptor_spec = artifact(root, descriptor_relative)
    copy_report = {
        "schema_version": "compact-copy-audit-1",
        "main_commit": main_commit,
        "main_audit_report": audit_report,
        "source_manifest_sha256": extracted.selection.source_manifest_sha256,
        "source_catalog_sha256": extracted.selection.catalog_sha256,
        "selection_sha256": json_sha256(extracted.selection.model_dump()),
        "descriptor_sha256": descriptor_sha256,
        "compact_manifest_sha256": json_sha256(compact.model_dump()),
        "counts": {s: c.model_dump() for s, c in compact.expected.items()},
        "artifacts": [
            s.model_dump(include={"path", "sha256", "size_bytes"})
            for s in extracted.shards
        ],
    }
    audit_relative = f"{prefix}/copy-audit.json"
    audit_path = root / audit_relative
    if audit_path.exists():
        if read_json(audit_path) != copy_report:
            raise ValueError("existing compact copy audit differs from release")
    else:
        write_json(audit_path, copy_report)
    audit_spec = artifact(root, audit_relative)
    expected = [*extracted.shards, descriptor_spec, audit_spec]
    identity = {
        "repo_id": execution.repo_id,
        "revision": execution.revision,
        "main_commit": main_commit,
        "descriptor_sha256": descriptor_sha256,
        "extracted_sha256": json_sha256(extracted.model_dump()),
        "manifest": extracted.manifest.model_dump(),
        "expected": [
            s.model_dump(include={"path", "sha256", "size_bytes"}) for s in expected
        ],
        # Recovery includes these newly created files only through this binding.
        "local_artifacts": [
            Artifact(
                path=f"subsets/{spec.path}",
                sha256=spec.sha256,
                size_bytes=spec.size_bytes,
            ).model_dump()
            for spec in (descriptor_spec, audit_spec)
        ],
    }
    save_once(output / "control/subset-publication.json", identity)
    staging = BoundedStaging(
        output / "subset-publication",
        execution.staging_bytes,
        metadata_reserve_bytes=execution.metadata_reserve_bytes,
    )
    backend = backend or HuggingFaceBackend(staging)
    publisher = Publisher(staging, backend, execution.repo_id, execution.revision)
    for spec in (main_manifest, main_catalog):
        check_stop()
        if backend.fingerprint(execution.repo_id, main_commit, spec.path) != (
            spec.sha256,
            spec.size_bytes,
        ):
            raise ValueError("immutable main publication differs from bound metadata")
    result_path = output / "control/subset-publication-result.json"
    if result_path.exists():
        previous = load(result_path)
        expected_result = {
            "state": "complete",
            "needs_binding": False,
            "main_commit": main_commit,
            "commit": previous["commit"],
            "descriptor_sha256": descriptor_sha256,
            "descriptor": descriptor_spec.model_dump(),
            "manifest": extracted.manifest.model_dump(),
            "audit": audit_spec.model_dump(),
        }
        if previous != expected_result:
            raise ValueError("completed compact publication differs from release")
        for spec in [*expected, extracted.manifest]:
            check_stop()
            if backend.fingerprint(
                execution.repo_id, previous["commit"], spec.path
            ) != (spec.sha256, spec.size_bytes):
                raise ValueError("completed immutable compact publication changed")
        return previous

    def stage(spec):
        check_stop()
        staged = staging.files / spec.path
        if staged.exists() or staged.is_symlink():
            verify(staging.files, spec)
            additional_bytes = 0
        else:
            additional_bytes = spec.size_bytes
        # The compact original remains under subsets while the publisher stages a copy.
        if (
            usage(output) + additional_bytes + execution.metadata_reserve_bytes
            > execution.staging_bytes
        ):
            raise ValueError("staging cannot reserve compact publication copy")
        staging.stage(cast(ArtifactLike, spec), _chunks(verify(root, spec)))

    for spec in expected:
        # The publisher reads these protocol attributes; the artifacts are frozen.
        publication_spec = cast(ArtifactLike, spec)
        stage(spec)
        check_stop()
        # Keep only the small descriptor/audit files for the release validator.
        publisher.publish(
            publication_spec,
            evict=spec.path not in {descriptor_spec.path, audit_spec.path},
        )
    stage(extracted.manifest)

    def validate(manifest_path, audited_path):
        actual = read_manifest(manifest_path)
        bound = read_descriptor(
            verify(staging.files, descriptor_spec), expected_sha256=descriptor_sha256
        )
        if (
            actual != compact
            or actual.state != "complete"
            or read_json(audited_path) != copy_report
            or bound != descriptor
            or bound.subset.manifest != extracted.manifest
            or bound.subset.shards != actual.shards
            or copy_report["counts"]
            != {s: c.model_dump() for s, c in actual.expected.items()}
            or copy_report["artifacts"]
            != [
                s.model_dump(include={"path", "sha256", "size_bytes"})
                for s in actual.shards
            ]
        ):
            raise ValueError(
                "compact release differs from successful main audit extraction"
            )

    check_stop()
    commit = publisher.publish_release(
        cast(ArtifactLike, extracted.manifest),
        [cast(ArtifactLike, spec) for spec in expected],
        cast(ArtifactLike, audit_spec),
        validate,
    )
    result = {
        "state": "complete",
        "needs_binding": False,
        "main_commit": main_commit,
        "commit": commit,
        "descriptor_sha256": descriptor_sha256,
        "descriptor": descriptor_spec.model_dump(),
        "manifest": extracted.manifest.model_dump(),
        "audit": audit_spec.model_dump(),
    }
    save_once(result_path, result)
    return result
