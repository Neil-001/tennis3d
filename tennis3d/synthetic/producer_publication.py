"""Producer orchestration around the verified dataset publisher."""

from contextlib import closing, contextmanager
from pathlib import Path

from tennis3d.artifacts import json_sha256, read_json
from tennis3d.datasets.io import read_manifest
from tennis3d.datasets.publication import (
    BoundedStaging,
    HuggingFaceBackend,
    Publisher,
    VerificationError,
    _commit,
)
from tennis3d.datasets.source import SourceSpec, hub_chunks
from tennis3d.synthetic.producer_state import load, verify


class Publication:
    def __init__(self, output, execution, backend=None):
        self.output, self.execution = Path(output), execution
        self.staging = BoundedStaging(
            self.output,
            execution.staging_bytes,
            metadata_reserve_bytes=execution.metadata_reserve_bytes,
        )
        self.publisher = None
        if execution.repo_id:
            self.publisher = Publisher(
                self.staging,
                backend or HuggingFaceBackend(self.staging),
                execution.repo_id,
                execution.revision,
            )

    def publish(self, spec, *, evict=False):
        if self.publisher:
            return self.publisher.publish(spec, evict=evict)
        return None

    def receipt(self, spec):
        ledger = self.publisher._load() if self.publisher else {}
        entries = ledger.get("entries", {})
        entry = entries.get(spec.path) if isinstance(entries, dict) else None
        if (
            not isinstance(entry, dict)
            or entry.get("state") != "verified"
            or any(
                entry.get(k) != v
                for k, v in spec.model_dump(
                    include={"path", "sha256", "size_bytes"}
                ).items()
            )
        ):
            raise ValueError("missing exact verified publication receipt")
        try:
            _commit(entry.get("commit"))
        except (VerificationError, TypeError):
            raise ValueError(
                "missing exact verified immutable publication receipt"
            ) from None
        return entry

    @contextmanager
    def resolve(self, spec):
        path = self.staging.local_path(spec)
        downloaded = not path.exists()
        if downloaded:
            entry = self.receipt(spec)
            # Only transport fields are consumed by hub_chunks. Staging, rather
            # than the training cache, owns admission and byte verification.
            source = SourceSpec(
                manifest=spec,
                split="train",
                root=str(self.staging.files),
                repo_id=self.execution.repo_id,
                revision=entry["commit"],
                timeout_seconds=30,
                transfer_seconds=600,
            )
            with closing(hub_chunks(source, spec)) as chunks:
                self.staging.stage(spec, chunks)
        else:
            verify(self.staging.files, spec)
        try:
            yield path
        finally:
            if downloaded and self.execution.evict:
                self.publish(spec, evict=True)

    def complete(
        self,
        manifest_spec,
        expected,
        audit_spec,
        *,
        manifest,
        audit_report,
        producer_spec,
    ):
        if not self.publisher:
            return None
        expected_specs = [
            s.model_dump(include={"path", "sha256", "size_bytes"}) for s in expected
        ]

        def validate(manifest_path, audit_path):
            actual = read_manifest(manifest_path)
            if (
                actual != manifest
                or actual.state != "complete"
                or read_json(audit_path) != audit_report
            ):
                raise ValueError(
                    "final release differs from the successful audited output"
                )
            producer = load(verify(self.staging.files, producer_spec))
            if producer["manifest_sha256"] != json_sha256(actual.model_dump()):
                raise ValueError("producer metadata is not bound to the final manifest")
            bound = producer["artifacts"]
            required = [s for s in expected_specs if s["path"] != producer_spec.path]
            if bound != required:
                raise ValueError("producer artifact set differs from publication")
            if audit_report["counts"] != {
                k: v.model_dump() for k, v in actual.expected.items()
            }:
                raise ValueError("audit counts differ from the planned release")

        return self.publisher.publish_release(
            manifest_spec, expected, audit_spec, validate
        )
