"""Select only producer-owned durable state for portable recovery."""

from pathlib import Path

from tennis3d.artifacts import json_sha256
from tennis3d.datasets.publication import (
    BoundedStaging,
    HuggingFaceBackend,
    Publisher,
    VerificationError,
    _commit,
)
from tennis3d.datasets.schema import Artifact
from tennis3d.synthetic.producer_recovery import create_bundle
from tennis3d.synthetic.producer_state import coordinator_lock, load, verify


def _receipts(output, plan, max_bytes):
    destination = plan["publication"]
    if not destination["repo_id"] or not destination["revision"]:
        raise ValueError("missing exact verified publication receipt")
    # Only parse the dataset publisher's ledger. Constructors do not resolve
    # credentials or contact the backend. Read at most the existing ledger plus
    # one byte.
    ledger_path = output / ".publication/receipts.json"
    ledger_bytes = ledger_path.stat().st_size if ledger_path.exists() else 0
    if ledger_bytes > max_bytes:
        raise ValueError("publication receipt ledger exceeds bundle cap")
    reserve = max(1024, 2 * ledger_bytes)
    staging = BoundedStaging(output, reserve + 1, metadata_reserve_bytes=reserve)
    return Publisher(
        staging,
        HuggingFaceBackend(staging),
        destination["repo_id"],
        destination["revision"],
    )._load()


def _verified_receipt(receipts, spec):
    entries = receipts.get("entries", {})
    if not isinstance(entries, dict):
        raise VerificationError("invalid publication receipt entries")
    entry = entries.get(spec.path)
    if (
        not isinstance(entry, dict)
        or entry.get("state") != "verified"
        or any(entry.get(k) != v for k, v in spec.model_dump().items())
        or not isinstance(entry.get("commit"), str)
    ):
        raise ValueError("missing exact verified publication receipt")
    return _commit(entry["commit"])


def recovery_bundle(output: Path, destination: Path, max_bytes: int):
    output = Path(output).absolute()
    with coordinator_lock(output):
        control = output / "control"
        if (control / "active.json").exists():
            raise ValueError("active generation must drain before recovery packaging")
        if (control / "subset-active.json").exists():
            raise ValueError("active extraction must drain before recovery packaging")
        plan = load(control / "plan.json")
        paths = []
        receipts = None
        for name in (
            "plan.json",
            "provenance.json",
            "failures.json",
            "last-failure.json",
            "incomplete.manifest.json",
            "performance.json",
            "timings.json",
            "result.json",
            "finalized.json",
            "generation-complete.json",
            "main-publication.json",
        ):
            if (control / name).is_file():
                paths.append(f"control/{name}")
        for directory in ("commits", "pools"):
            paths.extend(
                p.relative_to(output).as_posix()
                for p in sorted((control / directory).glob("*.json"))
            )
        # Every durable shard has an exact descriptor. Evicted shards are resolved
        # through verified receipts; pools are deterministically rebuilt and their
        # recorded hashes rechecked before the first resumed physical sample.
        for checkpoint in sorted((control / "commits").glob("*.json")):
            commit = load(checkpoint)
            for key in ("shard", "catalog", "admission"):
                spec = Artifact.model_validate(
                    {k: commit[key][k] for k in ("path", "sha256", "size_bytes")}
                )
                local = output / "files" / spec.path
                if key != "shard" or local.exists() or local.is_symlink():
                    verify(output / "files", spec)
                    paths.append(f"files/{spec.path}")
                else:
                    if receipts is None:
                        receipts = _receipts(output, plan, max_bytes)
                    _verified_receipt(receipts, spec)
        boundary_path = control / "generation-complete.json"
        if boundary_path.exists():
            boundary = load(boundary_path)
            if boundary["state"] != "generation-complete" or boundary[
                "plan_sha256"
            ] != json_sha256(plan):
                raise ValueError("generation boundary differs from the producer plan")
            for raw in [
                boundary["provenance"],
                *boundary["commits"],
                *boundary["pools"],
            ]:
                spec = Artifact.model_validate(raw)
                verify(control, spec)
                if f"control/{spec.path}" not in paths:
                    raise ValueError("generation boundary references unarchived state")
        prefix = plan["layout"]["prefix"]
        for name in (
            "catalog/index.json",
            "catalog.json",
            "audit.json",
            "summary.json",
            "manifest.json",
            "producer.json",
            "README.md",
            "LICENSE",
        ):
            relative = f"files/{prefix}/{name}"
            if (output / relative).is_file():
                paths.append(relative)
        receipt = ".publication/receipts.json"
        if (output / receipt).is_file():
            paths.append(receipt)
        if (control / "finalized.json").exists():
            finalized = load(control / "finalized.json")
            if finalized["plan_sha256"] != json_sha256(plan):
                raise ValueError("finalized checkpoint differs from the producer plan")
            for raw in [finalized["manifest"], *finalized["artifacts"]]:
                spec = Artifact.model_validate(raw)
                verify(output / "files", spec)
                paths.append(f"files/{spec.path}")
            for raw in finalized["subset_artifacts"]:
                spec = Artifact.model_validate(raw)
                verify(output / "subsets", spec)
                paths.append(f"subsets/{spec.path}")
        elif (
            (control / "main-publication.json").exists()
            or (control / "subset-publication.json").exists()
            or (control / "subset-publication-result.json").exists()
            or (
                (control / "result.json").exists()
                and load(control / "result.json").get("state") == "complete"
            )
        ):
            raise ValueError("published output is missing its finalized checkpoint")
        main_checkpoint = control / "main-publication.json"
        main = load(main_checkpoint) if main_checkpoint.exists() else None
        if main is not None:
            if main["manifest"] != finalized["manifest"]:
                raise ValueError("main publication checkpoint differs from release")
            if plan["publication"]["repo_id"]:
                # A verified receipt commit is always a full immutable SHA.
                if receipts is None:
                    receipts = _receipts(output, plan, max_bytes)
                if (
                    _verified_receipt(
                        receipts, Artifact.model_validate(main["manifest"])
                    )
                    != main["commit"]
                ):
                    raise ValueError("main immutable commit differs from receipt")
            elif main["commit"] is not None:
                raise ValueError("local publication cannot have an immutable commit")
        # Compact publication owns only its explicit hash-bound ledger and
        # descriptor checkpoints. SDK cache and credentials are never included.
        for relative in (
            "subset-publication/.publication/receipts.json",
            "control/subset-publication.json",
            "control/subset-publication-result.json",
        ):
            if (output / relative).is_file():
                paths.append(relative)
        compact_checkpoint = control / "subset-publication.json"
        if compact_checkpoint.exists():
            compact = load(compact_checkpoint)
            if (
                main is None
                or main["commit"] is None
                or compact["main_commit"] != main["commit"]
                or any(compact[k] != v for k, v in plan["publication"].items())
                or compact["manifest"] not in finalized["subset_artifacts"]
            ):
                raise ValueError("compact publication differs from main publication")
            for raw in compact["local_artifacts"]:
                spec = Artifact.model_validate(raw)
                verify(output, spec)
                paths.append(spec.path)
        compact_result = control / "subset-publication-result.json"
        if compact_result.exists():
            result = load(compact_result)
            if not compact_checkpoint.exists() or any(
                result[k] != compact[k]
                for k in ("main_commit", "manifest", "descriptor_sha256")
            ):
                raise ValueError("completed compact publication differs from release")
            if not isinstance(result["commit"], str):
                raise ValueError("compact publication requires an immutable commit")
            _commit(result["commit"])
            compact_receipts = _receipts(output / "subset-publication", plan, max_bytes)
            if (
                _verified_receipt(
                    compact_receipts, Artifact.model_validate(compact["manifest"])
                )
                != result["commit"]
            ):
                raise ValueError("compact immutable commit differs from receipt")
        return create_bundle(output, sorted(set(paths)), Path(destination), max_bytes)
