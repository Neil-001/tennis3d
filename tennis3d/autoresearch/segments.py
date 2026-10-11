"""Freeze a comparison segment and retain its candidate source outside Git reverts."""

import re
import subprocess
from pathlib import Path

from tennis3d.artifacts import atomic_file, json_sha256, read_json, write_json
from tennis3d.autoresearch.inputs import DEFAULT_LOSS_LABEL
from tennis3d.autoresearch.log import append, snapshot
from tennis3d.autoresearch.protocol import Protocol, identities, verify_sources


def safe_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", name):
        raise ValueError(
            "trial/segment names need 1-101 letters, digits, dots, hyphens or underscores"
        )
    return name


def durable_directory(root: Path, run: Path) -> None:
    """Allow an external directory or ignored run data, never tracked source."""
    resolved = run.resolve()
    if resolved.is_relative_to(root.resolve()):
        relative = str(resolved.relative_to(root.resolve()) / "artifact.json")
        ignored = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "-q", relative], check=False
        )
        if ignored.returncode != 0:
            raise ValueError(
                "experiment artifacts must be outside Git or in an ignored run directory"
            )


def archive_candidate(root: Path, output: Path, identity: dict) -> None:
    """Exact source bytes make an uncommitted candidate recoverable after a revert."""
    for relative in identity["candidate_files"]:
        path = output / "candidate" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # Bytes were hashed before this call and are checked again by the caller.
        with atomic_file(path) as stream:
            stream.write((root / relative).read_bytes())
    write_json(
        output / "candidate.json",
        {k: identity[k] for k in ("revision", "candidate_files", "candidate_sha256")},
    )


def create_segment(
    root: Path,
    run: Path,
    name: str,
    protocol: Protocol,
    *,
    data: dict,
    hardware: dict,
    training: dict | None = None,
    loss_label: str = DEFAULT_LOSS_LABEL,
) -> dict:
    """Caller holds .trial.lock. Data includes the final checksummed manifest identity."""
    safe_name(name)
    durable_directory(root, run)
    state = snapshot(run)
    if any(t["status"] == "running" for t in state["trials"]):
        raise ValueError("reconcile prior attempts before creating a segment")
    if any(s["segment"] == name for s in state["segments"]):
        raise ValueError("segment already exists")
    identity = identities(root)
    record = {
        "segment": name,
        "profile": protocol.profile,
        "protocol": protocol.model_dump(mode="json"),
        "data": data,
        "hardware": hardware,
        "training": training,
        "loss_label": loss_label,
        "frozen_files": identity["frozen_files"],
        "reference": {
            k: identity[k] for k in ("revision", "candidate_files", "candidate_sha256")
        },
    }
    record["sha256"] = json_sha256(record)
    directory = run / "segments" / name
    directory.mkdir(parents=True, exist_ok=False)
    archive_candidate(root, directory, identity)
    verify_sources(root, identity["frozen_files"], identity["candidate_files"])
    write_json(directory / "segment.json", record)
    append(run, "segment", **record)
    return record


def read_segment(run: Path, name: str | None = None) -> dict:
    segments = snapshot(run)["segments"]
    if not segments:
        raise ValueError("initialize a comparison segment before running a trial")
    event = next(
        (s for s in reversed(segments) if name is None or s["segment"] == name), None
    )
    if event is None:
        raise ValueError("unknown comparison segment")
    record = read_json(run / "segments" / safe_name(event["segment"]) / "segment.json")
    digest = json_sha256({k: v for k, v in record.items() if k != "sha256"})
    if digest != record.get("sha256") or digest != event.get("sha256"):
        raise ValueError("segment configuration differs from its durable log identity")
    return record


def publish_best(run: Path) -> None:
    """Regenerate the convenience pointer from durable kept events, including on resume."""
    for segment, best in snapshot(run)["best"].items():
        write_json(
            run / "segments" / safe_name(segment) / "best.json", best, overwrite=True
        )
