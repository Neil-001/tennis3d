"""Frozen comparison settings and source identities, separate from lifter config."""

import subprocess
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from tennis3d.artifacts import file_sha256, json_sha256
from tennis3d.config import load_config
from tennis3d.schemas import StrictModel

EDITABLE = frozenset(
    {
        "tennis3d/lifting/model.py",
        "tennis3d/lifting/losses.py",
        "tennis3d/lifting/train.py",
        "configs/lifter.yaml",
    }
)


class Protocol(StrictModel):
    profile: Literal["standard", "smoke"] = "standard"
    train_manifest: Path = Path("data/autoresearch/train.manifest.json")
    val_manifest: Path = Path("data/autoresearch/val.manifest.json")
    seed: int = Field(default=0, ge=0)
    device: Literal["cpu", "cuda"] = "cuda"
    train_seconds: float = Field(default=300.0, gt=0)
    total_seconds: float = Field(default=600.0, gt=0)
    loss_interval_seconds: float = Field(default=2.0, gt=0)

    @model_validator(mode="after")
    def budgets(self):
        if self.profile == "standard":
            if self.train_seconds != 300 or self.total_seconds != 600:
                raise ValueError(
                    "standard trials require 300s training and a 600s total deadline"
                )
        elif self.train_seconds > 30 or self.total_seconds > 120:
            raise ValueError(
                "smoke protocols are limited to 30s training and 120s total"
            )
        if self.total_seconds <= self.train_seconds:
            raise ValueError(
                "total deadline must leave time for setup, evaluation and checks"
            )
        return self


def load_protocol(path: Path) -> Protocol:
    """Manifest paths are relative to the launch directory; freeze them absolute."""
    protocol = load_config(Protocol, path)
    return protocol.model_copy(
        update={
            "train_manifest": protocol.train_manifest.resolve(),
            "val_manifest": protocol.val_manifest.resolve(),
        }
    )


def checkout() -> Path:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode or Path(result.stdout.strip()).resolve() != root:
        raise ValueError("experiments require an editable tennis3d Git checkout")
    return root


def sources(root: Path) -> dict[str, str]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        check=True,
        capture_output=True,
    )
    paths = sorted({p.decode() for p in result.stdout.split(b"\0") if p})
    return {p: file_sha256(root / p) for p in paths if (root / p).is_file()}


def identities(root: Path) -> dict:
    files = sources(root)
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    candidate = {p: h for p, h in files.items() if p in EDITABLE}
    return {
        "revision": revision,
        "candidate_files": candidate,
        "candidate_sha256": json_sha256(candidate),
        "frozen_files": {p: h for p, h in files.items() if p not in EDITABLE},
    }


def verify_sources(
    root: Path, frozen: dict[str, str], candidate: dict[str, str] | None = None
) -> dict:
    current = identities(root)
    actual = current["frozen_files"]
    changes = sorted(
        p for p in frozen.keys() | actual.keys() if frozen.get(p) != actual.get(p)
    )
    if changes:
        raise ValueError(
            "frozen files changed; start a new segment after review: "
            + ", ".join(changes)
        )
    if candidate is not None and current["candidate_files"] != candidate:
        raise ValueError("candidate files changed during the trial")
    return current
