"""Package-bound code revision and canonical content identity for run artifacts."""

import subprocess
from pathlib import Path

from tennis3d.artifacts import file_sha256, json_sha256


def code_revision() -> str:
    """Never attribute an installed wheel to a caller's unrelated Git checkout."""
    root = Path(__file__).resolve().parent.parent
    if not (root / ".git").exists() or not (root / "pyproject.toml").is_file():
        return "unavailable"
    try:
        top = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if Path(top).resolve() != root:
            return "unavailable"
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def code_identity() -> dict:
    """Hash actual package source bytes, including dirty/untracked candidate code.

    source_sha256 hashes a sorted canonical JSON mapping of package-relative file
    names to SHA-256. Resolved configuration has its own independent config hash.
    This works in a source checkout or wheel and does not require Git to be clean.
    """
    package = Path(__file__).resolve().parent
    files = {
        f"tennis3d/{path.relative_to(package).as_posix()}": file_sha256(path)
        for path in sorted(package.rglob("*.py"))
    }
    return {
        "code_revision": code_revision(),
        "source_sha256": json_sha256(files),
        "source_files": files,
    }
