"""Verified one-shard-at-a-time access for downstream training and data audits."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from tennis3d.artifacts import (
    ArtifactProvenance,
    file_sha256,
    json_sha256,
    read_json,
    read_npz,
)
from tennis3d.schemas import read_result
from tennis3d.synthetic import EVENT_CODES, KINDS


def read_manifest(path: str | Path) -> dict:
    """Paths in a split manifest are relative to that manifest's directory."""
    path = Path(path)
    manifest = read_json(path)
    if manifest.get("schema_version") != "1.0" or manifest.get("split") not in (
        "train",
        "val",
        "test",
    ):
        raise ValueError("unsupported synthetic split manifest")
    seen = set()
    for record in manifest["records"]:
        identity = (record["rally_id"], record["view_id"])
        if identity in seen or record["split"] != manifest["split"]:
            raise ValueError("duplicate view or inconsistent split")
        seen.add(identity)
        for key in ("path", "rally_path", "truth_path"):
            file = Path(record[key])
            if file.is_absolute() or ".." in file.parts:
                raise ValueError("manifest artifact paths must stay within the dataset")
    return manifest


def load_view(
    manifest_path: str | Path, record: dict, *, manifest: dict | None = None
) -> dict[str, np.ndarray]:
    """Read one checked shard; repeated callers may reuse read_manifest's result.

    A supplied manifest is a previously validated snapshot. Its shard checksum
    and embedded provenance checks remain mandatory on every access.
    """
    snapshot = read_manifest(manifest_path) if manifest is None else manifest
    return _load_view(Path(manifest_path), snapshot, record)


def _check_provenance(
    provenance: ArtifactProvenance, manifest: dict, upstream: dict[str, str]
) -> None:
    if (
        provenance.source_sha256 != manifest["generator_sha256"]
        or provenance.config_sha256 != manifest["config_sha256"]
        or provenance.upstream_sha256 != upstream
    ):
        raise ValueError("artifact provenance differs from manifest")


def _load_view(path: Path, manifest: dict, record: dict) -> dict[str, np.ndarray]:
    arrays, provenance = read_npz(
        _check_file(path.parent, record["path"], record["sha256"])
    )
    _check_provenance(
        provenance,
        manifest,
        {"rally": record["rally_sha256"], "shot_pool": record["pool_sha256"]},
    )
    validate_view(arrays)
    return arrays


def iter_views(
    manifest_path: str | Path,
) -> Iterator[tuple[dict, dict[str, np.ndarray]]]:
    manifest = read_manifest(manifest_path)
    for record in manifest["records"]:
        yield record, _load_view(Path(manifest_path), manifest, record)


def validate_view(arrays: dict[str, np.ndarray]) -> None:
    t = arrays["t_s"]
    n = len(t)
    if (
        t.dtype != np.float64
        or n < 2
        or t[0] != 0
        or not np.isfinite(t).all()
        or not np.all(t[1:] > t[:-1])
    ):
        raise ValueError("invalid synthetic timestamps")
    segments = len(arrays["segment_kind"])
    events = len(arrays["event_t_s"])
    if segments < 1:
        raise ValueError("a view must reference at least one physical segment")
    for key, shape in {
        "t_s": (n,),
        "xyz_m": (n, 3),
        "velocity_m_s": (n, 3),
        "spin_rad_s": (n, 3),
        "ball_uv": (n, 2),
        "ball_clean_uv": (n, 2),
        "ball_candidate_uv": (n, 2),
        "ball_in_frame": (n,),
        "ball_confidence": (n,),
        "court_clean_uv": (n, 14, 2),
        "estimated_camera_K": (n, 3, 3),
        "image_size": (2,),
        "segment_id": (n,),
        "segment_kind": (segments,),
        "prior_id": (segments,),
        "event_t_s": (events,),
        "event_precision_s": (events,),
        "event_code": (events,),
        "event_xyz_m": (events, 3),
        "hit_before": (segments - 1, 9),
        "hit_after": (segments - 1, 9),
        "court_uv": (n, 14, 2),
        "camera_K": (n, 3, 3),
        "camera_R": (n, 3, 3),
        "camera_t": (n, 3),
        "ball_observed": (n,),
        "label_valid": (n,),
        "camera_valid": (n,),
    }.items():
        if arrays[key].shape != shape:
            raise ValueError(f"wrong {key} shape")
    for key in ("ball_observed", "ball_in_frame", "label_valid", "camera_valid"):
        if arrays[key].dtype != np.bool_:
            raise ValueError(f"{key} must be boolean")
    for key in ("segment_id", "segment_kind", "prior_id", "event_code", "image_size"):
        if arrays[key].dtype.kind not in "iu":
            raise ValueError(f"{key} must be integer")
    if (
        np.any(arrays["segment_id"] < 0)
        or np.any(arrays["segment_id"] >= segments)
        or np.any(arrays["segment_id"][1:] < arrays["segment_id"][:-1])
    ):
        raise ValueError("segment_id must be ordered and index the segment table")
    if not np.isin(arrays["segment_kind"], np.arange(len(KINDS))).all() or np.any(
        arrays["prior_id"] < 0
    ):
        raise ValueError("invalid shot kind or prior index")
    if not np.isin(arrays["event_code"], list(EVENT_CODES.values())).all():
        raise ValueError("unknown event code")
    event_times = arrays["event_t_s"]
    if (
        event_times.dtype != np.float64
        or not np.isfinite(event_times).all()
        or np.any(event_times[1:] < event_times[:-1])
        or np.any(event_times < 0)
        or np.any(event_times > t[-1])
    ):
        raise ValueError("event times must be ordered within the view timeline")
    if not np.isfinite(arrays["event_precision_s"]).all() or np.any(
        arrays["event_precision_s"] < 0
    ):
        raise ValueError("event precision must be finite and nonnegative")
    if np.count_nonzero(arrays["event_code"] == EVENT_CODES["hit"]) > segments - 1:
        raise ValueError("more hit events than physical hit states")
    for key in ("event_xyz_m", "hit_before", "hit_after"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"nonfinite {key}")
    for key in ("xyz_m", "velocity_m_s", "spin_rad_s"):
        if not np.isfinite(arrays[key][arrays["label_valid"]]).all():
            raise ValueError(f"nonfinite valid {key}")
    observed = arrays["ball_observed"]
    if np.any(observed & ~arrays["ball_in_frame"]) or np.any(arrays["image_size"] <= 0):
        raise ValueError("observation outside clean image mask or invalid image size")
    for key in ("ball_clean_uv", "ball_candidate_uv"):
        points = arrays[key]
        if not (np.isfinite(points).all(axis=1) | np.isnan(points).all(axis=1)).all():
            raise ValueError(f"{key} must contain finite pairs or NaN pairs")
        if not np.isfinite(points[observed]).all():
            raise ValueError(f"nonfinite observed {key}")
    clean = arrays["ball_clean_uv"]
    in_frame = (
        np.isfinite(clean).all(axis=1)
        & (clean >= 0).all(axis=1)
        & (clean < arrays["image_size"]).all(axis=1)
    )
    if not np.array_equal(in_frame, arrays["ball_in_frame"]):
        raise ValueError("clean projection disagrees with ball_in_frame")
    if not np.isfinite(arrays["ball_uv"][observed]).all():
        raise ValueError("nonfinite observed ball coordinates")
    if not np.isnan(arrays["ball_uv"][~observed]).all():
        raise ValueError("missing coordinates must use NaN and a false mask")
    confidence = arrays["ball_confidence"]
    if (
        not np.isfinite(confidence).all()
        or np.any(confidence < 0)
        or np.any(confidence > 1)
        or np.any(confidence[~observed] != 0)
    ):
        raise ValueError("invalid ball confidence")
    for key in (
        "court_uv",
        "court_clean_uv",
        "camera_K",
        "camera_R",
        "camera_t",
        "estimated_camera_K",
    ):
        if not np.isfinite(arrays[key][arrays["camera_valid"]]).all():
            raise ValueError(f"nonfinite valid {key}")
    if not np.array_equal(arrays["hit_before"][:, :3], arrays["hit_after"][:, :3]):
        raise ValueError("position discontinuity at racket hit")


def _check_file(root: Path, relative: str, expected: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("manifest artifact paths must stay within the dataset")
    path = root / path
    if file_sha256(path) != expected:
        raise ValueError(f"Artifact checksum mismatch: {path}")
    return path


def audit_splits(root: str | Path) -> dict:
    root = Path(root)
    checksums = read_json(root / "manifest.sha256.json")
    manifest = read_json(_check_file(root, "manifest.json", checksums["manifest.json"]))
    if (
        json_sha256(read_json(root / "config.resolved.json"))
        != manifest["config_sha256"]
    ):
        raise ValueError("Resolved config checksum mismatch")
    families, rallies, pools = set(), set(), set()
    pool_owners = {}
    pool_hashes = set()
    for pool in manifest["pools"]:
        family = pool["family_id"]
        if family in pool_owners or pool["sha256"] in pool_hashes:
            raise ValueError("duplicate root pool family or checksum")
        if (
            pool["split"] not in ("train", "val", "test")
            or pool["config"]["split"] != pool["split"]
        ):
            raise ValueError("root pool split mismatch")
        if not isinstance(pool["config"]["size"], int) or pool["config"]["size"] < 1:
            raise ValueError("invalid root pool size")
        pool_owners[family] = pool
        pool_hashes.add(pool["sha256"])
    result = {}
    for split in ("train", "val", "test"):
        entry = manifest["splits"][split]
        path = _check_file(root, entry["path"], entry["sha256"])
        split_manifest = read_manifest(path)
        if (
            split_manifest["split"] != split
            or split_manifest["config_sha256"] != manifest["config_sha256"]
            or split_manifest["generator_sha256"] != manifest["generator_sha256"]
        ):
            raise ValueError("split manifest identity/config mismatch")
        records = split_manifest["records"]
        checked = set()
        rally_owners = {}
        for record in records:
            owner = pool_owners.get(record["family_id"])
            if owner is None or (owner["split"], owner["sha256"]) != (
                split,
                record["pool_sha256"],
            ):
                raise ValueError(
                    "view family/pool ownership differs from root manifest"
                )
            identity = (
                record["rally_path"],
                record["rally_sha256"],
                record["pool_sha256"],
            )
            rally_identity = (record["family_id"], *identity)
            previous = rally_owners.setdefault(record["rally_id"], rally_identity)
            if previous != rally_identity:
                raise ValueError(
                    "physical rally has inconsistent family or artifact identity"
                )
            if identity not in checked:
                _, provenance = read_npz(
                    _check_file(
                        path.parent, record["rally_path"], record["rally_sha256"]
                    )
                )
                _check_provenance(
                    provenance, split_manifest, {"shot_pool": record["pool_sha256"]}
                )
                checked.add(identity)
            truth = read_result(
                _check_file(path.parent, record["truth_path"], record["truth_sha256"])
            )
            if (
                truth.source.sha256 != record["sha256"]
                or truth.provenance.config_sha256 != split_manifest["config_sha256"]
            ):
                raise ValueError("truth provenance differs from manifest")
        current = [
            {r[key] for r in records}
            for key in ("family_id", "rally_id", "pool_sha256")
        ]
        for existing, now in zip((families, rallies, pools), current):
            if existing & now:
                raise ValueError("cross-split physical rally/prior leakage")
            existing.update(now)
        frames = 0
        for record in records:
            arrays = _load_view(path, split_manifest, record)
            if np.any(
                arrays["prior_id"] >= pool_owners[record["family_id"]]["config"]["size"]
            ):
                raise ValueError("prior index exceeds owning pool size")
            frames += len(arrays["t_s"])
        result[split] = {
            "rallies": len(current[1]),
            "views": len(records),
            "frames": frames,
        }
        if any(
            entry[key] != result[split][key]
            for key in ("rallies", "views")
            if key in entry
        ):
            raise ValueError("split counts differ from root manifest")
    if families != set(pool_owners):
        raise ValueError("root manifest contains unused pool families")
    return result
