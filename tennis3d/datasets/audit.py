"""Physical, projection and global membership audits without whole-data residency."""

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tennis3d.court.model import court_landmarks
from tennis3d.datasets.arrays import RallyRecord, validate_record
from tennis3d.datasets.io import ShardReader, write_manifest
from tennis3d.datasets.schema import Manifest, count_records, local_path
from tennis3d.synthetic import EVENT_CODES, KINDS


@dataclass(frozen=True)
class AuditedRecordSource:
    shard_path: str
    shard_sha256: str
    row_index: int
    split: str
    rally_id: str
    view_ids: tuple[int, ...]


RecordVisitor = Callable[[RallyRecord, AuditedRecordSource], None]


def _physical(a):
    t, n = a["t_s"], len(a["t_s"])
    segments, events = len(a["segment_kind"]), len(a["event_t_s"])
    if (
        n < 2
        or segments < 1
        or t[0] != 0
        or not np.isfinite(t).all()
        or not np.all(np.diff(t) > 0)
    ):
        raise ValueError("invalid physical timestamps/segments")
    lengths = {
        "xyz_m": n,
        "velocity_m_s": n,
        "spin_rad_s": n,
        "label_valid": n,
        "segment_id": n,
        "prior_id": segments,
        "event_code": events,
        "event_xyz_m": events,
        "event_precision_s": events,
        "hit_before": segments - 1,
        "hit_after": segments - 1,
    }
    if any(len(a[k]) != size for k, size in lengths.items()):
        raise ValueError("physical array length mismatch")
    for key, value in a.items():
        supported = (
            value[a["label_valid"]]
            if key in {"xyz_m", "velocity_m_s", "spin_rad_s"}
            else value
        )
        if not np.isfinite(supported).all():
            raise ValueError(f"nonfinite supported physical {key}")
    if (
        np.any(a["segment_id"] < 0)
        or np.any(a["segment_id"] >= segments)
        or np.any(np.diff(a["segment_id"]) < 0)
        or not np.isin(a["segment_kind"], np.arange(len(KINDS))).all()
        or np.any(a["prior_id"] < 0)
        or not np.isin(a["event_code"], list(EVENT_CODES.values())).all()
        or np.any(np.diff(a["event_t_s"]) < 0)
        or np.any(a["event_t_s"] < 0)
        or np.any(a["event_t_s"] > t[-1])
        or np.any(a["event_precision_s"] < 0)
    ):
        raise ValueError("invalid physical segment/event table")
    if not np.array_equal(a["hit_before"][:, :3], a["hit_after"][:, :3]):
        raise ValueError("physical racket position discontinuity")
    if np.count_nonzero(a["event_code"] == EVENT_CODES["hit"]) != segments - 1:
        raise ValueError("physical hit count differs from segment table")


def _project(K, R, t, xyz):
    camera = np.einsum("nij,nkj->nki", R, xyz) + t[:, None, :]
    pixel = np.einsum("nij,nkj->nki", K, camera)
    valid = camera[..., 2] > 0
    uv = np.full((*valid.shape, 2), np.nan)
    uv[valid] = pixel[valid, :2] / pixel[valid, 2, None]
    return uv


def audit_rally(record: RallyRecord) -> None:
    """Verify stored truths, never adjust masks or repair labels."""
    validate_record(record)
    p = record.physical
    _physical(p)
    for view in record.views:
        a, origin = view.arrays, view.metadata.physical_time_origin_s
        times = a["t_s"] + origin
        # Subtracting the excerpt origin and adding it back may round one ULP.
        indices = np.searchsorted(p["t_s"], times)
        indices = np.clip(indices, 0, len(p["t_s"]) - 1)
        before = np.maximum(indices - 1, 0)
        indices = np.where(
            abs(p["t_s"][before] - times) < abs(p["t_s"][indices] - times),
            before,
            indices,
        )
        if not np.allclose(p["t_s"][indices], times, rtol=0, atol=1e-12) or np.any(
            np.diff(indices) != 1
        ):
            raise ValueError("view timestamps do not form a physical excerpt")
        for key in ("xyz_m", "velocity_m_s", "spin_rad_s", "label_valid", "segment_id"):
            if not np.array_equal(a[key], p[key][indices], equal_nan=True):
                raise ValueError(f"view/physical {key} mismatch")
        for key in ("segment_kind", "prior_id", "hit_before", "hit_after"):
            if not np.array_equal(a[key], p[key]):
                raise ValueError(f"view/physical {key} mismatch")
        select = (p["event_t_s"] >= p["t_s"][indices[0]]) & (
            p["event_t_s"] <= p["t_s"][indices[-1]]
        )
        for key in ("event_t_s", "event_code", "event_xyz_m", "event_precision_s"):
            expected = p[key][select] - origin if key == "event_t_s" else p[key][select]
            if not np.array_equal(a[key], expected):
                raise ValueError(f"view/physical {key} mismatch")
        mask = a["camera_valid"]
        K, R, t = a["camera_K"][mask], a["camera_R"][mask], a["camera_t"][mask]
        if (
            not np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=1e-8)
            or not np.allclose(np.linalg.det(R), 1, atol=1e-8)
            or not np.all(K[:, (0, 1), (0, 1)] > 0)
            or not np.allclose(K[:, 2], [0, 0, 1], rtol=0, atol=1e-12)
            or not np.allclose(K[:, 1, 0], 0, rtol=0, atol=1e-12)
        ):
            raise ValueError("invalid saved camera geometry")
        # Missing XYZ labels cannot provide a reference projection. Preserve the
        # independently saved observations and continue checking camera/court data.
        ball_mask = mask & a["label_valid"]
        ball = _project(
            a["camera_K"][ball_mask],
            a["camera_R"][ball_mask],
            a["camera_t"][ball_mask],
            a["xyz_m"][ball_mask, None, :],
        )[:, 0]
        if not np.allclose(
            a["ball_clean_uv"][ball_mask], ball, rtol=1e-12, atol=1e-9, equal_nan=True
        ):
            raise ValueError("saved ball_clean_uv projection mismatch")
        court = np.broadcast_to(court_landmarks(), (len(K), 14, 3))
        clean = _project(K, R, t, court)
        estimated = _project(a["estimated_camera_K"][mask], R, t, court)
        for name, value in (
            ("court_clean_uv", clean),
            ("court_uv", estimated),
        ):
            if not np.allclose(
                a[name][mask], value, rtol=1e-12, atol=1e-9, equal_nan=True
            ):
                raise ValueError(f"saved {name} projection mismatch")


def _make_arrays_read_only(record: RallyRecord) -> None:
    for arrays in (record.physical, *(view.arrays for view in record.views)):
        for array in arrays.values():
            array.flags.writeable = False


def audit_dataset(
    root: str | Path,
    manifest: Manifest,
    *,
    resolver=None,
    on_record: RecordVisitor | None = None,
) -> dict:
    """Resolver(shard) is a context manager yielding one local immutable path.

    Default resolution uses root/shard.path. The producer or streaming tools may
    supply a bounded pinned remote cache, releasing each shard when its context
    exits. Every audit opens a fresh verified reader and covers every numeric row,
    including physical data. After those checks, on_record receives the borrowed
    record with read-only arrays and its original manifest source identity. Its
    exceptions abort the audit.
    """
    manifest = Manifest.model_validate(manifest.model_dump())
    for shard in manifest.shards:
        context = (
            resolver(shard)
            if resolver
            else nullcontext(local_path(Path(root), shard.path))
        )
        with (
            context as path,
            ShardReader(path, shard, provenance=manifest.provenance) as reader,
        ):
            for row in range(len(shard.records)):
                record = reader.read(row)
                audit_rally(record)
                if on_record is not None:
                    _make_arrays_read_only(record)
                    index = shard.records[row]
                    on_record(
                        record,
                        AuditedRecordSource(
                            shard_path=shard.path,
                            shard_sha256=shard.sha256,
                            row_index=row,
                            split=index.split,
                            rally_id=index.rally_id,
                            view_ids=tuple(view.view_id for view in index.views),
                        ),
                    )
    return {
        "schema_version": "2.0",
        "shards": len(manifest.shards),
        "counts": {
            s: v.model_dump() for s, v in count_records(manifest.shards).items()
        },
    }


def finalize_manifest(
    path: str | Path,
    manifest: Manifest,
    *,
    root: str | Path | None = None,
    resolver=None,
    on_record: RecordVisitor | None = None,
    on_audit_complete: Callable[[dict], None] | None = None,
    overwrite: bool = False,
) -> Manifest:
    """Publish a complete local manifest only after exact counts and full audits."""
    complete = Manifest.model_validate({**manifest.model_dump(), "state": "complete"})
    report = audit_dataset(
        Path(path).parent if root is None else root,
        complete,
        resolver=resolver,
        on_record=on_record,
    )
    if on_audit_complete is not None:
        on_audit_complete(report)
    write_manifest(path, complete, overwrite=overwrite)
    return complete
