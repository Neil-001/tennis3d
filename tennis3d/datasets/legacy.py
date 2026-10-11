"""Import genuine v1 NPZ data without resimulation or viewer JSON duplication."""

from pathlib import Path

from tennis3d.artifacts import file_sha256, read_json, read_npz
from tennis3d.datasets.arrays import RallyRecord, ViewRecord
from tennis3d.datasets.audit import finalize_manifest
from tennis3d.datasets.io import TARGET_BYTES, ShardWriter, write_manifest
from tennis3d.datasets.schema import (
    Counts,
    Manifest,
    Provenance,
    RallyMetadata,
    ViewMetadata,
)
from tennis3d.synthetic.data import audit_splits, load_view, read_manifest


def convert_legacy(
    source: str | Path,
    output: str | Path,
    *,
    target_bytes: int = TARGET_BYTES,
    row_group_rallies: int = 16,
) -> Manifest:
    """Retain the original source/config/runtime/seed identities, never claim regeneration."""
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("conversion output must be empty")
    counts = audit_splits(source)
    raw = read_json(source / "manifest.json")
    provenance = Provenance(
        generator_version=raw["generator_version"],
        generator_sha256=raw["generator_sha256"],
        code_revision=raw["code_revision"],
        runtime_versions=raw["runtime_versions"],
        config_sha256=raw["config_sha256"],
        config=read_json(source / "config.resolved.json"),
        seed=raw["seed"],
        pools=raw["pools"],
    )
    expected = {
        s: Counts(rallies=v["rallies"], views=v["views"], frames=v["frames"])
        for s, v in counts.items()
    }
    output.mkdir(parents=True, exist_ok=True)
    shards = []
    manifest_path = output / "manifest.json"

    def snapshot():
        return Manifest(provenance=provenance, expected=expected, shards=list(shards))

    write_manifest(manifest_path, snapshot())
    writer = None
    try:
        for split in ("train", "val", "test"):
            path = source / f"{split}.manifest.json"
            manifest = read_manifest(path)
            groups = {}
            for row in manifest["records"]:
                groups.setdefault(row["rally_id"], []).append(row)
            for records in groups.values():
                first = records[0]
                physical_path = source / first["rally_path"]
                if file_sha256(physical_path) != first["rally_sha256"]:
                    raise ValueError("physical rally checksum mismatch")
                physical, _ = read_npz(physical_path)
                rally = RallyRecord(
                    RallyMetadata(**{k: first[k] for k in RallyMetadata.model_fields}),
                    physical,
                    [
                        ViewRecord(
                            ViewMetadata(
                                **{k: r[k] for k in ViewMetadata.model_fields}
                            ),
                            load_view(path, r, manifest=manifest),
                        )
                        for r in records
                    ],
                )
                if writer is None:
                    relative = f"shards/part-{len(shards):05d}.parquet"
                    writer = ShardWriter(
                        output / relative,
                        provenance,
                        target_bytes=target_bytes,
                        row_group_rallies=row_group_rallies,
                        relative_path=relative,
                    )
                writer.append(rally)
                if writer.target_reached:
                    shards.append(writer.close())
                    writer = None
                    write_manifest(manifest_path, snapshot(), overwrite=True)
        if writer is not None:
            shards.append(writer.close())
            writer = None
        return finalize_manifest(manifest_path, snapshot(), overwrite=True)
    finally:
        if writer is not None:
            writer.abort()
