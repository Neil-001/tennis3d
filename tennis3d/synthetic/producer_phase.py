"""Bind the generation phase without checkpointing the global numeric audit."""

from itertools import islice

from tennis3d.artifacts import file_sha256, json_sha256
from tennis3d.datasets.schema import Manifest, Pool
from tennis3d.synthetic.producer_records import family_seed, plan_rallies
from tennis3d.synthetic.producer_state import artifact, load


def generation_boundary(control, plan, manifest):
    """Small ordered metadata bindings; numeric bytes remain in shards/pools."""
    return {
        "state": "generation-complete",
        "plan_sha256": json_sha256(plan),
        "provenance": artifact(control, "provenance.json").model_dump(),
        "commits": [
            artifact(control, f"commits/{index:06d}.json").model_dump()
            for index in range(len(manifest.shards))
        ],
        "pools": [
            artifact(control, f"pools/{split}-{index:06d}.json").model_dump()
            for split in ("train", "val", "test")
            for index in range(sum(p.split == split for p in manifest.provenance.pools))
        ],
    }


def validate_generation(control, boundary, manifest, cfg, execution, pool_cache=None):
    """Require every planned identity before finalization.

    Finalization also requires the original numeric pool files. It hashes them
    without planning family physics, decoding pools, or creating a simulator.
    """
    if pool_cache is not None:
        checkpoint = control / "generation-complete.json"
        if not checkpoint.is_file() or load(checkpoint) != boundary:
            raise ValueError("finalization requires matching complete generation state")
    provenance = manifest.provenance
    plan = load(control / "plan.json")
    if (
        provenance.generator_sha256 != plan["source_sha256"]
        or provenance.runtime_versions != plan["runtime_versions"]
        or provenance.config != plan["recipe"]
        or provenance.seed != cfg.seed
    ):
        raise ValueError("generation provenance differs from the plan")
    # Complete membership validation is metadata-only. Numeric success remains
    # the responsibility of the existing full global audit.
    Manifest(
        provenance=provenance,
        shards=manifest.shards,
        expected=manifest.expected,
        state="complete",
    )
    pools = {}
    for split in ("train", "val", "test"):
        for index in range(cfg.family_count(split)):
            checkpoint = control / f"pools/{split}-{index:06d}.json"
            saved = load(checkpoint)
            if not isinstance(saved, dict) or "pool" not in saved:
                raise ValueError("invalid generation pool checkpoint")
            pool = Pool.model_validate(saved["pool"])
            seed = family_seed(cfg.seed, split, index)
            if (
                pool.split != split
                or pool.family_id != f"{split}-family-{index:05d}-{seed}"
            ):
                raise ValueError("generation pool identity differs from the plan")
            pools[(split, index)] = pool
            if pool_cache is not None:
                name = f"{split}-{plan['pool_source_sha256']}-{json_sha256(pool.config)}.npz"
                path = pool_cache / name
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or file_sha256(path) != pool.sha256
                ):
                    raise ValueError(
                        "finalization requires original checksum-verified pool bytes"
                    )
    if list(pools.values()) != provenance.pools:
        raise ValueError("generation pool descriptors differ from provenance")
    planned = iter(plan_rallies(cfg, cohort_size=execution.family_cohort))
    for index, shard in enumerate(manifest.shards):
        batch = list(islice(planned, execution.shard_rallies))
        if (
            shard.path != f"{execution.prefix}/shards/{index:06d}.parquet"
            or shard.row_group_rallies != execution.row_group_rallies
            or len(shard.records) != len(batch)
        ):
            raise ValueError("generation shard layout differs from the plan")
        for record, rally in zip(shard.records, batch):
            pool = pools[(rally.split, rally.family_index)]
            if (
                record.rally_id,
                record.split,
                record.seed,
                record.family_id,
                record.pool_sha256,
            ) != (
                rally.rally_id,
                rally.split,
                rally.seed,
                pool.family_id,
                pool.sha256,
            ) or [v.view_id for v in record.views] != list(range(cfg.views_per_rally)):
                raise ValueError(
                    "generation record order/identity differs from the plan"
                )
    if next(planned, None) is not None:
        raise ValueError("generation state is partial")
