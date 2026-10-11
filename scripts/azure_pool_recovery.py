"""Copy original planned pool bytes alongside the producer's descriptor bundle."""

import argparse
import shutil
import stat
from pathlib import Path

from tennis3d.artifacts import atomic_file, canonical_json, file_sha256, json_sha256
from tennis3d.synthetic.config import SimulationConfig
from tennis3d.synthetic.producer_records import plan_families
from tennis3d.synthetic.producer_state import load, save


def regular(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("pool recovery requires regular files")


def families(output):
    plan = load(Path(output) / "control/plan.json")
    cfg = SimulationConfig.model_validate(plan["recipe"])
    planned = {
        family.cache_name: family
        for family in plan_families(
            cfg, plan["pool_source_sha256"], plan["runtime_versions"]
        )
    }
    return plan, planned


def checkpoint(output, family):
    path = (
        Path(output)
        / "control/pools"
        / (f"{family.split}-{family.family_index:06d}.json")
    )
    if not path.exists() and not path.is_symlink():
        return None
    regular(path)
    pool = load(path)["pool"]
    if pool["family_id"] != family.family_id or pool["config"] != family.pool_config:
        raise ValueError("pool checkpoint differs from planned family/config")
    return pool


def directory(path):
    if path.is_symlink() or not path.is_dir():
        raise ValueError("pool recovery requires a regular directory")


def select(output, cache):
    """Ignore only planned lock/writer temporaries; never transport them."""
    plan, planned = families(output)
    directory(cache)
    locks = {Path(name).with_suffix(".lock").name for name in planned}
    temporary_prefixes = tuple(f".{name}." for name in planned)
    for path in cache.iterdir():
        regular(path)
        if not (
            path.name in planned
            or path.name in locks
            or path.name.startswith(temporary_prefixes)
        ):
            raise ValueError("unplanned file in numeric pool cache")
    files = {}
    for name, family in planned.items():
        pool = checkpoint(output, family)
        path = cache / name
        if not path.exists():
            if pool is not None:
                raise ValueError("checkpointed original pool file is missing")
            continue
        digest = file_sha256(path)
        if pool is not None and digest != pool["sha256"]:
            raise ValueError("original pool bytes differ from durable checkpoint")
        files[name] = {
            "sha256": digest,
            "size_bytes": path.stat().st_size,
            "checkpointed": pool is not None,
        }
    if not files:
        raise ValueError("no completed planned pool files to preserve")
    return {
        "schema": "original-pools-v1",
        "plan_sha256": json_sha256(plan),
        "files": files,
    }


def verify(output, cache, inventory, *, all_checkpointed=False):
    plan, planned = families(output)
    directory(cache)
    if (
        inventory["schema"] != "original-pools-v1"
        or inventory["plan_sha256"] != json_sha256(plan)
        or not inventory["files"]
    ):
        raise ValueError("original pool inventory differs from producer plan")
    for name, expected in inventory["files"].items():
        if name not in planned:
            raise ValueError("inventory contains an unplanned pool name")
        path = cache / name
        regular(path)
        if (
            path.stat().st_size != expected["size_bytes"]
            or file_sha256(path) != expected["sha256"]
        ):
            raise ValueError("transferred original pool size/hash mismatch")
        pool = checkpoint(output, planned[name])
        if (all_checkpointed or expected["checkpointed"]) and pool is None:
            raise ValueError("resumed pool lacks its durable checkpoint")
        if pool is not None and pool["sha256"] != expected["sha256"]:
            raise ValueError("resumed checkpoint replaced original pool bytes")


def copy(output, source, destination, inventory, max_bytes):
    verify(output, source, inventory)
    # A plan hash stays fixed while pools finish. An older inventory must not
    # omit any checkpointed original bytes from the bundle being restored.
    _, planned = families(output)
    for name, family in planned.items():
        if name not in inventory["files"] and checkpoint(output, family) is not None:
            raise ValueError("inventory omits a checkpointed original pool")
    size = sum(item["size_bytes"] for item in inventory["files"].values())
    if size + len(canonical_json(inventory)) + 256 > max_bytes:
        raise ValueError("original pool copy exceeds recovery allowance")
    destination.mkdir(parents=True, exist_ok=False)
    for name in inventory["files"]:
        with (
            (source / name).open("rb") as incoming,
            atomic_file(destination / name) as outgoing,
        ):
            shutil.copyfileobj(incoming, outgoing, 1 << 20)
    verify(output, destination, inventory)
    return size


def snapshot(output, cache, destination, inventory_path, max_bytes):
    inventory = select(output, cache)
    copy(output, cache, destination, inventory, max_bytes)
    save(inventory_path, inventory)
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("snapshot", "restore"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-cache", type=Path, required=True)
    parser.add_argument("--copy", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, required=True)
    args = parser.parse_args()
    if args.action == "snapshot":
        snapshot(
            args.output, args.pool_cache, args.copy, args.inventory, args.max_bytes
        )
    else:
        copy(
            args.output,
            args.pool_cache,
            args.copy,
            load(args.inventory),
            args.max_bytes,
        )


if __name__ == "__main__":
    main()
