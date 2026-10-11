"""Model-free original-byte transport; no numerical generation or cloud calls."""

from pathlib import Path

import pytest

from scripts import azure_pool_recovery as recovery
from tennis3d.artifacts import file_sha256
from tennis3d.synthetic.config import SimulationConfig
from tennis3d.synthetic.producer_records import plan_families
from tennis3d.synthetic.producer_recovery import create_bundle, restore_bundle
from tennis3d.synthetic.producer_state import load, save


@pytest.fixture
def original_pools(tmp_path):
    output, cache = tmp_path / "output", tmp_path / "cache"
    cache.mkdir()
    cfg = SimulationConfig(
        seed=20260927,
        train_rallies=360,
        val_rallies=45,
        test_rallies=45,
        train_families=8,
        val_families=1,
        test_families=1,
        pool_size=96,
        views_per_rally=2,
    )
    plan = {
        "recipe": cfg.model_dump(),
        "pool_source_sha256": "a" * 64,
        "runtime_versions": {"numpy": "fixture"},
    }
    save(output / "control/plan.json", plan)
    families = list(plan_families(cfg, "a" * 64, plan["runtime_versions"]))
    for index, family in enumerate(families[:2]):
        (cache / family.cache_name).write_bytes(
            b"original completed NPZ fixture" + bytes([index])
        )
    family = families[0]
    save(
        output / "control/pools/train-000000.json",
        {
            "pool": {
                "family_id": family.family_id,
                "config": family.pool_config,
                "sha256": file_sha256(cache / family.cache_name),
            },
            "diagnostics": {"pool_built": True},
        },
    )
    return output, cache, families


def test_preserves_checkpointed_and_completed_not_yet_checkpointed_bytes(
    original_pools, tmp_path
):
    output, cache, families = original_pools
    (cache / Path(families[0].cache_name).with_suffix(".lock")).write_bytes(b"")
    (cache / ("." + families[1].cache_name + ".writer-temp")).write_bytes(b"unfinished")
    copied, inventory_path = tmp_path / "copied", tmp_path / "inventory.json"
    inventory = recovery.snapshot(output, cache, copied, inventory_path, 1 << 20)
    assert set(inventory["files"]) == {f.cache_name for f in families[:2]}
    assert inventory["files"][families[0].cache_name]["checkpointed"]
    assert not inventory["files"][families[1].cache_name]["checkpointed"]
    assert sorted(p.name for p in copied.iterdir()) == sorted(inventory["files"])
    fresh = tmp_path / "fresh"
    recovery.copy(output, copied, fresh, load(inventory_path), 1 << 20)
    for name in inventory["files"]:
        assert (fresh / name).read_bytes() == (cache / name).read_bytes()


@pytest.mark.parametrize(
    "kind",
    ["symlink", "unplanned", "corrupt", "missing", "wrong_family", "wrong_config"],
)
def test_refuses_unrecoverable_or_unplanned_pool_content(original_pools, kind):
    output, cache, families = original_pools
    file = cache / families[0].cache_name
    path = output / "control/pools/train-000000.json"
    if kind == "symlink":
        file.unlink()
        file.symlink_to(cache / families[1].cache_name)
    elif kind == "unplanned":
        (cache / "token").write_bytes(b"must never be transported")
    elif kind == "corrupt":
        file.write_bytes(b"changed original")
    elif kind == "missing":
        file.unlink()
    else:
        value = load(path)
        if kind == "wrong_family":
            value["pool"]["family_id"] = "different"
        else:
            value["pool"]["config"] = {}
        save(path, value, overwrite=True)
    with pytest.raises((ValueError, FileNotFoundError)):
        recovery.select(output, cache)


def test_transfer_hash_budget_and_fresh_destination_are_enforced(
    original_pools, tmp_path
):
    output, cache, families = original_pools
    inventory = recovery.select(output, cache)
    with pytest.raises(ValueError, match="allowance"):
        recovery.copy(output, cache, tmp_path / "too_small", inventory, 1)
    assert not (tmp_path / "too_small").exists()
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        recovery.copy(output, cache, existing, inventory, 1 << 20)
    fresh = tmp_path / "fresh"
    recovery.copy(output, cache, fresh, inventory, 1 << 20)
    (fresh / families[1].cache_name).write_bytes(b"transit corruption")
    with pytest.raises(ValueError, match="size/hash"):
        recovery.verify(output, fresh, inventory)


def test_restore_cannot_omit_a_checkpoint_or_change_plan(original_pools, tmp_path):
    output, cache, _ = original_pools
    inventory = recovery.select(output, cache)
    checkpoint = output / "control/pools/train-000000.json"
    checkpoint.unlink()
    with pytest.raises(ValueError, match="lacks"):
        recovery.verify(output, cache, inventory)
    plan = load(output / "control/plan.json")
    plan["runtime_versions"]["numpy"] = "different"
    save(output / "control/plan.json", plan, overwrite=True)
    with pytest.raises(ValueError, match="plan"):
        recovery.verify(output, cache, inventory)


def test_newer_bundle_rejects_older_same_plan_inventory(original_pools, tmp_path):
    output, cache, families = original_pools
    second = cache / families[1].cache_name
    original_second = second.read_bytes()
    second.unlink()
    preserved = tmp_path / "older-pools"
    older = recovery.snapshot(
        output, cache, preserved, tmp_path / "older.json", 1 << 20
    )
    second.write_bytes(original_second)
    save(
        output / "control/pools/train-000001.json",
        {
            "pool": {
                "family_id": families[1].family_id,
                "config": families[1].pool_config,
                "sha256": file_sha256(second),
            }
        },
    )
    bundle = tmp_path / "newer.zip"
    create_bundle(
        output,
        [
            "control/plan.json",
            "control/pools/train-000000.json",
            "control/pools/train-000001.json",
        ],
        bundle,
        1 << 20,
    )
    restored = tmp_path / "restored"
    restore_bundle(bundle, restored, 1 << 20)
    destination = tmp_path / "rejected-cache"
    with pytest.raises(ValueError, match="omits a checkpointed"):
        recovery.copy(restored, preserved, destination, older, 1 << 20)
    assert not destination.exists()
    current = recovery.select(output, cache)
    assert current["plan_sha256"] == older["plan_sha256"]
    recovery.copy(restored, cache, tmp_path / "matching-cache", current, 1 << 20)
    # Post-resume original-byte verification allows newly completed pools.
    recovery.verify(restored, cache, older, all_checkpointed=True)
