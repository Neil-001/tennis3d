"""Stream one rally/view shard at a time and publish checksummed split manifests."""

import hashlib
import importlib.metadata
import time
from pathlib import Path

import mujoco
import numpy as np

from tennis3d.artifacts import (
    ArtifactProvenance,
    file_sha256,
    json_sha256,
    write_json,
    write_npz,
)
from tennis3d.config import record_config
from tennis3d.provenance import code_revision
from tennis3d.schemas import write_result
from tennis3d.synthetic import GENERATOR_VERSION, KINDS
from tennis3d.synthetic.cameras import observe
from tennis3d.synthetic.config import PhysicsConfig, SimulationConfig
from tennis3d.synthetic.export import truth_result
from tennis3d.synthetic.priors import choose
from tennis3d.synthetic.rally import EVENT_CODES, assemble
from tennis3d.synthetic.shot_pool import load_or_build
from tennis3d.synthetic.simulate import Simulator


def generator_hash() -> str:
    package = Path(__file__).resolve().parents[1]
    files = [
        *package.joinpath("synthetic").glob("*.py"),
        *package.joinpath("geometry").glob("*.py"),
    ]
    files.extend(
        package / name
        for name in (
            "__init__.py",
            "artifacts.py",
            "config.py",
            "schemas.py",
            "provenance.py",
            "court/model.py",
        )
    )
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("mujoco", "numpy", "scipy", "pydantic", "PyYAML")
    }


def excerpt(
    arrays: dict[str, np.ndarray], start: int, stop: int
) -> tuple[dict[str, np.ndarray], float]:
    """Keep original physical time offset, exact timestamps and in-range events."""
    first, last = arrays["t_s"][start], arrays["t_s"][stop - 1]
    events = (arrays["event_t_s"] >= first) & (arrays["event_t_s"] <= last)
    frame_keys = {
        "t_s",
        "xyz_m",
        "velocity_m_s",
        "spin_rad_s",
        "label_valid",
        "segment_id",
    }
    output = {
        key: value[start:stop].copy()
        if key in frame_keys
        else (value[events].copy() if key.startswith("event_") else value.copy())
        for key, value in arrays.items()
    }
    output["t_s"] -= first
    output["event_t_s"] -= first
    return output, float(first)


def family_seed_for(seed: int, split_index: int, family_index: int) -> int:
    """A family identity is independent of family counts and generation order.

    The first family in each split retains the legacy one-family smoke seed.
    """
    return int(
        np.random.SeedSequence(
            seed, spawn_key=(split_index + 3 * family_index,)
        ).generate_state(1)[0]
    )


def family_physics(cfg: SimulationConfig, seed: int) -> PhysicsConfig:
    rng = np.random.default_rng(seed)
    physical = cfg.physics.model_dump()
    for parameter in ("friction", "damping"):
        physical[parameter] *= float(
            rng.uniform(1 - cfg.physics_variation, 1 + cfg.physics_variation)
        )
    if cfg.family_priors is not None:
        component = choose(rng, cfg.family_priors.components)
        bounds = component.model_dump(exclude={"name", "weight"}, exclude_none=True)
        for parameter, (low, high) in bounds.items():
            physical[parameter] = float(rng.uniform(low, high))
    return PhysicsConfig.model_validate(physical)


def pool_settings(
    cfg: SimulationConfig,
    physical: PhysicsConfig,
    seed: int,
    split: str,
    runtimes: dict[str, str],
) -> dict:
    """Only pool-affecting configuration, including the complete proposal recipe."""
    return {
        **cfg.model_dump(include={"shot_priors", "family_priors", "physics_variation"}),
        "physics": physical.model_dump(),
        "size": cfg.pool_size,
        "attempts": cfg.pool_attempts,
        "seed": seed,
        "split": split,
        "generator": GENERATOR_VERSION,
        "mujoco": mujoco.__version__,
        "runtime_versions": runtimes,
    }


def generate(
    output: Path,
    cfg: SimulationConfig,
    pool_cache: Path | None = None,
    *,
    diagnostics: dict | None = None,
) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output is not empty: {output}. Select a new run directory."
        )
    output.mkdir(parents=True, exist_ok=True)
    config_hash = record_config(output / "config.resolved.json", cfg)
    source_hash, revision = generator_hash(), code_revision()
    runtimes = runtime_versions()
    started = time.perf_counter()
    started_cpu = time.process_time()
    pool_cache = pool_cache or output / "pools"
    split_records, pool_records = {}, []
    for split_index, (split, count) in enumerate(
        [
            ("train", cfg.train_rallies),
            ("val", cfg.val_rallies),
            ("test", cfg.test_rallies),
        ]
    ):
        records = []
        family_count = cfg.family_count(split)
        for family_index in range(family_count):
            family_seed = family_seed_for(cfg.seed, split_index, family_index)
            family_id = f"{split}-family-{family_index:03d}-{family_seed}"
            physical = family_physics(cfg, family_seed)
            sim = Simulator(physical)
            pool_config = pool_settings(cfg, physical, family_seed, split, runtimes)
            pool_hash = json_sha256(pool_config)
            provenance = ArtifactProvenance(
                source_sha256=source_hash,
                config_sha256=pool_hash,
                code_revision=GENERATOR_VERSION,
            )
            pool_path = pool_cache / f"{split}-{source_hash[:12]}-{pool_hash}.npz"
            pool = load_or_build(pool_path, provenance, sim, cfg, family_seed)
            pool_sha = file_sha256(pool_path)
            pool_records.append(
                {
                    "family_id": family_id,
                    "split": split,
                    "sha256": pool_sha,
                    "config": pool_config,
                    "attempts": pool.attempts,
                }
            )
            for rally_index in range(family_index, count, family_count):
                rally_seed = int(
                    np.random.SeedSequence(
                        [cfg.seed, split_index, rally_index, 1]
                    ).generate_state(1)[0]
                )
                rally_id = f"{split}-rally-{rally_index:05d}"
                rally = assemble(sim, pool, cfg, rally_seed)
                base_provenance = ArtifactProvenance(
                    source_sha256=source_hash,
                    config_sha256=config_hash,
                    upstream_sha256={"shot_pool": pool_sha},
                    code_revision=revision,
                )
                physical_path = Path("rallies") / f"{rally_id}.npz"
                write_npz(output / physical_path, rally.arrays, base_provenance)
                physical_hash = file_sha256(output / physical_path)
                for view in range(cfg.views_per_rally):
                    view_seed = int(
                        np.random.SeedSequence([rally_seed, view, 2]).generate_state(1)[
                            0
                        ]
                    )
                    arrays, offset = rally.arrays, 0.0
                    if view:
                        length = len(arrays["t_s"])
                        arrays, offset = excerpt(arrays, length // 5, length * 4 // 5)
                    observed = observe(arrays, cfg, view_seed)
                    shard = Path("shards") / f"{rally_id}-view-{view:02d}.npz"
                    provenance = ArtifactProvenance(
                        source_sha256=source_hash,
                        config_sha256=config_hash,
                        upstream_sha256={"rally": physical_hash, "shot_pool": pool_sha},
                        code_revision=revision,
                    )
                    write_npz(output / shard, observed, provenance)
                    shard_hash = file_sha256(output / shard)
                    truth_path = Path("truth") / f"{rally_id}-view-{view:02d}.json"
                    write_result(
                        output / truth_path,
                        truth_result(
                            observed,
                            name=shard.stem,
                            source_hash=shard_hash,
                            config_hash=config_hash,
                            revision=revision,
                        ),
                    )
                    records.append(
                        {
                            "rally_id": rally_id,
                            "family_id": family_id,
                            "split": split,
                            "view_id": view,
                            "seed": rally_seed,
                            "view_seed": view_seed,
                            "fps": rally.fps,
                            "attempts": rally.attempts,
                            "physical_time_origin_s": offset,
                            "frames": len(arrays["t_s"]),
                            "path": shard.as_posix(),
                            "sha256": shard_hash,
                            "rally_path": physical_path.as_posix(),
                            "rally_sha256": physical_hash,
                            "pool_sha256": pool_sha,
                            "truth_path": truth_path.as_posix(),
                            "truth_sha256": file_sha256(output / truth_path),
                        }
                    )
            if diagnostics is not None:
                diagnostics[family_id] = dict(pool.diagnostics)
        manifest = {
            "schema_version": "1.0",
            "generator_version": GENERATOR_VERSION,
            "generator_sha256": source_hash,
            "mujoco_version": mujoco.__version__,
            "runtime_versions": runtimes,
            "config_sha256": config_hash,
            "seed": cfg.seed,
            "split": split,
            "records": records,
        }
        path = output / f"{split}.manifest.json"
        write_json(path, manifest)
        split_records[split] = {
            "path": path.name,
            "sha256": file_sha256(path),
            "rallies": count,
            "views": len(records),
        }
    elapsed = time.perf_counter() - started
    cpu_elapsed = time.process_time() - started_cpu
    manifest = {
        "schema_version": "1.0",
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": source_hash,
        "code_revision": revision,
        "mujoco_version": mujoco.__version__,
        "runtime_versions": runtimes,
        "config_sha256": config_hash,
        "seed": cfg.seed,
        "parameters": cfg.model_dump(),
        "pools": pool_records,
        "splits": split_records,
        "event_codes": EVENT_CODES,
        "segment_kinds": list(KINDS),
        "generation_seconds": elapsed,
        "generation_cpu_seconds": cpu_elapsed,
        "rallies_per_wall_hour": 3600
        * (cfg.train_rallies + cfg.val_rallies + cfg.test_rallies)
        / elapsed,
        "rallies_per_cpu_hour": 3600
        * (cfg.train_rallies + cfg.val_rallies + cfg.test_rallies)
        / cpu_elapsed,
    }
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "manifest.sha256.json",
        {"manifest.json": file_sha256(output / "manifest.json")},
    )
    return manifest
