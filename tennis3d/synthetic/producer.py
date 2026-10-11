"""Deterministic bounded main-dataset coordinator. Legacy NPZ stays separate."""

import gc
import hashlib
import importlib.metadata
import itertools
import platform
import resource
import sys
import time
from pathlib import Path

from tennis3d.artifacts import atomic_file, json_sha256, read_json
from tennis3d.datasets.io import ShardWriter, write_manifest
from tennis3d.datasets.schema import (
    Artifact,
    Counts,
    Manifest,
    Pool,
    Provenance,
    Shard,
    count_records,
)
from tennis3d.datasets.subset import SelectionPolicy
from tennis3d.provenance import code_revision
from tennis3d.synthetic import GENERATOR_VERSION
from tennis3d.synthetic.producer_catalog import DEFINITIONS, definition_hash, summarize
from tennis3d.synthetic.producer_config import ProducerConfig
from tennis3d.synthetic.producer_finalize import finalization, recover_extraction
from tennis3d.synthetic.producer_parallel import OrderedWorkers, StopController, Stopped
from tennis3d.synthetic.producer_phase import generation_boundary, validate_generation
from tennis3d.synthetic.producer_publication import Publication
from tennis3d.synthetic.producer_records import (
    IDENTITY_VERSION,
    audit_plan_seeds,
    plan_families,
    plan_rallies,
    prepare_family,
    produce_record,
)
from tennis3d.synthetic.producer_state import (
    artifact,
    coordinator_lock,
    fsync_dir,
    load,
    save,
    save_once,
    usage,
    verify,
)
from tennis3d.synthetic.producer_subset_publication import publish_subset

PRODUCER_VERSION = "main-producer-v2"


class PoolPreparationError(ValueError):
    def __init__(self, pool_key, detail):
        self.producer_pool_key = pool_key
        super().__init__(pool_key, detail)

    def __str__(self):
        return f"pool preparation failed for {self.producer_pool_key}: {self.args[1]}"


def source_identity():
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def pool_source_identity():
    """Pool caches bind physics/prior code, independently of transport changes."""
    package = Path(__file__).resolve().parents[1]
    paths = list(package.joinpath("geometry").glob("*.py"))
    paths += [
        package / "synthetic" / name
        for name in (
            "__init__.py",
            "config.py",
            "priors.py",
            "rally.py",
            "shots.py",
            "simulate.py",
            "shot_pool.py",
            "generate.py",
            "producer_records.py",
        )
    ]
    paths += [
        package / name
        for name in (
            "__init__.py",
            "artifacts.py",
            "schemas.py",
            "config.py",
            "court/model.py",
        )
    ]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_identity():
    return {
        **{
            name: importlib.metadata.version(name)
            for name in (
                "mujoco",
                "numpy",
                "scipy",
                "pydantic",
                "PyYAML",
                "pyarrow",
                "huggingface-hub",
            )
        },
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.system(),
        "byteorder": sys.byteorder,
    }


def _timed(function, *args, **kwargs):
    started, cpu = time.monotonic(), time.process_time()
    result, diagnostic = function(*args, **kwargs)
    diagnostic["performance"] = {
        "wall_seconds": time.monotonic() - started,
        "cpu_seconds": time.process_time() - cpu,
    }
    return result, diagnostic


def _prepare(task):
    *family, cache_size = task
    try:
        return _timed(prepare_family, *family, cache_size=cache_size)
    except Exception as error:  # noqa: BLE001 - preserve any failed pool identity across IPC
        raise PoolPreparationError(
            f"pool:{task[1]}:{task[2]}", str(error)[:400]
        ) from None


def _produce(task):
    args, kwargs = task
    return _timed(produce_record, *args, **kwargs)


def _batches(values, size):
    iterator = iter(values)
    while batch := list(itertools.islice(iterator, size)):
        yield batch


def _expected(cfg, shards):
    actual = count_records(shards)
    # Frame counts are a deterministic generation result, not a sampling target.
    # Rally/view counts always remain the original plan, including on failure.
    return {
        split: Counts(
            rallies=getattr(cfg, f"{split}_rallies"),
            views=getattr(cfg, f"{split}_rallies") * cfg.views_per_rally,
            frames=actual[split].frames,
        )
        for split in ("train", "val", "test")
    }


def _recover_active(files, control):
    active = control / "active.json"
    if not active.exists():
        return
    transaction = load(active)
    committed = control / "commits" / transaction["commit"]
    if not committed.exists():
        for relative in transaction["paths"]:
            Artifact.relative_path(relative)
            path = files / relative
            if path.is_symlink():
                raise ValueError("unexpected symlink in interrupted transaction")
            path.unlink(missing_ok=True)
            # atomic_file's name contains the exact owned destination name.
            for partial in path.parent.glob(f".{path.name}.*"):
                if partial.is_symlink() or not partial.is_file():
                    raise ValueError("unexpected interrupted writer entry")
                partial.unlink()
    active.unlink()
    fsync_dir(control)


def _commits(control):
    result = []
    for index, path in enumerate(sorted((control / "commits").glob("*.json"))):
        if path.name != f"{index:06d}.json":
            raise ValueError("noncontiguous producer commit ledger")
        commit = load(path)
        if not isinstance(commit, dict) or set(commit) != {
            "shard",
            "catalog",
            "admission",
        }:
            raise ValueError("invalid producer commit ledger")
        result.append(commit)
    return result


def _performance(started, cpu_start, output, execution, status):
    own = resource.getrusage(resource.RUSAGE_SELF)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "status": status,
        "wall_seconds": time.monotonic() - started,
        "cpu_seconds_including_children": own.ru_utime
        + own.ru_stime
        + child.ru_utime
        + child.ru_stime
        - cpu_start,
        "output_bytes": usage(output),
        "execution": execution.model_dump(),
        "note": "local runtime; aggregate instantaneous RSS and disk peaks require the qualification sampler",
    }


def generate_main(
    output,
    cfg,
    execution=None,
    *,
    pool_cache=None,
    resume=False,
    phase="complete",
    stop=None,
    backend=None,
    fault=None,
):
    """Run the requested phase or raise with durable recoverable planned output.

    fault is an internal crash-boundary test hook. It is never part of identity.
    """
    if phase not in {"complete", "generate", "finalize"}:
        raise ValueError("unknown producer phase")
    if phase == "finalize" and not resume:
        raise ValueError("finalization requires --resume and complete generation state")
    output = Path(output).absolute()
    execution = execution or ProducerConfig()
    execution.recipe_reservation(cfg)
    started = time.monotonic()
    own, child = (
        resource.getrusage(resource.RUSAGE_SELF),
        resource.getrusage(resource.RUSAGE_CHILDREN),
    )
    cpu_start = own.ru_utime + own.ru_stime + child.ru_utime + child.ru_stime
    stop = stop or StopController(execution.drain_seconds)
    fault = fault or (lambda phase: None)
    pool_cache = Path(pool_cache).absolute() if pool_cache else output / "pools"
    if pool_cache.is_symlink():
        raise ValueError("pool cache cannot be a symlink")
    with coordinator_lock(output), stop:
        control = output / "control"
        plan_path = control / "plan.json"
        if phase == "finalize" and not (control / "generation-complete.json").is_file():
            raise ValueError("finalization requires complete generation state")
        if plan_path.exists() and not resume:
            raise FileExistsError("output already has a plan; use compatible --resume")
        if not plan_path.exists() and any(
            p.name != ".producer.lock" for p in output.iterdir()
        ):
            raise FileExistsError("new output must be empty")
        source, runtimes = source_identity(), runtime_identity()
        pool_source = pool_source_identity()
        policy = None
        if execution.subset_policy:
            policy = SelectionPolicy.model_validate(read_json(execution.subset_policy))
        plan = {
            "producer": PRODUCER_VERSION,
            "identity": IDENTITY_VERSION,
            "source_sha256": source,
            "pool_source_sha256": pool_source,
            "runtime_versions": runtimes,
            "recipe": cfg.model_dump(),
            "layout": execution.layout(),
            "publication": {
                "repo_id": execution.repo_id,
                "revision": execution.revision,
            },
            "coverage": {
                "definitions": DEFINITIONS,
                "source_sha256": definition_hash(),
            },
            "subset_policy": policy.model_dump() if policy is not None else None,
        }
        if plan_path.exists():
            saved_plan = load(plan_path)
            if {k: v for k, v in saved_plan.items() if k != "seed_audit"} != plan:
                raise ValueError(
                    "resume requires matching source/runtime/recipe/identity/order/layout/destination"
                )
        else:
            seed_audit = audit_plan_seeds(cfg, max_proposals=execution.max_proposals)
            plan["seed_audit"] = seed_audit
            save(plan_path, plan)
        plan = load(plan_path)
        for directory in (control / "commits", control / "pools", pool_cache):
            directory.mkdir(parents=True, exist_ok=True)
        if phase == "generate" and (control / "finalized.json").exists():
            raise ValueError("generation-only cannot resume an already finalized run")
        publication = Publication(output, execution, backend)
        files, prefix = publication.staging.files, execution.prefix
        _recover_active(files, control)
        recover_extraction(output, control)
        commits = _commits(control)
        shards = [Shard.model_validate(c.pop("shard")) for c in commits]
        for commit, shard in zip(commits, shards):
            stop.check()
            path = files / shard.path
            retained = path.exists() or path.is_symlink()
            if retained:
                verify(files, shard)
                if phase != "finalize":
                    stop.check()
                    publication.publish(shard, evict=execution.evict)
            if not retained or (phase == "finalize" and publication.publisher):
                # Absent bytes reuse their exact first-readback receipt, undownloaded.
                publication.receipt(shard)
            for key in ("catalog", "admission"):
                verify(files, Artifact.model_validate(commit[key]))
        pools = []
        timings = {
            "pools": [],
            "records": {"count": 0, "wall_seconds": 0.0, "cpu_seconds": 0.0},
            "audit_seconds": None,
        }
        status = "incomplete"
        writer = None
        failures_path = control / "failures.json"
        failures = load(failures_path) if failures_path.exists() else {}
        try:
            if phase != "finalize":
                with OrderedWorkers(
                    execution.workers, execution.inflight, stop
                ) as workers:
                    # Durable descriptor per pool, so interrupted cold preparation reuses
                    # the exact hash-checked cache rather than rebuilding other pools.
                    families = list(plan_families(cfg, pool_source, runtimes))
                    missing, existing = [], {}

                    def pool_checkpoint(family):
                        name = f"{family.split}-{family.family_index:06d}.json"
                        return control / "pools" / name

                    for family in families:
                        checkpoint = pool_checkpoint(family)
                        if checkpoint.exists():
                            existing[(family.split, family.family_index)] = load(
                                checkpoint
                            )
                        if not (control / "finalized.json").exists() and (
                            not checkpoint.exists()
                            or not (pool_cache / family.cache_name).exists()
                        ):
                            missing.append(family)
                    tasks = (
                        (
                            cfg,
                            f.split,
                            f.family_index,
                            pool_cache,
                            pool_source,
                            runtimes,
                            execution.worker_cache_families,
                        )
                        for f in missing
                    )
                    if any(
                        failures.get(f"pool:{f.split}:{f.family_index}", 0)
                        >= execution.max_failures
                        for f in missing
                    ):
                        raise ValueError(
                            "persisted pool preparation failure budget exhausted"
                        )
                    for family, result in zip(missing, workers.map(_prepare, tasks)):
                        pool, diagnostics = result
                        timing = diagnostics.pop("performance", None)
                        timings["pools"].append(
                            {
                                "family_id": pool.family_id,
                                "cold": diagnostics.get("pool_built", False),
                                **(timing or {}),
                            }
                        )
                        value = {"pool": pool.model_dump(), "diagnostics": diagnostics}
                        previous = existing.get((family.split, family.family_index))
                        if previous is not None:
                            if previous["pool"] != value["pool"]:
                                raise ValueError(
                                    "rebuilt pool differs from its durable descriptor"
                                )
                        else:
                            save(pool_checkpoint(family), value)
                            existing[(family.split, family.family_index)] = value
                        if (
                            usage(output) + execution.shard_reservation()
                            > execution.staging_bytes
                        ):
                            raise ValueError(
                                "pool preparation exhausted staging reservations"
                            )
                    pools = [
                        Pool.model_validate(existing[(f.split, f.family_index)]["pool"])
                        for f in families
                    ]
                    # Consumers validate this exact frozen SimulationConfig. Producer
                    # and seed audit metadata are separately bound in producer.json.
                    resolved_config = cfg.model_dump()
                    provenance = Provenance(
                        generator_version=f"{GENERATOR_VERSION}/{PRODUCER_VERSION}/{IDENTITY_VERSION}",
                        generator_sha256=source,
                        code_revision=code_revision(),
                        runtime_versions=runtimes,
                        config=resolved_config,
                        config_sha256=json_sha256(resolved_config),
                        seed=cfg.seed,
                        pools=pools,
                    )
                    saved_provenance = control / "provenance.json"
                    if saved_provenance.exists():
                        prior = Provenance.model_validate(load(saved_provenance))
                        # Git revision can change without a source change, e.g. ordinary
                        # integration merge. Persist the original source provenance.
                        provenance = provenance.model_copy(
                            update={"code_revision": prior.code_revision}
                        )
                        del prior
                    save_once(saved_provenance, provenance.model_dump())
                    pool_lookup = {
                        (p.split, f.family_index): p for p, f in zip(pools, families)
                    }
                    planned = plan_rallies(cfg, cohort_size=execution.family_cohort)
                    for shard_number, batch in enumerate(
                        _batches(planned, execution.shard_rallies)
                    ):
                        if shard_number < len(commits):
                            if [r.rally_id for r in shards[shard_number].records] != [
                                p.rally_id for p in batch
                            ]:
                                raise ValueError(
                                    "committed shard order differs from the plan"
                                )
                            continue
                        stop.check()
                        if (
                            failures.get(f"transaction:{shard_number}", 0)
                            >= execution.max_failures
                        ):
                            raise ValueError(
                                "persisted shard transaction failure budget exhausted"
                            )
                        if (
                            usage(output) + execution.shard_reservation()
                            > execution.staging_bytes
                        ):
                            raise ValueError(
                                "staging reservation unavailable; publish or increase the disk limit"
                            )
                        shard_path = f"{prefix}/shards/{shard_number:06d}.parquet"
                        catalog_path = f"{prefix}/catalog/{shard_number:06d}.json"
                        admission_path = f"{prefix}/admission/{shard_number:06d}.json"
                        save(
                            control / "active.json",
                            {
                                "commit": f"{shard_number:06d}.json",
                                "paths": [shard_path, catalog_path, admission_path],
                            },
                            overwrite=True,
                        )
                        catalog, admissions = [], []

                        def tasks(planned_batch=batch):
                            for rally in planned_batch:
                                if (
                                    failures.get(rally.rally_id, 0)
                                    >= execution.max_failures
                                ):
                                    raise ValueError(
                                        f"persisted failure budget exhausted for {rally.rally_id}"
                                    )
                                yield (
                                    (
                                        cfg,
                                        rally.split,
                                        rally.rally_index,
                                        pool_lookup[(rally.split, rally.family_index)],
                                        pool_cache,
                                        pool_source,
                                        runtimes,
                                    ),
                                    {
                                        "cache_size": execution.worker_cache_families,
                                        "max_proposals": execution.max_proposals,
                                        "max_record_bytes": execution.max_record_bytes,
                                    },
                                )

                        current = None
                        try:
                            with ShardWriter(
                                files / shard_path,
                                provenance,
                                relative_path=shard_path,
                                row_group_rallies=execution.row_group_rallies,
                            ) as writer:
                                for current, (record, diagnostics) in zip(
                                    batch, workers.map(_produce, tasks())
                                ):
                                    nbytes = sum(
                                        a.nbytes for a in record.physical.values()
                                    ) + sum(
                                        a.nbytes
                                        for v in record.views
                                        for a in v.arrays.values()
                                    )
                                    if nbytes > execution.max_record_bytes:
                                        raise ValueError(
                                            f"record {current.rally_id} exceeds its byte admission"
                                        )
                                    timing = diagnostics.pop("performance", {})
                                    timings["records"]["count"] += 1
                                    for key in ("wall_seconds", "cpu_seconds"):
                                        timings["records"][key] += timing.get(key, 0.0)
                                    writer.append(record)
                                    catalog.append(summarize(record))
                                    admissions.append(diagnostics)
                                shard = writer.close()
                            fsync_dir((files / shard_path).parent)
                            fault("shard")
                            save(files / catalog_path, {"records": catalog})
                            save(files / admission_path, {"records": admissions})
                            fault("catalog")
                            commit = {
                                "shard": shard.model_dump(),
                                "catalog": artifact(files, catalog_path).model_dump(),
                                "admission": artifact(
                                    files, admission_path
                                ).model_dump(),
                            }
                            save(
                                control / "commits" / f"{shard_number:06d}.json", commit
                            )
                            fault("ledger")
                            shards.append(shard)
                            commit.pop("shard")
                            commits.append(commit)
                            (control / "active.json").unlink()
                            fsync_dir(control)
                            stop.check()
                            publication.publish(shard, evict=execution.evict)
                            fault("receipt")
                        except Stopped:
                            raise
                        except Exception as error:
                            # A failed future has not yielded its plan yet. The first
                            # uncommitted row is retained as incomplete, never replaced.
                            failed = getattr(
                                error,
                                "rally_id",
                                batch[len(catalog)].rally_id
                                if len(catalog) < len(batch)
                                else f"transaction:{shard_number}",
                            )
                            failure_diagnostics = getattr(error, "diagnostics", None)
                            if failure_diagnostics is not None:
                                save(
                                    control / "last-failure.json",
                                    failure_diagnostics,
                                    overwrite=True,
                                )
                            failures[failed] = failures.get(failed, 0) + 1
                            save(failures_path, failures, overwrite=True)
                            raise
                # No process-owned simulation/pool/IPC memory survives finalization.
                pool_lookup = {}
                del families, missing, existing, tasks
                gc.collect()
            else:
                provenance = Provenance.model_validate(
                    load(control / "provenance.json")
                )
                pools = provenance.pools
            manifest = Manifest(
                provenance=provenance, shards=shards, expected=_expected(cfg, shards)
            )
            boundary = generation_boundary(control, plan, manifest)
            validate_generation(
                control,
                boundary,
                manifest,
                cfg,
                execution,
                pool_cache if phase == "finalize" else None,
            )
            if phase != "finalize":
                stop.check()
                save_once(control / "generation-complete.json", boundary)
            write_manifest(
                control / "incomplete.manifest.json", manifest, overwrite=True
            )
            if phase == "generate":
                stop.check()
                status = "generation-complete"
                return {
                    "state": status,
                    "counts": {
                        split: counts.model_dump()
                        for split, counts in manifest.expected.items()
                    },
                }
            if failures.get("finalization", 0) >= execution.max_failures:
                raise ValueError("persisted finalization failure budget exhausted")
            try:
                complete, report, finalized, audit_seconds = finalization(
                    output,
                    files,
                    control,
                    manifest,
                    commits,
                    plan,
                    execution,
                    publication,
                    stop,
                    policy,
                    fault,
                )
            except Stopped:
                raise
            except Exception:
                failures["finalization"] = failures.get("finalization", 0) + 1
                save(failures_path, failures, overwrite=True)
                raise
            stop.check()
            # Keep one full set of record indexes during metadata publication.
            # The prospective complete model was independently validated.
            manifest = complete
            shards = complete.shards
            gc.collect()
            index_path = f"{prefix}/catalog/index.json"
            save_once(
                files / index_path,
                {
                    "definitions": plan["coverage"],
                    "parts": [c["catalog"] for c in commits],
                    "source_manifest_sha256": json_sha256(complete.model_dump()),
                },
            )
            timings["audit_seconds"] = audit_seconds
            manifest_path = finalized["manifest"]["path"]
            audit_path = finalized["audit_path"]
            summary_path = finalized["summary_path"]
            expected = [
                Artifact(path=s.path, sha256=s.sha256, size_bytes=s.size_bytes)
                for s in shards
            ]
            expected.extend(
                Artifact.model_validate(c[key])
                for c in commits
                for key in ("catalog", "admission")
            )
            expected.extend(
                [
                    artifact(files, index_path),
                    artifact(files, finalized["catalog_path"]),
                    artifact(files, audit_path),
                    artifact(files, summary_path),
                ]
            )
            card_path = f"{prefix}/README.md"
            license_path = f"{prefix}/LICENSE"
            for relative, text in (
                (
                    card_path,
                    f"# Synthetic tennis {execution.purpose}\n\nPurpose: {execution.purpose}. Numeric MuJoCo simulation only.\nAll {sum(v.rallies for v in complete.expected.values())} planned physical rallies and their declared views passed a complete numeric audit.\nThis {'qualification' if execution.purpose == 'qualification' else 'candidate'} data does not freeze production counts, recipe or search quotas.\nProducer identities, admission conditioning and exact artifacts are bound by producer.json.\nDataset license: CC BY 4.0. Attribution: N-il/tennis-3d-synthetic.\n",
                ),
                (
                    license_path,
                    "Creative Commons Attribution 4.0 International (CC BY 4.0).\nhttps://creativecommons.org/licenses/by/4.0/\nAttribution: N-il/tennis-3d-synthetic, synthetic numeric MuJoCo data.\n",
                ),
            ):
                path = files / relative
                if path.exists():
                    if path.read_text() != text:
                        raise ValueError("release card/license differs from the plan")
                else:
                    with atomic_file(path) as stream:
                        stream.write(text.encode())
                    fsync_dir(path.parent)
                expected.append(artifact(files, relative))
            producer_path = f"{prefix}/producer.json"
            save_once(
                files / producer_path,
                {
                    "plan": plan,
                    "manifest_sha256": json_sha256(complete.model_dump()),
                    "artifacts": [s.model_dump() for s in expected],
                },
            )
            producer_spec = artifact(files, producer_path)
            expected.append(producer_spec)
            main_checkpoint = control / "main-publication.json"
            if main_checkpoint.exists():
                saved = load(main_checkpoint)
                if saved["manifest"] != artifact(files, manifest_path).model_dump():
                    raise ValueError("main publication checkpoint differs from release")
                commit = saved["commit"]
                if publication.publisher:
                    receipt = publication.receipt(artifact(files, manifest_path))
                    if receipt["commit"] != commit:
                        raise ValueError("main immutable commit differs from receipt")
            else:
                for spec in expected:
                    stop.check()
                    publication.publish(
                        spec, evict=execution.evict and spec.path.endswith(".parquet")
                    )
                stop.check()
                commit = publication.complete(
                    artifact(files, manifest_path),
                    expected,
                    artifact(files, audit_path),
                    manifest=complete,
                    audit_report=report,
                    producer_spec=producer_spec,
                )
                save(
                    main_checkpoint,
                    {
                        "manifest": artifact(files, manifest_path).model_dump(),
                        "commit": commit,
                    },
                )
                fault("main-publication")
            stop.check()
            subset_result = None
            if policy is not None:
                # Binding rereads the durable source metadata. The successful
                # audit and main commit no longer need the in-memory indexes.
                del complete, manifest, provenance
                writer = None
                shards, pools = [], []
                gc.collect()
                subset_result = publish_subset(
                    output,
                    execution,
                    commit,
                    artifact(files, manifest_path),
                    artifact(files, finalized["catalog_path"]),
                    files,
                    report,
                    backend=backend,
                    check_stop=stop.check,
                )
                fault("subset-publication")
            stop.check()
            status = "complete"
            save(
                control / "result.json",
                {
                    "state": status,
                    "manifest": artifact(files, manifest_path).model_dump(),
                    "commit": commit,
                    "subset": subset_result,
                },
                overwrite=True,
            )
            return {
                "state": status,
                "manifest": manifest_path,
                "commit": commit,
                "counts": report["counts"],
                "subset": subset_result,
            }
        except Exception as error:
            pool_key = getattr(error, "producer_pool_key", None)
            if pool_key is not None:
                failures[pool_key] = failures.get(pool_key, 0) + 1
                save(failures_path, failures, overwrite=True)
            raise
        finally:
            _recover_active(files, control)
            recover_extraction(output, control)
            save(control / "timings.json", timings, overwrite=True)
            if pools and (control / "provenance.json").exists():
                write_manifest(
                    control / "incomplete.manifest.json",
                    Manifest(
                        provenance=Provenance.model_validate(
                            load(control / "provenance.json")
                        ),
                        shards=shards,
                        expected=_expected(cfg, shards),
                    ),
                    overwrite=True,
                )
            save(
                control / "performance.json",
                _performance(started, cpu_start, output, execution, status),
                overwrite=True,
            )
