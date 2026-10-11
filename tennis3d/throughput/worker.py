"""Frozen subprocess entry point; imports exactly one captured source tree.

This file is executed by absolute oracle path. Its imports intentionally select
PYTHONPATH, never an editable environment's historic checkout.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import resource
import socket
import stat
import time
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

# Captured before importing editable candidate modules. Imports at EOF are
# intentional so function defaults retain the original notification callable.


def _timing_boundary(name=None, write=os.write):
    if os.write is not write:
        raise ValueError("candidate changed frozen timing notifications")
    if name is not None:
        write(1, f"THROUGHPUT_BOUNDARY {name}\n".encode("ascii"))


def digest(value):
    """Exact typed tensors/arrays, including NaNs and RNG tuple representations."""
    h = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()
        if isinstance(item, np.ndarray):
            h.update(str(item.dtype).encode())
            h.update(str(item.shape).encode())
            h.update(item.tobytes())
        elif isinstance(item, dict):
            for k in sorted(item, key=str):
                visit(k)
                visit(item[k])
        elif isinstance(item, (tuple, list)):
            for x in item:
                visit(x)
        else:
            h.update(repr(item).encode())
            h.update(b"\0")

    visit(value)
    return h.hexdigest()


def seed(value):
    random.seed(value)
    np.random.seed(value % 2**32)
    torch.manual_seed(value)


def recipe(root, descriptor, *, fixture=False):
    raw = yaml.safe_load(
        (root / "configs/stage4-preparation-main-v1/full-run.yaml").read_text()
    )
    raw.update(train_manifest=str(descriptor), val_manifest=str(descriptor))
    if fixture:
        raw.update(device="cpu", batch_size=2, max_frames=32, max_steps=4)
    return TrainConfig.model_validate(raw)


def update(model, ema, optimizer, cfg, batch, device, *, audit=None):
    f, target, valid = batch
    f, target, valid = f.to(device), target.to(device), valid.to(device)
    if audit is not None:
        audit(batch_digest((f, target, valid)))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = vertical_radial_position_loss(model(f), target, valid, f.padding)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite training loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), cfg.gradient_clip, error_if_nonfinite=True
    )
    optimizer.step()
    ema.update_parameters(model)
    synchronize(device)


def model_state(cfg, device):
    seed(cfg.seed)
    model = Lifter(cfg.model).to(device)
    optimizer = build_optimizer(model, cfg.learning_rate)
    for group in optimizer.param_groups:
        group["lr_schedule"] = _cosine_policy(group["lr"], cfg.model.schedule_steps)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg.ema_decay))
    ema.update_parameters(model)
    return model, ema, optimizer


def batch_digest(batch):
    f, target, valid = batch
    return digest(
        ({field.name: getattr(f, field.name) for field in fields(f)}, target, valid)
    )


def initial_state(data, cfg, workload):
    from tennis3d.lifting.data import WindowSampler

    sampler = WindowSampler(data, cfg)
    state = sampler.state_dict()
    locations = data._source.locations
    shards = np.array([locations[int(i)][0] for i in sampler.order])
    changes = np.flatnonzero(shards[1:] != shards[:-1]) + 1
    position = (
        0 if workload["boundary"] == 0 else int(changes[workload["boundary"] - 1]) - 24
    )
    state.update(cursor=position, draws=position)
    return state, shards, position


class _PolicyObserved(Exception):
    pass


def install_source_guards(
    module, raw, rules, *, observe=False, reads_started=None, artifact_reads=None
):
    """Check actual facilities, before they can fetch outside the owned budget."""
    expected_root = Path(raw["cache_dir"]).absolute()
    cache_class, reader_class = module.VerifiedCache, module.ShardReader

    def artifact_path(artifact):
        key = json_sha256(
            {
                "repo": raw["repo_id"],
                "commit": raw["revision"],
                **artifact.model_dump(include={"path", "sha256", "size_bytes"}),
            }
        )
        return expected_root / key

    def checked_cache(root, spec):
        actual = Path(root).absolute()
        if actual != expected_root or any(
            p.is_symlink() for p in (actual, *actual.parents)
        ):
            raise ValueError("actual source cache is outside the owned path")
        if (
            spec.cache_bytes != rules["cache_bytes"]
            or spec.max_group_bytes != raw["max_group_bytes"]
            or spec.root is not None
        ):
            raise ValueError("source cache/admission policy changed")
        if observe:
            raise _PolicyObserved("cache")
        cache = cache_class(root, spec)
        if (
            cache.root.absolute() != expected_root
            or cache.spec.cache_bytes != rules["cache_bytes"]
        ):
            cache.close()
            raise ValueError("actual source cache is outside the owned budget")
        original_get = cache.get

        def checked_get(artifact):
            if (
                cache.root.absolute() != expected_root
                or cache.spec.cache_bytes != rules["cache_bytes"]
                or cache.spec.root is not None
            ):
                raise ValueError("actual source cache changed its owned budget")
            if (
                reads_started is not None
                and not reads_started[0]
                and artifact.sha256 != raw["manifest"]["sha256"]
            ):
                raise ValueError("uncounted shard preparation before stream startup")
            path = original_get(artifact)
            if (
                Path(path).absolute() != artifact_path(artifact)
                or Path(path).is_symlink()
            ):
                raise ValueError("cache artifact escaped the owned pinned path")
            if (
                path.stat().st_size != artifact.size_bytes
                or file_sha256(path) != artifact.sha256
            ):
                raise ValueError("cache artifact integrity changed")
            cache_usage(expected_root, rules["cache_bytes"])
            return path

        cache.__dict__["get"] = checked_get
        return cache

    def checked_reader(path, shard, *, provenance, max_group_bytes=None):
        if reads_started is not None and not reads_started[0]:
            raise ValueError("uncounted decoded preparation before stream startup")
        if max_group_bytes != raw["max_group_bytes"]:
            raise ValueError("actual reader row-group admission policy changed")
        if Path(path).absolute() != artifact_path(shard):
            raise ValueError("actual reader escaped the owned pinned cache")
        if observe:
            raise _PolicyObserved("reader")
        reader = reader_class(
            path, shard, provenance=provenance, max_group_bytes=max_group_bytes
        )
        if reader.max_group_bytes != raw[
            "max_group_bytes"
        ] or reader.path.absolute() != artifact_path(shard):
            reader.close()
            raise ValueError("actual reader changed its owned admission policy")
        original_read = reader.read

        def checked_read(index):
            if reader.max_group_bytes != raw[
                "max_group_bytes"
            ] or reader.path.absolute() != artifact_path(shard):
                raise ValueError("actual reader changed its owned admission policy")
            result = original_read(index)
            if artifact_reads is not None and (
                not artifact_reads or artifact_reads[-1] != shard.path
            ):
                artifact_reads.append(shard.path)
            return result

        reader.__dict__["read"] = checked_read
        return reader

    module.__dict__["VerifiedCache"] = checked_cache
    module.__dict__["ShardReader"] = checked_reader
    return expected_root


def cache_usage(root, bound):
    total = 0
    if not root.exists():
        return total
    for path in root.iterdir():
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            # A cache eviction or atomic partial-file rename can remove this name.
            continue
        if not stat.S_ISREG(metadata.st_mode) or (
            path.name != "lock"
            and re.fullmatch(r"[0-9a-f]{64}(\.partial)?", path.name) is None
        ):
            raise ValueError("source cache contains an unowned path")
        total += metadata.st_size
    if total > bound:
        raise ValueError("actual source cache exceeds frozen disk bound")
    return total


def source_policy_gate(request):
    """Exercise candidate's real V2 constructors without downloading test data."""
    import tennis3d.datasets.source as module

    root = Path(request["source"])
    raw = json.loads(
        (root / "configs/stage4-preparation-main-v1/train.source.json").read_text()
    )
    raw["cache_dir"] = str(Path(request["output_dir"]) / "source-cache")
    rules = {"cache_bytes": raw["cache_bytes"]}
    owned = install_source_guards(module, raw, rules, observe=True)
    try:
        module.V2Source(Path(request["output_dir"]) / "source.json", raw)
    except _PolicyObserved as observed:
        if str(observed) != "cache":
            raise
    else:
        raise ValueError("source did not use owned cache admission")
    spec = module.SourceSpec.model_validate(raw)
    artifact = spec.manifest
    key = json_sha256(
        {
            "repo": raw["repo_id"],
            "commit": raw["revision"],
            **artifact.model_dump(include={"path", "sha256", "size_bytes"}),
        }
    )
    source = object.__new__(module.V2Source)
    source.locations = [(0, 0, 0)]
    source.views, source.view_files = {}, {}
    source.active, source.reader, source.root = None, None, None
    source.spec = spec
    source.__dict__["cache"] = SimpleNamespace(pinned=None, get=lambda _: owned / key)
    source.__dict__["manifest"] = SimpleNamespace(shards=[artifact], provenance=None)
    try:
        source.read(0)
    except _PolicyObserved as observed:
        if str(observed) != "reader":
            raise
    else:
        raise ValueError("source did not use bounded reader admission")
    return {"owned_cache": True, "bounded_reader": True}


def measure(request, *, auditing=False, notify=_timing_boundary):
    root, rules = Path(request["source"]), request["rules"]
    workload = request["workload"]
    started = request["process_started"]
    descriptor = Path(request["descriptor"])
    cfg = recipe(root, descriptor).model_copy(update={"seed": workload["seed"]})
    torch.set_num_threads(rules["threads"])
    model, ema, optimizer = model_state(cfg, "cuda")
    torch.cuda.reset_peak_memory_stats()
    model_setup = time.perf_counter() - started
    # Candidate transport must still pass through this declared reader facility.
    import tennis3d.datasets.source as source_module

    raw_descriptor = json.loads(descriptor.read_text())
    reads_started, artifact_reads, fetched_artifacts = [False], [], []
    owned_cache = install_source_guards(
        source_module,
        raw_descriptor,
        rules,
        reads_started=reads_started,
        artifact_reads=artifact_reads,
    )
    transport = source_module.hub_chunks
    fetched, disk_peak = [0], [0]

    def counted(spec: SourceSpec, artifact: Artifact):
        fetched_artifacts.append(artifact.path)
        for chunk in transport(spec, artifact):
            fetched[0] += len(chunk)
            usage = cache_usage(owned_cache, rules["cache_bytes"]) + len(chunk)
            if usage > rules["cache_bytes"]:
                raise ValueError("actual transfer exceeded frozen disk budget")
            disk_peak[0] = max(disk_peak[0], usage)
            yield chunk

    source_module.__dict__["hub_chunks"] = counted
    traces, device_traces = [], []
    data = pipeline = None
    try:
        data_started = time.perf_counter()
        data = ViewDataset(descriptor)
        if data.identity != request["data_identity"]:
            raise ValueError("main data identity changed")
        source = data._source
        assert source is not None and source.cache is not None
        if source.root is not None or source.cache.root.absolute() != owned_cache:
            raise ValueError("actual source cache is outside the owned path")
        if (
            source.spec.cache_bytes != rules["cache_bytes"]
            or source.spec.subset_sha256 is not None
        ):
            raise ValueError("source cache/subset contract changed")
        # The initial state always comes from frozen oracle preparation.
        pipeline = create_pipeline(
            data,
            cfg,
            depth=request["depth"],
            max_bytes=rules["prefetch_bytes"],
            max_batches=rules["steps"],
        )
        fresh = pipeline.state_dict()
        for key in ("identity", "order", "rng", "data_rng"):
            if fresh[key] != request["initial_state"][key]:
                raise ValueError("candidate initial sampler semantics changed")
        pipeline.load_state_dict(request["initial_state"])
        data_setup = time.perf_counter() - data_started
        setup = time.perf_counter() - started
        notify("stream")
        reads_started[0] = True
        stream_started = time.perf_counter()
        sustained_started = None
        for step_index in range(rules["steps"]):
            batch = pipeline.next_batch()
            # Constructor/transfer guards run on the producer. Its live reader
            # may be between shards while this already-prepared batch is consumed.
            if auditing:
                traces.append(batch_digest(batch))
                f, target, valid = batch
                device_traces.append(
                    batch_digest((f.to("cuda"), target.to("cuda"), valid.to("cuda")))
                )
            else:
                update(model, ema, optimizer, cfg, batch, "cuda")
            if step_index + 1 == rules["steady_skip_steps"]:
                synchronize("cuda")
                sustained_started = time.perf_counter()
                notify("sustained")
        synchronize("cuda")
        completed = time.perf_counter()
        notify("complete")
        elapsed = completed - started
        stream_elapsed = completed - stream_started
        assert sustained_started is not None
        sustained_elapsed = completed - sustained_started
        final = pipeline.state_dict()
        # Expensive final-state diagnostics occur after the timed completed work.
        final_digest = digest(final)
        global_digest = digest(random_state())
        shutdown = time.perf_counter()
        pipeline.close()
        metrics = pipeline.metrics
        cache_peak = max(disk_peak[0], cache_usage(owned_cache, rules["cache_bytes"]))
        data.verify_identity()
        data.close()
        report = {
            "phase": "audit" if auditing else "measure",
            "source_sha256": request["source_sha256"],
            "cache_template_sha256": request["cache_template_sha256"],
            "read_artifact_paths": list(dict.fromkeys(artifact_reads)),
            "fetched_artifact_paths": list(dict.fromkeys(fetched_artifacts)),
            "elapsed_s": elapsed,
            "stream_elapsed_s": stream_elapsed,
            "sustained_elapsed_s": sustained_elapsed,
            "setup_s": setup,
            "model_setup_s": model_setup,
            "data_setup_s": data_setup,
            "shutdown_s": time.perf_counter() - shutdown,
            "steps": 0 if auditing else rules["steps"],
            "accepted_batches": rules["steps"],
            "windows": rules["steps"] * cfg.batch_size,
            "windows_per_second": rules["steps"] * cfg.batch_size / elapsed,
            "stall_mean_s": float(np.mean(metrics["stall_seconds"])),
            "stall_p95_s": float(np.percentile(metrics["stall_seconds"], 95)),
            "queue_delay_mean_s": float(np.mean(metrics["queue_delay_seconds"])),
            "ram_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
            "gpu_reserved_peak_bytes": torch.cuda.max_memory_reserved(),
            "cache_peak_bytes": cache_peak,
            "bytes_fetched": fetched[0],
            "pipeline": metrics,
            "sampler_sha256": final_digest,
            "rng_sha256": global_digest,
            "draws": final["draws"],
            "rejected": final["rejected"],
            "initial_state_sha256": digest(request["initial_state"]),
            "source_identity": request["data_identity"],
            "environment": environment("cuda"),
            "software": {
                "python": platform.python_version(),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch",
                        "numpy",
                        "pyarrow",
                        "pydantic",
                        "PyYAML",
                        "huggingface-hub",
                    )
                },
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
            },
            "hostname": socket.gethostname(),
            "loadavg": os.getloadavg(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
            "slurm_memory_mb": os.environ.get("SLURM_MEM_PER_NODE"),
        }
        report["contract_sha256"] = json_sha256(
            {
                "source_sha256": request["source_sha256"],
                "cache_template_sha256": request["cache_template_sha256"],
                "rules": rules,
                "workload": workload,
                "initial_state": request["initial_state"],
                "data_identity": request["data_identity"],
                "software": report["software"],
            }
        )
        if auditing:
            report.update(batch_sha256=traces, device_batch_sha256=device_traces)
        return report
    finally:
        if pipeline is not None:
            pipeline.close()
        if data is not None:
            data.close()


def fixture_gate(request):
    root = Path(request["source"])
    fixture = root / "tests/fixtures/lifting"
    cfg = recipe(root, fixture / "train.manifest.json", fixture=True)
    torch.set_num_threads(2)
    if request.get("maintained"):
        from tennis3d.lifting.checkpoint import read_checkpoint
        from tennis3d.lifting.train import TrainControl, train

        cfg = cfg.model_copy(
            update={"val_manifest": str(fixture / "val.manifest.json")}
        )
        result = train(
            cfg,
            request["output_dir"],
            control=TrainControl(validate_during_training=False, evaluate_at_end=False),
        )
        state = read_checkpoint(result["last_checkpoint"])
        return {
            key: digest(state[key])
            for key in ("model", "ema", "optimizer", "sampler", "rng")
        }
    model, ema, optimizer = model_state(cfg, "cpu")
    with ViewDataset(cfg.train_manifest) as data:
        pipeline = create_pipeline(data, cfg, depth=request["depth"], max_batches=4)
        try:
            for step in range(4):
                update(model, ema, optimizer, cfg, pipeline.next_batch(), "cpu")
                if request.get("resume") and step == 1:
                    # Write/read an actual checkpoint while the queue may contain work.
                    time.sleep(0.05)
                    checkpoint = Path(request["output_dir"]) / "resume.pt"
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "ema": ema.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "sampler": pipeline.state_dict(),
                            "rng": random_state(),
                        },
                        checkpoint,
                    )
                    pipeline.close()
                    saved = torch.load(checkpoint, weights_only=False)
                    model, ema, optimizer = model_state(cfg, "cpu")
                    model.load_state_dict(saved["model"])
                    ema.load_state_dict(saved["ema"])
                    optimizer.load_state_dict(saved["optimizer"])
                    restore_random_state(saved["rng"], "cpu")
                    pipeline = create_pipeline(
                        data, cfg, depth=request["depth"], max_batches=2
                    )
                    pipeline.load_state_dict(saved["sampler"])
            return {
                "model": digest(model.state_dict()),
                "ema": digest(ema.state_dict()),
                "optimizer": digest(optimizer.state_dict()),
                "sampler": digest(pipeline.state_dict()),
                "rng": digest(random_state()),
            }
        finally:
            pipeline.close()


def lifecycle_gate(request):
    """Independent fixed retained-view traces for each sampler lifecycle."""
    import threading

    from tennis3d.synthetic.corrupt import corrupt_observations

    root = Path(request["source"])
    fixture = root / "tests/fixtures/lifting/train.manifest.json"
    cfg = recipe(root, fixture, fixture=True)
    seed(42)
    torch.set_num_threads(2)
    traces = {}

    def consume(pipeline, count):
        return [
            (batch_digest(pipeline.next_batch()), digest(pipeline.state_dict()))
            for _ in range(count)
        ]

    with ViewDataset(fixture) as data:
        arrays = data[0]
        complete = []
        for value in (42, 44, 137):
            rng = np.random.default_rng(value)
            corrupted = corrupt_observations(
                arrays["ball_clean_uv"],
                arrays["t_s"],
                arrays["image_size"],
                data.simulation_config(),
                rng,
            )
            complete.append((digest(corrupted), digest(rng.bit_generator.state)))
        traces["complete_view_corruption"] = complete
        for name in (
            "ordinary_epochs",
            "rejections",
            "checkpoint_stop_resume",
            "failure_resume",
            "inflight_stop_resume",
        ):
            case_cfg = cfg
            if name == "rejections":
                case_cfg = cfg.model_copy(
                    update={"batch_size": 4, "min_frames": 1, "max_frames": 1}
                )
            pipeline = create_pipeline(data, case_cfg, depth=request["depth"])
            if name == "rejections":
                pipeline._sampler.corruption_config = (
                    pipeline._sampler.corruption_config.model_copy(
                        update={"missing_probability": 0.5}
                    )
                )
            try:
                if name in ("ordinary_epochs", "rejections"):
                    traces[name] = consume(pipeline, 8)
                    if name == "rejections" and pipeline.state_dict()["rejected"] == 0:
                        raise ValueError("rejection gate exercised no rejection")
                    continue
                traces[name] = consume(pipeline, 1)
                saved = pipeline.state_dict()
                if name == "checkpoint_stop_resume":
                    if request["depth"]:
                        with pipeline._condition:
                            if not pipeline._condition.wait_for(
                                lambda pipeline=pipeline: (
                                    len(pipeline._queue) >= request["depth"]
                                    or pipeline._failure is not None
                                ),
                                timeout=10,
                            ):
                                raise ValueError(
                                    "checkpoint gate did not reach queued work"
                                )
                            if pipeline._failure is not None:
                                raise ValueError("checkpoint producer failed")
                    checkpoint = Path(request["output_dir"]) / "lifecycle.pt"
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"sampler": saved, "rng": random_state()}, checkpoint)
                    pipeline.close()
                    actual = torch.load(checkpoint, weights_only=False)
                    traces[name].append(
                        (digest(pipeline.state_dict()), digest(actual["sampler"]))
                    )
                    saved = actual["sampler"]
                    restore_random_state(actual["rng"], "cpu")
                elif name == "failure_resume":
                    pipeline.close()
                    pipeline.load_state_dict(saved)
                    original = pipeline._sampler._next_view
                    calls = [0]

                    def failed_read(original=original, calls=calls):
                        result = original()
                        calls[0] += 1
                        if calls[0] == 2:
                            raise OSError("owned injected failure after draw")
                        return result

                    pipeline._sampler._next_view = failed_read
                    try:
                        pipeline.next_batch()
                    except OSError:
                        pass
                    else:
                        raise ValueError("failure gate did not fail")
                    traces[name].append((digest(pipeline.state_dict()), digest(saved)))
                    pipeline._sampler._next_view = original
                    pipeline.close()
                else:
                    pipeline.close()
                    pipeline.load_state_dict(saved)
                    if request["depth"]:
                        entered, release = threading.Event(), threading.Event()
                        original = pipeline._sampler.next_batch

                        def delayed_read(
                            original=original, entered=entered, release=release
                        ):
                            entered.set()
                            if not release.wait(10):
                                raise ValueError("inflight gate stalled")
                            return original()

                        pipeline._sampler.next_batch = delayed_read
                        errors = []

                        def consumer(pipeline=pipeline, errors=errors):
                            try:
                                pipeline.next_batch()
                            except RuntimeError:
                                errors.append(True)

                        thread = threading.Thread(target=consumer)
                        thread.start()
                        if not entered.wait(10):
                            raise ValueError("inflight gate did not enter read")
                        closer = threading.Thread(target=pipeline.close)
                        closer.start()
                        with pipeline._condition:
                            if not pipeline._condition.wait_for(
                                lambda pipeline=pipeline: pipeline._closed, timeout=10
                            ):
                                raise ValueError("inflight close did not start")
                        release.set()
                        thread.join(10)
                        closer.join(10)
                        if thread.is_alive() or closer.is_alive() or errors != [True]:
                            raise ValueError("inflight stop did not complete")
                    else:
                        pipeline.close()
                    traces[name].append((digest(pipeline.state_dict()), digest(saved)))
                # A fresh pipeline resumes the consumed state, no skipped/repeated draws.
                pipeline.close()
                pipeline = create_pipeline(
                    data, case_cfg, depth=request["depth"], max_batches=4
                )
                pipeline.load_state_dict(saved)
                traces[name].extend(consume(pipeline, 4))
            finally:
                pipeline.close()
    return {"traces": traces, "rng_sha256": digest(random_state())}


def prepare(request):
    """Oracle-only cache conditioning and deterministic workload state capture."""
    root, rules = Path(request["source"]), request["rules"]
    descriptor = Path(request["descriptor"])
    started = time.perf_counter()
    with ViewDataset(descriptor) as data:
        cfg = recipe(root, descriptor)
        identity = data.identity
        if (
            identity["membership_sha256"] != rules["membership_sha256"]
            or identity["revision"] != rules["revision"]
        ):
            raise ValueError("pinned main identity mismatch")
        source = data._source
        assert source is not None and source.cache is not None
        source.cache.pinned = source.cache.get(source.spec.manifest)
        states, shard_indices = {}, {}
        for workload in rules["workloads"]:
            wc = cfg.model_copy(update={"seed": workload["seed"]})
            state, shards, position = initial_state(data, wc, workload)
            states[workload["id"]] = state
            shard_indices[workload["id"]] = [
                int(x)
                for x in dict.fromkeys(
                    shards[
                        position : position + rules["steps"] * cfg.batch_size
                    ].tolist()
                )
            ]
        # Fill with real artifacts, past 4 GiB. No generated filler or staging61GB.
        eviction_target = shard_indices[rules["workloads"][-1]["id"]][0]
        stress = [eviction_target] + [
            i for i in range(len(source.manifest.shards)) if i != eviction_target
        ][: rules["eviction_shards"] - 1]
        fetched, evictions = 0, []
        for si in stress:
            before = set(source.cache.root.iterdir())
            shard = source.manifest.shards[si]
            source.cache.get(shard)
            fetched += shard.size_bytes
            evictions.extend(p.name for p in before - set(source.cache.root.iterdir()))
        target = source.cache.path(source.manifest.shards[eviction_target])
        if target.exists() or not evictions:
            raise ValueError("eviction conditioning did not evict the target")
        cache_keys = {p.name for p in source.cache.root.iterdir() if p.name != "lock"}
        templates = {}
        for workload in rules["workloads"]:
            targets = [source.manifest.shards[i] for i in shard_indices[workload["id"]]]
            target_keys = {source.cache.path(a).name for a in targets}
            artifacts = [source.spec.manifest]
            if workload["condition"] == "warm":
                artifacts.extend(targets)
            else:
                if cache_keys & target_keys:
                    raise ValueError(
                        "stress did not evict all declared transfer targets"
                    )
                survivors = sorted(
                    (
                        a
                        for a in source.manifest.shards
                        if source.cache.path(a).name in cache_keys
                        and source.cache.path(a).name not in target_keys
                    ),
                    key=lambda a: (-a.size_bytes, a.path),
                )
                size, target_bytes = (
                    source.spec.manifest.size_bytes,
                    sum(a.size_bytes for a in targets),
                )
                for artifact in survivors:
                    if size + target_bytes > rules["cache_bytes"]:
                        break
                    artifacts.append(artifact)
                    size += artifact.size_bytes
                if size + target_bytes <= rules["cache_bytes"]:
                    raise ValueError("survivors cannot exercise timed cache pressure")

            def entry(artifact):
                return {
                    "key": source.cache.path(artifact).name,
                    **artifact.model_dump(include={"path", "sha256", "size_bytes"}),
                }

            size = sum(a.size_bytes for a in artifacts)
            if size > rules["cache_bytes"]:
                raise ValueError("declared starting cache exceeds the fixed budget")
            templates[workload["condition"]] = {
                "artifacts": [
                    a.model_dump(include={"path", "sha256", "size_bytes"})
                    for a in artifacts
                ],
                "inventory": sorted(
                    (entry(a) for a in artifacts), key=lambda a: a["key"]
                ),
                "cache_bytes": rules["cache_bytes"],
                "initial_bytes": size,
                "available_bytes": rules["cache_bytes"] - size,
                "target_inventory": [entry(a) for a in targets],
                "target_bytes": sum(a.size_bytes for a in targets),
                "targets_present": workload["condition"] == "warm",
                "overflow_bytes": max(
                    0, size + sum(a.size_bytes for a in targets) - rules["cache_bytes"]
                )
                if workload["condition"] == "evicted"
                else 0,
            }
        return {
            "data_identity": identity,
            "states": states,
            "shard_indices": shard_indices,
            "conditioning_s": time.perf_counter() - started,
            "conditioning_bytes": fetched,
            "evicted_keys": evictions,
            "cache_peak_bytes": source.cache.peak_bytes,
            "stress_shards": stress,
            "cache_keys": sorted(cache_keys),
            "cache_templates": templates,
            "shards": [
                sh.model_dump(exclude={"records"}) for sh in source.manifest.shards
            ],
        }


def main(*, timing_check=_timing_boundary):
    timing_check()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("measure", "audit", "fixture", "lifecycle", "policy", "prepare"),
    )
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    # Verify source selection before importing data into a measurement.
    import tennis3d

    if (
        Path(tennis3d.__file__).resolve().parent
        != Path(request["source"]).resolve() / "tennis3d"
    ):
        raise ValueError("environment imported a different source checkout")
    result = {
        "measure": measure,
        "audit": lambda request: measure(request, auditing=True),
        "fixture": fixture_gate,
        "lifecycle": lifecycle_gate,
        "policy": source_policy_gate,
        "prepare": prepare,
    }[args.mode](request)
    result["request_sha256"] = file_sha256(args.request)
    result["report_sha256"] = json_sha256(result)
    args.output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")


import numpy as np
import torch
import yaml
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from tennis3d.artifacts import file_sha256, json_sha256
from tennis3d.datasets.schema import Artifact
from tennis3d.datasets.source import SourceSpec
from tennis3d.lifting.checkpoint import random_state, restore_random_state
from tennis3d.lifting.data import ViewDataset
from tennis3d.lifting.losses import vertical_radial_position_loss
from tennis3d.lifting.model import Lifter
from tennis3d.lifting.runtime import environment, synchronize
from tennis3d.lifting.settings import TrainConfig
from tennis3d.lifting.train import _cosine_policy, build_optimizer
from tennis3d.throughput.pipeline import create_pipeline

if __name__ == "__main__":
    main()
