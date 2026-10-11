"""Continue a legacy single-process evaluation with its original installed package.

This script is deliberately outside tennis3d: saved predictions remain bound to
their actual producer's package bytes. Run both phases with that immutable wheel,
not with a later tennis3d checkout. The wrapper has separate durable provenance.
"""

import argparse
import os
import signal
import time
from concurrent.futures import CancelledError, ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from tennis3d.artifacts import _loads, file_sha256, json_sha256, read_json, write_json
from tennis3d.evaluation.prediction_journal import (
    PredictionJournal,
    arrays_sha256,
    durable_directory,
    sync_directory,
)

MODES = ("corrupted", "clean")
COUNTS = {"rallies": 5000, "views": 10000, "frames": 1829263}


def forbidden():
    raise ValueError("CPU continuation cannot compute a prediction or missing report")


def saved_prediction(path, identity, index, record, mode):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"xyz", "covered", "disagreement", "metadata"}:
            raise ValueError("prediction archive schema differs")
        metadata = _loads(archive["metadata"].tobytes())
        values = {k: archive[k] for k in ("xyz", "covered", "disagreement")}
    key = metadata["key"]
    if (
        {k: key[k] for k in ("identity", "index", "record", "mode")}
        != {"identity": identity, "index": index, "record": record, "mode": mode}
        or metadata["metadata_sha256"]
        != json_sha256({k: v for k, v in metadata.items() if k != "metadata_sha256"})
        or metadata["arrays_sha256"] != arrays_sha256(values)
        or type(metadata["prediction_wall_s"]) is not float
        or not np.isfinite(metadata["prediction_wall_s"])
        or metadata["prediction_wall_s"] < 0
    ):
        raise ValueError("saved prediction checksum or producer differs")
    return metadata


def inventory(journal, dataset):
    files = {}
    for index, record in enumerate(dataset.records):
        for mode in MODES:
            relative = f"predictions/{index:05d}-{mode}.npz"
            path = journal.root / relative
            saved_prediction(path, journal.identity, index, record, mode)
            files[relative] = file_sha256(path)
    if {p.name for p in (journal.root / "predictions").iterdir()} != {
        Path(name).name for name in files
    }:
        raise ValueError("prediction journal has extra files")
    return files


def counts(dataset, expected):
    value = {
        "rallies": len({r["rally_id"] for r in dataset.records}),
        "views": len(dataset),
        "frames": sum(r["frames"] for r in dataset.records),
    }
    if value != expected:
        raise ValueError("complete source membership differs")
    return value


def check_completion(output, operations, wrapper, invocations):
    path = operations / "analysis-complete.json"
    if not path.exists():
        return
    receipt = read_json(path)
    names = {"metrics.json", "report.md"} | {
        f"{directory}/{index:05d}.json"
        for directory in ("physical", "physical-journal")
        for index in range(COUNTS["views"])
    }
    if (
        receipt["status"] != "measured"
        or receipt["wrapper_sha256"] != wrapper
        or receipt["inference_seal_sha256"]
        != file_sha256(operations / "inference-complete.json")
        or set(receipt["files"]) != names
        or receipt["invocations"] != {p.name: file_sha256(p) for p in invocations}
        or any(
            file_sha256(output / name) != digest
            for name, digest in receipt["files"].items()
        )
        or any(
            {p.name for p in (output / directory).iterdir()}
            != {f"{index:05d}.json" for index in range(COUNTS["views"])}
            for directory in ("physical", "physical-journal")
        )
    ):
        raise ValueError("previous completed continuation changed")


def persist_scoring(output):
    # Legacy write_report uses write_text; explicitly persist its content too.
    for name in ("metrics.json", "report.md"):
        with (output / name).open("rb") as stream:
            os.fsync(stream.fileno())
    for directory in ("physical", "physical-journal"):
        durable_directory(output / directory)
    sync_directory(output)


def inference(release, source, journal, device, expected):
    from tennis3d.evaluation.synthetic import input_mode
    from tennis3d.inference.predict import predict_view
    from tennis3d.lifting.checkpoint import load_lifter
    from tennis3d.lifting.data import ViewDataset
    from tennis3d.lifting.runtime import synchronize

    model, _ = load_lifter(release.checkpoint.path, device)
    with ViewDataset(source) as dataset:
        dataset.verify_identity()
        counts(dataset, expected)
        for index, record in enumerate(dataset.records):
            if journal.stop_requested:
                raise InterruptedError("warning received; predictions retained")
            paths = [journal.root / f"predictions/{index:05d}-{m}.npz" for m in MODES]
            if all(p.exists() for p in paths):
                for path, mode in zip(paths, MODES, strict=True):
                    saved_prediction(path, journal.identity, index, record, mode)
                    journal.counts["predictions_reused"] += 1
                continue
            arrays = dataset[index]
            if not (arrays["camera_valid"] & arrays["label_valid"]).any():
                raise ValueError("view has no supported labels")
            for mode in MODES:
                inputs = input_mode(arrays, mode)

                def compute(inputs=inputs):
                    synchronize(device)
                    started = time.perf_counter()
                    values = predict_view(
                        model, inputs, device=device, require_complete=True
                    )
                    synchronize(device)
                    return *values, time.perf_counter() - started

                journal.prediction(index, record, arrays, mode, compute)
        dataset.verify_identity()
        if file_sha256(release.checkpoint.path) != release.checkpoint.sha256:
            raise ValueError("checkpoint changed")
        return {
            "status": "predictions_complete_not_scored",
            "producer_identity": journal.identity,
            "source": {
                "identity": dataset.identity,
                "manifest_sha256": dataset.manifest_sha256,
                "records_sha256": json_sha256(dataset.records),
                "counts": counts(dataset, expected),
            },
            "files": inventory(journal, dataset),
        }


def fit_inputs(release, dataset, record, arrays, inputs, xyz, covered):
    # Exact numeric arrays and metadata of the original 3b7ced35 callback.
    numeric = {
        "t_s": arrays["t_s"],
        "xyz_m": xyz,
        "available": covered,
        "input_observed": inputs["ball_observed"],
        "truth_xyz_m": arrays["xyz_m"],
        "label_valid": arrays["label_valid"] & arrays["camera_valid"],
        **{
            k: arrays[k]
            for k in ("event_t_s", "event_code", "event_end_t_s")
            if k in arrays
        },
    }
    metadata = {
        "artifact_id": f"{record['rally_id']}-view-{record['view_id']}",
        "mode": "corrupted",
        "domain": "synthetic",
        "split": "test",
        "time_basis": "simulation",
        "timing_note": "Exact frozen simulation timestamps; full views without time deduplication",
        "source_identity": {**dataset.identity, "record": record},
        "prediction_identity": {
            "checkpoint_sha256": release.checkpoint.sha256,
            "step": release.step,
            "weights": "EMA",
            "mode": "corrupted",
        },
        "segmentation_source": "simulator",
        "segmentation_method": "saved contact events and event episodes",
        "event_grouping": "saved simulator contact episodes; no truth trajectory supplied to fitting",
    }
    return numeric, metadata


def start_worker():
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def timed_fit(numeric, metadata):
    from tennis3d.evaluation.physical import evaluate_physical

    wall, cpu = time.perf_counter(), time.process_time()
    try:
        return (
            evaluate_physical(numeric, metadata=metadata),
            None,
            time.perf_counter() - wall,
            time.process_time() - cpu,
        )
    except Exception as error:  # noqa: BLE001 - preserve measured cost of failed fits
        return (
            None,
            f"{type(error).__name__}: {error}",
            time.perf_counter() - wall,
            time.process_time() - cpu,
        )


class ReplayJournal(PredictionJournal):
    def prediction(self, index, record, arrays, mode, compute):
        return super().prediction(index, record, arrays, mode, forbidden)

    def physical_report(self, index, compute):
        return super().physical_report(index, forbidden)


def analysis(
    release,
    provenance,
    source,
    journal,
    seal,
    workers,
    expected,
    timing,
    *,
    fit=timed_fit,
):
    from tennis3d.evaluation.delivery import _synthetic_released
    from tennis3d.evaluation.synthetic import input_mode
    from tennis3d.lifting.data import ViewDataset

    failures, pending = [], {}

    def commit(index, future):
        try:
            report, error, wall, cpu = future.result()
        except CancelledError:
            return
        except Exception as death:  # noqa: BLE001 - no final result after worker death
            timing["fits_without_timing"] += 1
            failures.append(str(death))
            return
        timing["worker_wall_s"] += wall
        timing["worker_cpu_s"] += cpu
        if error:
            failures.append(error)
        else:
            PredictionJournal.physical_report(journal, index, lambda: report)

    with ViewDataset(source) as dataset:
        dataset.verify_identity()
        counts(dataset, expected)
        if (
            seal["producer_identity"] != journal.identity
            or seal["source"]
            != {
                "identity": dataset.identity,
                "manifest_sha256": dataset.manifest_sha256,
                "records_sha256": json_sha256(dataset.records),
                "counts": expected,
            }
            or seal["files"] != inventory(journal, dataset)
        ):
            raise ValueError("completed predictions or pinned membership differ")
        started = time.perf_counter()
        finished = False
        with ProcessPoolExecutor(
            workers, mp_context=get_context("spawn"), initializer=start_worker
        ) as pool:
            try:
                for index, record in enumerate(dataset.records):
                    if journal.stop_requested or failures:
                        break
                    arrays = dataset[index]
                    xyz, covered, _, _ = journal.prediction(
                        index, record, arrays, "corrupted", forbidden
                    )
                    if (
                        journal.root / "physical-journal" / f"{index:05d}.json"
                    ).exists():
                        journal.physical_report(index, forbidden)
                        continue
                    numeric, metadata = fit_inputs(
                        release,
                        dataset,
                        record,
                        arrays,
                        input_mode(arrays, "corrupted"),
                        xyz,
                        covered,
                    )
                    pending[index] = pool.submit(fit, numeric, metadata)
                    while len(pending) >= 2 * workers and not failures:
                        oldest = next(iter(pending))
                        commit(oldest, pending.pop(oldest))
                else:
                    finished = True
            finally:
                if not finished:
                    for future in pending.values():
                        future.cancel()
                for index, future in pending.items():
                    commit(index, future)
        timing["physics_pass_wall_s"] = time.perf_counter() - started
        if failures:
            raise ValueError(f"physical worker failed: {failures[0]}")
        if journal.stop_requested or not finished:
            raise InterruptedError("warning received; finished reports retained")
        dataset.verify_identity()
    # The original scorer reads every raw source and both modes, validates each
    # journal input hash, and computes the unchanged full denominators/summary.
    return _synthetic_released(
        release, provenance, source, journal, "cpu", expected_counts=expected
    )


def run(
    phase, release_path, source, output, operations, *, workers=4, scheduler_receipts=()
):
    if phase == "analysis":
        if (
            os.environ.get("SLURM_JOB_GPUS")
            or os.environ.get("SLURM_STEP_GPUS")
            or os.environ.get("SLURM_GPUS_ON_NODE", "0") not in ("", "0")
        ):
            raise ValueError("analysis must release the GPU allocation")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from tennis3d.evaluation.delivery import checked_release
    from tennis3d.lifting.runtime import environment
    from tennis3d.provenance import code_identity

    output, operations = Path(output).resolve(), Path(operations).resolve()
    if (
        output == operations
        or output in operations.parents
        or operations in output.parents
    ):
        raise ValueError("continuation provenance needs a separate directory")
    spec = read_json(source)
    if (
        spec.get("revision") != "c2e782053694cfde305362e670ebbb4f4e70884c"
        or spec.get("manifest", {}).get("sha256")
        != "b32fe9cc97226c9d67468bcf7724f7d3ffc8924541798796744f9ba62bba6707"
        or spec.get("split") != "test"
        or spec.get("subset_sha256")
        or spec.get("exclude_rally_ids")
        or spec.get("repo_id") != "N-il/tennis-3d-synthetic"
        or spec.get("cache_bytes", 1 << 31) > 1 << 30
        or spec.get("max_group_bytes", 1 << 31) > 1 << 28
    ):
        raise ValueError("continuation requires complete pinned test source")
    release, _, provenance = checked_release(release_path)
    identity = read_json(output / "identity.json")
    if (
        set(identity)
        != {
            "environment",
            "release_sha256",
            "source_descriptor_sha256",
            "source_sha256",
            "device",
        }
        or identity["release_sha256"] != provenance["release_sha256"]
        or identity["source_descriptor_sha256"] != file_sha256(source)
        or identity["source_sha256"] != code_identity()["source_sha256"]
    ):
        raise ValueError(
            "continuation must import the original producer's immutable package"
        )
    device = identity["device"] if phase == "inference" else "cpu"
    if phase == "inference" and environment(device) != identity["environment"]:
        raise ValueError("inference producer environment differs")
    if type(workers) is not int or not 1 <= workers <= min(
        8, len(os.sched_getaffinity(0)), int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    ):
        raise ValueError("worker count exceeds available CPUs")
    journal_type = PredictionJournal if phase == "inference" else ReplayJournal
    with journal_type(output, identity, resume=True) as journal:
        durable_directory(operations)
        invocations = sorted(operations.glob("[0-9][0-9][0-9][0-9].json"))
        wrapper = file_sha256(__file__)
        previous = None
        for ordinal, invocation in enumerate(invocations):
            prior = read_json(invocation)
            if (
                invocation.name != f"{ordinal:04d}.json"
                or prior["wrapper_sha256"] != wrapper
                or prior["producer_identity"] != identity
                or prior["previous_invocation_sha256"] != previous
            ):
                raise ValueError(
                    "continuation implementation, producer or invocation chain changed"
                )
            previous = file_sha256(invocation)
        check_completion(output, operations, wrapper, invocations)
        seal_path = operations / "inference-complete.json"
        if phase == "inference" and seal_path.exists():
            raise ValueError("inference is complete; verify and score on CPU")
        if phase == "analysis":
            # Retire the verified receipt before publishing a new invocation.
            # A killed restart can then resume without leaving a stale receipt.
            (operations / "analysis-complete.json").unlink(missing_ok=True)
            sync_directory(operations)
        path = operations / f"{len(invocations):04d}.json"
        record = {
            "phase": phase,
            "status": "running",
            "started_epoch_s": time.time(),
            "wrapper_sha256": wrapper,
            "producer_identity": identity,
            "consumer_environment": environment(device),
            "workers": workers,
            "outstanding_views_limit": 2 * workers,
            "scheduler": {
                k: v for k, v in os.environ.items() if k.startswith("SLURM_")
            },
            "scheduler_receipts": [
                {"path": str(p), "sha256": file_sha256(p)} for p in scheduler_receipts
            ],
            "previous_invocation_sha256": file_sha256(invocations[-1])
            if invocations
            else None,
            "timing": {
                "worker_cpu_s": 0.0,
                "worker_wall_s": 0.0,
                "fits_without_timing": 0,
            },
        }
        write_json(path, record)
        sync_directory(operations)
        started, cpu = time.perf_counter(), time.process_time()
        try:
            if phase == "inference":
                seal = inference(release, source, journal, device, COUNTS)
                seal["wrapper_sha256"] = wrapper
                seal["sha256"] = json_sha256(seal)
                write_json(seal_path, seal)
            else:
                seal = read_json(seal_path)
                if (
                    seal["sha256"]
                    != json_sha256({k: v for k, v in seal.items() if k != "sha256"})
                    or seal["wrapper_sha256"] != wrapper
                ):
                    raise ValueError("prediction completion seal differs")
                for name in ("metrics.json", "report.md"):
                    (output / name).unlink(missing_ok=True)
                sync_directory(output)
                sync_directory(operations)
                analysis(
                    release,
                    provenance,
                    source,
                    journal,
                    seal,
                    workers,
                    COUNTS,
                    record["timing"],
                )
                persist_scoring(output)
                files = {
                    name: file_sha256(output / name)
                    for name in ("metrics.json", "report.md")
                }
                for directory in ("physical", "physical-journal"):
                    paths = sorted((output / directory).iterdir())
                    if {p.name for p in paths} != {
                        f"{i:05d}.json" for i in range(COUNTS["views"])
                    }:
                        raise ValueError("physical report file set differs")
                    files.update(
                        {f"{directory}/{p.name}": file_sha256(p) for p in paths}
                    )
            record["status"] = "completed"
        except BaseException as error:
            record.update(
                status="interrupted"
                if isinstance(error, InterruptedError)
                else "failed",
                error=f"{type(error).__name__}: {error}",
            )
            raise
        finally:
            record.update(
                ended_epoch_s=time.time(),
                wall_s=time.perf_counter() - started,
                process_cpu_s=time.process_time() - cpu,
                journal_counts=dict(journal.counts),
            )
            write_json(path, record, overwrite=True)
            sync_directory(operations)
        if phase == "analysis":
            write_json(
                operations / "analysis-complete.json",
                {
                    "status": "measured",
                    "files": files,
                    "inference_seal_sha256": file_sha256(seal_path),
                    "wrapper_sha256": wrapper,
                    "invocations": {
                        p.name: file_sha256(p)
                        for p in sorted(operations.glob("[0-9][0-9][0-9][0-9].json"))
                    },
                },
            )
            sync_directory(operations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("inference", "analysis"))
    for name in ("release", "source", "output", "operations"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scheduler-receipt", type=Path, action="append", default=[])
    args = parser.parse_args()
    run(
        args.phase,
        args.release,
        args.source,
        args.output,
        args.operations,
        workers=args.workers,
        scheduler_receipts=args.scheduler_receipt,
    )


if __name__ == "__main__":
    main()
