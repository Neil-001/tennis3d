"""Final synthetic evaluation as a sealed inference phase and a CPU-only analysis.

The inference phase saves every clean/corrupted full-view prediction with its
original duration and then publishes a seal over all of them. The analysis phase
runs the single-process command's scoring on those exact predictions without a
model or CUDA. Each phase owns a new root, identity, writer lock and journal;
neither accepts a root written by the other or by the single-process command.
"""

import fcntl
import hashlib
import io
import os
import platform
import signal
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from tennis3d.artifacts import _loads, file_sha256, json_sha256, read_json, write_json
from tennis3d.evaluation.prediction_journal import (
    PredictionJournal,
    arrays_sha256,
    durable_directory,
    sync_directory,
)
from tennis3d.schemas import StrictModel

MODES = ("corrupted", "clean")
INFERENCE, ANALYSIS = "synthetic_inference", "synthetic_analysis"
SEAL = "inference-complete.json"
RECEIPT = "analysis-complete.json"
INFERENCE_MEMBERS = {"identity.json", "writer.lock", "invocations", "predictions"}
ANALYSIS_MEMBERS = {
    "identity.json",
    "writer.lock",
    "invocations",
    "physical",
    "physical-journal",
    "metrics.json",
    "report.md",
    RECEIPT,
}
STATUSES = ("running", "complete", "interrupted", "failed")
MAX_PHYSICS_WORKERS = 8
PREDICTION_ARRAYS = ("xyz", "covered", "disagreement")
PROTOCOL = {
    "modes": list(MODES),
    "window": {"max_frames": 250, "stride": 125, "merge": "centre weights"},
    "weights": "EMA",
}
SCHEDULER_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_ACCOUNT",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_CPUS_PER_TASK",
    "SLURM_MEM_PER_NODE",
    "SLURM_JOB_START_TIME",
    "SLURM_JOB_END_TIME",
    "SLURM_RESTART_COUNT",
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)
TIMING_DEFINITIONS = {
    "prediction_wall_s": "Original synchronized predict_view wall time of one full view and mode, saved with the prediction when it was computed and never re-measured. It includes CPU feature building, host/device transfer and window merging; it is not GPU kernel time or utilization.",
    "invocation_wall_s": "Process wall time of one phase invocation. A terminated invocation keeps status running with no duration.",
    "allocated_gpu_time": "Only scheduler accounting measures allocated GPU time. The recorded scheduler environment and reservation spans are self-reported.",
    "analysis": "Reader, replay, scoring and physics wall times are CPU-only analysis work. They are not inference time.",
    "process_cpu_s": "User plus system CPU seconds of the invocation's own process, excluding worker processes.",
    "physics_fit": "Wall seconds of physical fits computed in the analysis process itself.",
    "physics_fit_worker_wall": "Sum over returned worker fits, failed ones included, of each fit's own wall seconds. Fits overlap, so this exceeds elapsed time.",
    "physics_fit_worker_cpu": "Sum over returned worker fits, failed ones included, of the worker process's user plus system CPU seconds during the fit. Fits lost with a dead worker are counted in physics.fits_without_timing and are not in this sum.",
    "physics_worker_pass": "Elapsed wall seconds of the pass that reads inputs, verifies corrupted predictions, runs worker fits and commits their reports.",
}


class GpuReservation(StrictModel):
    """The ledger owner's grant of allocated GPU-seconds to one inference root."""

    schema_version: Literal["1.0"] = "1.0"
    ledger_owner: str = Field(min_length=1)
    recorded_utc: str
    cap_gpu_seconds: float = Field(gt=0)
    # Everything already charged or reserved elsewhere against the cap.
    charged_gpu_seconds: float = Field(ge=0)
    reserved_gpu_seconds: float = Field(gt=0)
    gpus: Literal[1] = 1
    # Absolute path of the only inference root this grant may fund.
    inference_root: str = Field(min_length=1)


def utc_now():
    return datetime.now(UTC).isoformat()


def hardware(device):
    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    result = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "cpu_affinity": len(os.sched_getaffinity(0)),
        "memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        "gpu": None,
    }
    if device == "cuda":
        import torch

        properties = torch.cuda.get_device_properties(0)
        result["gpu"] = {
            "name": properties.name,
            "memory_bytes": properties.total_memory,
            "uuid": str(getattr(properties, "uuid", "")) or None,
            "visible_devices": torch.cuda.device_count(),
        }
    return result


def members(directory):
    """Published names in a directory; dot-prefixed names are unpublished temporaries."""
    directory = Path(directory)
    if not directory.is_dir():
        return set()
    return {p.name for p in directory.iterdir() if not p.name.startswith(".")}


def read_invocations(root, phase, identity):
    """Return a phase root's own invocation records in order, with file hashes."""
    names = sorted(members(Path(root) / "invocations"))
    if names != [f"{i:04d}.json" for i in range(len(names))]:
        raise ValueError("phase invocation records are not a contiguous sequence")
    rows, previous = [], None
    for ordinal, name in enumerate(names):
        path = Path(root) / "invocations" / name
        try:
            row = read_json(path)
            wall_s = row["wall_s"]
            valid = (
                (row["schema_version"], row["phase"], row["ordinal"])
                == ("1.0", phase, ordinal)
                and row["previous_record_sha256"] == previous
                and row["identity_sha256"] == json_sha256(identity)
                and row["status"] in STATUSES
                and (
                    wall_s is None
                    if row["status"] == "running"
                    else type(wall_s) is float and wall_s >= 0
                )
                and isinstance(row["timings_s"], dict)
                and isinstance(row["counts"], dict)
                and "gpu_reservation" in row
            )
        except (OSError, ValueError, KeyError, TypeError):
            valid = False
        if not valid:
            raise ValueError(f"invocation record {name} is damaged or foreign")
        previous = file_sha256(path)
        rows.append({**row, "record_sha256": previous})
    return rows


def record_hashes(rows):
    return [
        {
            "file": f"invocations/{row['ordinal']:04d}.json",
            "sha256": row["record_sha256"],
        }
        for row in rows
    ]


class Invocation:
    """Durable start and end record of one phase process under the writer lock."""

    def __init__(self, root, phase, identity, device, started, counts):
        from tennis3d.lifting.runtime import environment
        from tennis3d.provenance import code_identity

        self.root = Path(root)
        self.prior = read_invocations(self.root, phase, identity)
        self.path = self.root / "invocations" / f"{len(self.prior):04d}.json"
        self.started = started
        self.cpu_started = time.process_time()
        self.timings: dict[str, float] = {}
        self.budget = None
        code = code_identity()
        self.record = {
            "schema_version": "1.0",
            "phase": phase,
            "ordinal": len(self.prior),
            # Each record names the finished bytes of the one before it.
            "previous_record_sha256": self.prior[-1]["record_sha256"]
            if self.prior
            else None,
            "status": "running",
            "error": None,
            "started_utc": utc_now(),
            "ended_utc": None,
            "wall_s": None,
            "identity_sha256": json_sha256(identity),
            "device": device,
            "pid": os.getpid(),
            "hardware": hardware(device),
            "environment": environment(device),
            "code": {
                "code_revision": code["code_revision"],
                "source_sha256": code["source_sha256"],
            },
            "scheduler_environment": {
                key: os.environ[key] for key in SCHEDULER_KEYS if key in os.environ
            },
            "gpu_reservation": None,
            "timings_s": self.timings,
            "counts": counts,
        }

    def add(self, name, seconds):
        self.timings[name] = self.timings.get(name, 0.0) + seconds

    def snapshot(self):
        return {**self.record, "wall_s": time.perf_counter() - self.started}

    def __enter__(self):
        durable_directory(self.path.parent)
        write_json(self.path, self.record)
        sync_directory(self.path.parent)
        return self

    def __exit__(self, kind, error, traceback):
        import torch

        if self.budget is not None:
            self.budget.record["allocation_elapsed_s"] = self.budget.elapsed()
        self.record.update(
            status="complete"
            if error is None
            else "interrupted"
            if isinstance(error, InterruptedError)
            else "failed",
            error=None if error is None else f"{type(error).__name__}: {error}",
            ended_utc=utc_now(),
            wall_s=time.perf_counter() - self.started,
            process_cpu_s=time.process_time() - self.cpu_started,
            cuda_initialized=torch.cuda.is_initialized(),
        )
        write_json(self.path, self.record, overwrite=True)
        sync_directory(self.path.parent)


class Budget:
    """Hold one CUDA invocation inside the remaining reserved GPU-seconds.

    The grant names its only inference root. Attach it before the invocation
    starts so that the durable start record already names it. Under a scheduler
    the job's own time limit must fit in what remains, so a stalled process
    cannot outlive the reservation.
    """

    def __init__(self, path, invocation):
        reservation = GpuReservation.model_validate(read_json(path))
        if (
            reservation.charged_gpu_seconds + reservation.reserved_gpu_seconds
            > reservation.cap_gpu_seconds
        ):
            raise ValueError("GPU reservation exceeds the remaining recorded budget")
        if Path(reservation.inference_root) != invocation.root.resolve():
            raise ValueError("GPU reservation was granted to another inference root")
        digest = file_sha256(path)
        used = 0.0
        for row in invocation.prior:
            prior = row["gpu_reservation"]
            if prior is None or prior["sha256"] != digest:
                continue
            if row["status"] == "running":
                raise ValueError(
                    "an earlier invocation under this GPU reservation has no end record; "
                    "the ledger owner must charge its scheduler usage and record a new reservation"
                )
            used += prior["allocation_elapsed_s"]
        self.remaining = reservation.reserved_gpu_seconds - used
        start, end = (
            os.environ.get(f"SLURM_JOB_{name}_TIME", "") for name in ("START", "END")
        )
        limit = None
        if "SLURM_JOB_ID" in os.environ:
            if not (start.isdigit() and end.isdigit()):
                raise ValueError(
                    "the scheduler did not export the job start and end times needed "
                    "to bound this allocation"
                )
            limit = float(end) - float(start)
            if limit > self.remaining:
                raise ValueError(
                    "the job time limit exceeds the remaining reserved GPU-seconds"
                )
        self.basis = "SLURM_JOB_START_TIME" if limit is not None else "invocation_start"
        self.origin = (
            float(start)
            if limit is not None
            else time.time() - (time.perf_counter() - invocation.started)
        )
        invocation.budget = self
        self.record = invocation.record["gpu_reservation"] = {
            "sha256": digest,
            "reservation": reservation.model_dump(),
            "used_before_s": used,
            "allocation_basis": self.basis,
            "scheduler_time_limit_s": limit,
            "allocation_elapsed_s": self.elapsed(),
        }

    def elapsed(self):
        return max(time.time() - self.origin, 0.0)

    def check(self):
        self.record["allocation_elapsed_s"] = self.elapsed()
        if self.record["allocation_elapsed_s"] >= self.remaining:
            raise InterruptedError(
                "reserved GPU-seconds are spent; resume saved predictions under a new reservation"
            )


def read_prediction(path):
    """Return (file SHA-256, metadata, arrays) of one self-consistent prediction."""
    path = Path(path)
    try:
        payload = path.read_bytes()
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if sorted(archive.files) != sorted((*PREDICTION_ARRAYS, "metadata")):
                raise ValueError("unexpected prediction arrays")
            metadata = _loads(archive["metadata"].tobytes())
            values = {key: archive[key] for key in PREDICTION_ARRAYS}
        consistent = (
            metadata["metadata_sha256"]
            == json_sha256(
                {k: v for k, v in metadata.items() if k != "metadata_sha256"}
            )
            and metadata["arrays_sha256"] == arrays_sha256(values)
            and type(metadata["prediction_wall_s"]) is float
            and metadata["prediction_wall_s"] >= 0
        )
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
        consistent = False
    if not consistent:
        raise ValueError(f"prediction {path.name} is missing, torn or damaged")
    return hashlib.sha256(payload).hexdigest(), metadata, values


def prediction_name(index, mode):
    return f"predictions/{index:05d}-{mode}.npz"


def sealed_prediction(root, identity, index, record, mode):
    """Read one prediction; return its seal entry derived from the file, and arrays."""
    digest, metadata, values = read_prediction(root / prediction_name(index, mode))
    key = metadata["key"]
    if {k: v for k, v in key.items() if k != "input_sha256"} != {
        "identity": identity,
        "index": index,
        "record": record,
        "mode": mode,
    }:
        raise ValueError("prediction journal checksum or identity differs")
    return {
        "index": index,
        "mode": mode,
        "file": prediction_name(index, mode),
        "file_sha256": digest,
        "input_sha256": key["input_sha256"],
        "arrays_sha256": metadata["arrays_sha256"],
        "metadata_sha256": metadata["metadata_sha256"],
        "prediction_wall_s": metadata["prediction_wall_s"],
    }, values


def wall_totals(entries):
    return {
        mode: sum(row["prediction_wall_s"] for row in entries if row["mode"] == mode)
        for mode in MODES
    }


def inference_identity(release, provenance, source_path, device):
    from tennis3d.lifting.runtime import environment
    from tennis3d.provenance import code_identity

    return {
        "phase": INFERENCE,
        "phase_schema_version": "1.0",
        "environment": environment(device),
        "release_sha256": provenance["release_sha256"],
        "checkpoint_sha256": release.checkpoint.sha256,
        "checkpoint_step": release.step,
        "source_descriptor_sha256": file_sha256(source_path),
        "source_sha256": code_identity()["source_sha256"],
        "device": device,
        "protocol": PROTOCOL,
    }


def infer(
    release_path,
    source_path,
    output,
    *,
    device="cpu",
    resume=False,
    reservation_path=None,
):
    """Save and seal every full-view prediction of the pinned complete test split."""
    from tennis3d.evaluation.delivery import checked_release, require_final_source

    release, _, provenance = checked_release(release_path)
    require_final_source(source_path)
    return _infer_released(
        release,
        provenance,
        source_path,
        output,
        device,
        resume=resume,
        reservation_path=reservation_path,
    )


def _infer_released(
    release,
    provenance,
    source_path,
    output,
    device,
    *,
    resume=False,
    reservation_path=None,
    expected_counts=None,
):
    from tennis3d.evaluation.delivery import full_test_counts
    from tennis3d.evaluation.synthetic import input_mode
    from tennis3d.inference.predict import predict_view
    from tennis3d.lifting.checkpoint import load_lifter
    from tennis3d.lifting.data import ViewDataset
    from tennis3d.lifting.runtime import resolve_device, synchronize

    process_started = time.perf_counter()
    device = resolve_device(device)
    if (device == "cuda") != (reservation_path is not None):
        raise ValueError(
            "CUDA inference requires the ledger owner's GPU reservation; CPU inference takes none"
        )
    if device == "cuda" and (Path(output) / SEAL).exists():
        raise ValueError(
            "inference is already sealed; verify it on CPU through the analysis phase"
        )
    identity = inference_identity(release, provenance, source_path, device)
    with PredictionJournal(output, identity, resume=resume) as journal:
        root = journal.root
        if (root / SEAL).exists():
            with ViewDataset(source_path) as dataset:
                full_test_counts(dataset, expected_counts)
                return checked_seal(root, identity, dataset, predictions=True)
        invocation = Invocation(
            root, INFERENCE, identity, device, process_started, journal.counts
        )
        budget = reservation_path and Budget(reservation_path, invocation)
        with invocation:
            if budget:
                budget.check()
            started = time.perf_counter()
            model, _ = load_lifter(release.checkpoint.path, device)
            invocation.add("model_load", time.perf_counter() - started)
            entries = []
            with ViewDataset(source_path) as dataset:
                dataset.verify_identity()
                counts = full_test_counts(dataset, expected_counts)
                for index, record in enumerate(dataset.records):
                    if journal.stop_requested:
                        raise InterruptedError(
                            "evaluation warning received; resume saved predictions"
                        )
                    if budget:
                        budget.check()
                    if not all(
                        (root / prediction_name(index, mode)).exists() for mode in MODES
                    ):
                        started = time.perf_counter()
                        arrays = dataset[index]
                        invocation.add("reader", time.perf_counter() - started)
                        if not (arrays["camera_valid"] & arrays["label_valid"]).any():
                            raise ValueError("evaluation view has no supported labels")
                        for mode in MODES:
                            inputs = input_mode(arrays, mode)

                            def compute(inputs=inputs):
                                if budget:
                                    budget.check()
                                synchronize(device)
                                started = time.perf_counter()
                                xyz, covered, disagreement = predict_view(
                                    model, inputs, device=device, require_complete=True
                                )
                                synchronize(device)
                                wall_s = time.perf_counter() - started
                                invocation.add("prediction", wall_s)
                                return xyz, covered, disagreement, wall_s

                            journal.prediction(index, record, arrays, mode, compute)
                    else:
                        # A resumed view skips the reader under the GPU allocation.
                        # Its saved input checksum is sealed below and the analysis
                        # phase compares it with the pinned source.
                        journal.counts["predictions_reused"] += len(MODES)
                    for mode in MODES:
                        entries.append(
                            sealed_prediction(root, identity, index, record, mode)[0]
                        )
                dataset.verify_identity()
                synchronize(device)
                if file_sha256(release.checkpoint.path) != release.checkpoint.sha256:
                    raise ValueError("checkpoint changed during final inference")
                if {
                    f"predictions/{name}" for name in members(root / "predictions")
                } != {row["file"] for row in entries} or members(
                    root
                ) != INFERENCE_MEMBERS:
                    raise ValueError("inference root holds undeclared files")
                source = {
                    "descriptor_sha256": identity["source_descriptor_sha256"],
                    "manifest_sha256": dataset.manifest_sha256,
                    "identity": dataset.identity,
                    "split": dataset.manifest["split"],
                    "counts": counts,
                    "records_sha256": json_sha256(dataset.records),
                }
            # Nothing is sealed once the reservation is spent.
            if budget:
                budget.check()
        # The seal binds every finished invocation record, so it follows them.
        seal = {
            "schema_version": "1.0",
            "kind": "synthetic_inference_seal",
            "status": "predictions_sealed",
            "phase": INFERENCE,
            "identity": identity,
            "identity_sha256": json_sha256(identity),
            "source": source,
            "predictions": entries,
            "prediction_wall_s_total": wall_totals(entries),
            "invocations": record_hashes(read_invocations(root, INFERENCE, identity)),
            "timing_definitions": TIMING_DEFINITIONS,
            "limitation": "Predictions only. No metric, export or physics result exists until the analysis phase completes.",
        }
        seal["seal_sha256"] = json_sha256(seal)
        write_json(root / SEAL, seal)
        sync_directory(root)
        return seal


def checked_seal(root, identity, dataset, *, predictions=False):
    """Verify a published seal against the caller's identity and pinned source.

    Each prediction file is read and compared with its seal entry here when
    `predictions` is set; analysis instead does that as it consumes each one.
    """
    root = Path(root)
    seal = read_json(root / SEAL)
    views = len(dataset)
    expected = [
        (index, mode, prediction_name(index, mode))
        for index in range(views)
        for mode in MODES
    ]
    if (
        seal.get("seal_sha256")
        != json_sha256({k: v for k, v in seal.items() if k != "seal_sha256"})
        or (seal["kind"], seal["status"], seal["phase"])
        != ("synthetic_inference_seal", "predictions_sealed", INFERENCE)
        or seal["identity"] != identity
        or read_json(root / "identity.json") != identity
        or seal["identity_sha256"] != json_sha256(identity)
    ):
        raise ValueError("inference seal checksum or identity differs")
    source = seal["source"]
    if (
        source["descriptor_sha256"] != file_sha256(dataset.path)
        or source["manifest_sha256"] != dataset.manifest_sha256
        or source["identity"] != dataset.identity
        or source["split"] != dataset.manifest["split"]
        or source["counts"]["views"] != views
        or source["records_sha256"] != json_sha256(dataset.records)
    ):
        raise ValueError("inference seal covers a different source membership")
    entries = seal["predictions"]
    if (
        [(row["index"], row["mode"], row["file"]) for row in entries] != expected
        or members(root / "predictions") != {Path(name).name for *_, name in expected}
        or members(root) != INFERENCE_MEMBERS | {SEAL}
    ):
        raise ValueError("sealed inference is incomplete or holds undeclared files")
    if seal["prediction_wall_s_total"] != wall_totals(entries) or seal[
        "invocations"
    ] != record_hashes(read_invocations(root, INFERENCE, identity)):
        raise ValueError("inference seal timings or invocation records differ")
    if predictions:
        for row in entries:
            index, mode = row["index"], row["mode"]
            if (
                sealed_prediction(root, identity, index, dataset.records[index], mode)[
                    0
                ]
                != row
            ):
                raise ValueError(
                    f"sealed prediction {row['file']} differs from the seal"
                )
    return seal


class SealedJournal(PredictionJournal):
    """Analysis journal whose predictions come only from a sealed inference root."""

    def __init__(self, root, identity, sealed_root, seal, *, resume):
        super().__init__(root, identity, resume=resume)
        self.sealed_root = Path(sealed_root)
        self.seal = seal
        self.add = lambda name, seconds: None

    def prediction(self, index, record, arrays, mode, compute):
        if self.stop_requested:
            raise InterruptedError("evaluation warning received; resume the analysis")
        values, wall_s = self.sealed(index, record, arrays, mode)
        self.counts["predictions_reused"] += 1
        return values["xyz"], values["covered"], values["disagreement"], wall_s

    def sealed(self, index, record, arrays, mode):
        """Verify one sealed prediction against the seal and the pinned inputs."""
        started = time.perf_counter()
        entry = self.seal["predictions"][index * len(MODES) + MODES.index(mode)]
        derived, values = sealed_prediction(
            self.sealed_root, self.seal["identity"], index, record, mode
        )
        if derived != entry or entry["input_sha256"] != arrays_sha256(arrays):
            raise ValueError(
                f"sealed prediction {entry['file']} differs from the seal or the pinned inputs"
            )
        self.add("sealed_prediction_read_and_verify", time.perf_counter() - started)
        return values, entry["prediction_wall_s"]

    def physical_report(self, index, compute):
        def timed():
            started = time.perf_counter()
            try:
                return compute()
            finally:
                self.add("physics_fit", time.perf_counter() - started)

        return super().physical_report(index, timed)


def fit_report(numeric, metadata):
    """One independent physical fit with the evaluator's defaults."""
    from importlib import import_module

    return import_module("tennis3d.evaluation.physical").evaluate_physical(
        numeric, metadata=metadata
    )


def timed_fit(fit, numeric, metadata):
    """Worker task: (report, error, wall seconds, CPU seconds), also when it fails."""
    wall, cpu = time.perf_counter(), time.process_time()
    report = error = None
    try:
        report = fit(numeric, metadata)
    except Exception as failure:  # noqa: BLE001 - reported to the coordinator with its cost
        error = f"{type(failure).__name__}: {failure}"
    return report, error, time.perf_counter() - wall, time.process_time() - cpu


def start_worker():
    # Only the coordinator acts on the scheduler warning; workers finish their fit.
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def fit_ahead(journal, release, dataset, workers, fit, physics):
    """Fit uncommitted corrupted views in worker processes.

    Workers only compute. This process reads inputs, verifies each sealed
    prediction and commits reports to its journal in view order, holding at most
    two submitted views per worker. The serial scoring pass then finds every
    report committed and computes nothing itself. On a warning or a failed task,
    fits still queued are cancelled and those a worker already holds are
    committed when they finish. Every returned fit adds its wall and CPU seconds,
    failed or not; `physics` counts the fits whose worker died unmeasured.
    """
    from concurrent.futures import CancelledError, ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    from multiprocessing import get_context

    from tennis3d.evaluation.delivery import physical_fit_inputs
    from tennis3d.evaluation.synthetic import input_mode

    pending, failures, finished = {}, [], False

    def commit(index, future):
        try:
            report, error, wall_s, cpu_s = future.result()
        except CancelledError:
            return
        except Exception as death:  # noqa: BLE001 - any lost task stops the run
            # A dead worker returns no measurement for the fits it took with it.
            physics["fits_without_timing"] += 1
            failures.append((index, f"{type(death).__name__}: {death}"))
            return
        journal.add("physics_fit_worker_wall", wall_s)
        journal.add("physics_fit_worker_cpu", cpu_s)
        if error is not None:
            physics["failed_fits_timed"] += 1
            failures.append((index, error))
            return
        PredictionJournal.physical_report(journal, index, lambda: report)

    with ProcessPoolExecutor(
        workers, mp_context=get_context("spawn"), initializer=start_worker
    ) as pool:
        try:
            for index, record in enumerate(dataset.records):
                if journal.stop_requested or failures:
                    break
                if (journal.root / "physical-journal" / f"{index:05d}.json").exists():
                    continue
                arrays = dataset[index]
                values, _ = journal.sealed(index, record, arrays, "corrupted")
                numeric, metadata = physical_fit_inputs(
                    release,
                    dataset,
                    record,
                    arrays,
                    input_mode(arrays, "corrupted"),
                    values["xyz"],
                    values["covered"],
                )
                try:
                    pending[index] = pool.submit(timed_fit, fit, numeric, metadata)
                except BrokenProcessPool as death:
                    failures.append((index, f"{type(death).__name__}: {death}"))
                    break
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
    if failures:
        index, error = failures[0]
        raise ValueError(f"physical fit of view {index} failed in a worker: {error}")
    # A warning during the final drain must not reach the scoring pass.
    if journal.stop_requested or not finished:
        raise InterruptedError("evaluation warning received; resume the analysis")


def analyze(
    release_path,
    source_path,
    inference,
    output,
    *,
    resume=False,
    allocation_receipts=(),
    physics_workers=1,
):
    """Score a sealed inference root on CPU into the single-process result schema."""
    from tennis3d.evaluation.delivery import checked_release, require_final_source

    release, _, provenance = checked_release(release_path)
    require_final_source(source_path)
    return _analyze_released(
        release,
        provenance,
        source_path,
        inference,
        output,
        resume=resume,
        allocation_receipts=allocation_receipts,
        physics_workers=physics_workers,
    )


def _analyze_released(
    release,
    provenance,
    source_path,
    inference,
    output,
    *,
    resume=False,
    allocation_receipts=(),
    expected_counts=None,
    physics_workers=1,
    fit=fit_report,
):
    from importlib import import_module

    import torch

    from tennis3d.evaluation.delivery import (
        full_test_counts,
        score_synthetic,
        write_report,
    )
    from tennis3d.evaluation.synthetic import preflight
    from tennis3d.lifting.data import ViewDataset
    from tennis3d.lifting.runtime import environment
    from tennis3d.provenance import code_identity

    summarize_physical = import_module(
        "tennis3d.evaluation.physical_controls"
    ).summarize_physical
    process_started = time.perf_counter()
    # The scheduler's grant can be smaller than the CPUs the process may run on.
    declared = os.environ.get("SLURM_CPUS_PER_TASK", "")
    allocated = min(
        len(os.sched_getaffinity(0)),
        int(declared) if declared.isdigit() else MAX_PHYSICS_WORKERS,
    )
    if type(physics_workers) is not int or not 1 <= physics_workers <= min(
        MAX_PHYSICS_WORKERS, allocated
    ):
        raise ValueError(
            f"physics workers must be 1 to {MAX_PHYSICS_WORKERS} and no more than the allocated CPUs"
        )
    physics = {
        "workers": physics_workers,
        "outstanding_views_limit": 2 * physics_workers if physics_workers > 1 else 1,
        "worker_start_method": "spawn" if physics_workers > 1 else None,
        "cpu_affinity": len(os.sched_getaffinity(0)),
        "scheduler_cpus_per_task": int(declared) if declared.isdigit() else None,
        "failed_fits_timed": 0,
        "fits_without_timing": 0,
    }
    # The first two variables list allocated device indices; the last is a count.
    held = (
        os.environ.get("SLURM_JOB_GPUS")
        or os.environ.get("SLURM_STEP_GPUS")
        or os.environ.get("SLURM_GPUS_ON_NODE", "0") not in ("", "0")
    )
    if held or torch.cuda.is_initialized():
        raise ValueError(
            "analysis is CPU-only; release the GPU allocation before it starts"
        )
    inference, output = Path(inference).resolve(), Path(output).resolve()
    if (
        output == inference
        or inference in output.parents
        or output in inference.parents
    ):
        raise ValueError("analysis needs its own root outside the inference root")
    try:
        lock = (inference / "writer.lock").open("rb")
    except OSError as error:
        raise ValueError("inference root has no completed writer") from error
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("inference root still has an active writer") from error
        if not (inference / SEAL).is_file():
            raise ValueError(
                "inference is incomplete; no seal was published and no subset is scored"
            )
        sealed_identity = read_json(inference / "identity.json")
        if (
            sealed_identity.get("phase") != INFERENCE
            or sealed_identity.get("protocol") != PROTOCOL
            or sealed_identity["release_sha256"] != provenance["release_sha256"]
            or sealed_identity["checkpoint_sha256"] != release.checkpoint.sha256
            or sealed_identity["checkpoint_step"] != release.step
            or sealed_identity["source_descriptor_sha256"] != file_sha256(source_path)
            or sealed_identity["source_sha256"] != code_identity()["source_sha256"]
        ):
            raise ValueError(
                "sealed inference used a different phase, release, checkpoint, source or code"
            )
        receipts = [
            {
                "name": Path(path).name,
                "sha256": file_sha256(path),
                "bytes": Path(path).stat().st_size,
            }
            for path in allocation_receipts
        ]
        identity = {
            "phase": ANALYSIS,
            "phase_schema_version": "1.0",
            "environment": environment("cpu"),
            "device": "cpu",
            "inference_seal_sha256": file_sha256(inference / SEAL),
            "inference_identity_sha256": json_sha256(sealed_identity),
            "release_sha256": provenance["release_sha256"],
            "source_descriptor_sha256": file_sha256(source_path),
            "source_sha256": code_identity()["source_sha256"],
            "inference_allocation_receipts": receipts,
        }

        class TimedViews(ViewDataset):
            reader_s = 0.0

            def __getitem__(self, index):
                started = time.perf_counter()
                try:
                    return super().__getitem__(index)
                finally:
                    self.reader_s += time.perf_counter() - started

        with TimedViews(source_path) as dataset:
            started = time.perf_counter()
            full_test_counts(dataset, expected_counts)
            seal = checked_seal(inference, sealed_identity, dataset)
            if file_sha256(inference / SEAL) != identity["inference_seal_sha256"]:
                raise ValueError("inference seal changed during analysis")
            seal_verify_s = time.perf_counter() - started
            with SealedJournal(
                output, identity, inference, seal, resume=resume
            ) as journal:
                supersede_completion(journal.root, identity, seal)
                invocation = Invocation(
                    journal.root,
                    ANALYSIS,
                    identity,
                    "cpu",
                    process_started,
                    journal.counts,
                )
                # The durable start record already states the worker configuration.
                invocation.record["physics"] = physics
                with invocation:
                    journal.add = add = invocation.add
                    add("seal_verify", seal_verify_s)
                    started = time.perf_counter()
                    preflight(dataset, MODES)
                    add("preflight_including_reader", time.perf_counter() - started)
                    timings = invocation.timings
                    if physics_workers > 1:
                        started = time.perf_counter()
                        try:
                            fit_ahead(
                                journal, release, dataset, physics_workers, fit, physics
                            )
                        finally:
                            add("physics_worker_pass", time.perf_counter() - started)
                    before = (
                        dataset.reader_s,
                        timings.get("sealed_prediction_read_and_verify", 0.0),
                    )
                    started = time.perf_counter()
                    try:
                        metrics, exported, physical = score_synthetic(
                            None,
                            release,
                            dataset,
                            journal,
                            "cpu",
                            expected_counts=expected_counts,
                        )
                    finally:
                        add("scoring_pass", time.perf_counter() - started)
                        add("reader", dataset.reader_s)
                    add(
                        "scoring_metrics_export_and_journal",
                        timings["scoring_pass"]
                        - (dataset.reader_s - before[0])
                        - (timings["sealed_prediction_read_and_verify"] - before[1])
                        - timings.get("physics_fit", 0.0),
                    )
                    if journal.counts["predictions_reused"] != len(seal["predictions"]):
                        raise ValueError(
                            "analysis did not consume every sealed prediction"
                        )
                    started = time.perf_counter()
                    physical_summary = summarize_physical(
                        read_json(path) for path in physical
                    )
                    add("physical_summary", time.perf_counter() - started)
                    if (
                        file_sha256(release.checkpoint.path)
                        != release.checkpoint.sha256
                    ):
                        raise ValueError("checkpoint changed during final evaluation")
                    check_analysis_root(journal.root, len(dataset), complete=True)
                    inference_runs = read_invocations(
                        inference, INFERENCE, sealed_identity
                    )
                    analysis_runs = [*invocation.prior, invocation.snapshot()]
                    result = {
                        "schema_version": "1.0",
                        "kind": "model_only_synthetic",
                        "status": "measured",
                        "provenance": provenance,
                        "source_descriptor_sha256": file_sha256(source_path),
                        "execution_identity": journal.identity,
                        "prediction_journal": journal.counts,
                        "metrics": metrics,
                        "exported_result_metrics": exported.finish(),
                        "physical": physical_summary,
                        "runtime_s": {
                            "inference_prediction_wall_s_total": sum(
                                seal["prediction_wall_s_total"].values()
                            ),
                            "inference_invocations_recorded_wall_s": recorded_wall(
                                inference_runs
                            ),
                            "analysis_invocations_recorded_wall_s": recorded_wall(
                                analysis_runs
                            ),
                        },
                        "environment": environment("cpu"),
                        "phases": {
                            "timing_definitions": TIMING_DEFINITIONS,
                            "inference": {
                                "seal_file_sha256": identity["inference_seal_sha256"],
                                "seal_sha256": seal["seal_sha256"],
                                "identity": sealed_identity,
                                "source": seal["source"],
                                "prediction_count": len(seal["predictions"]),
                                "predictions_sha256": json_sha256(seal["predictions"]),
                                "prediction_wall_s_total": seal[
                                    "prediction_wall_s_total"
                                ],
                                "invocations": inference_runs,
                                "invocations_without_end_record": unfinished(
                                    inference_runs
                                ),
                                "scheduler_accounting_receipts": receipts,
                            },
                            "analysis": {
                                "identity_sha256": json_sha256(identity),
                                "cuda_initialized": torch.cuda.is_initialized(),
                                "invocations": analysis_runs,
                                "invocations_without_end_record": unfinished(
                                    invocation.prior
                                ),
                                "final_invocation": "last listed invocation; snapshot taken before this file was written",
                            },
                        },
                        "limitations": [
                            "Model-only inputs; no video detector",
                            "Clean and corrupted modes use frozen views; no new stress distribution generated",
                            "No independently measured full-training/convergence baseline or new stress comparison; old compact-model scores are not comparable evidence",
                            "Predictions come from a separate sealed inference phase. Prediction timing sums the retained original per-view durations; it is not GPU kernel time or utilization",
                            "Invocation wall times are process spans. Allocated GPU time is established only by scheduler accounting, and an invocation without an end record has no measured duration",
                            "Analysis, replay, reader and physics times are CPU-only and measured on the recorded analysis hardware",
                        ],
                    }
                    if torch.cuda.is_initialized():
                        raise ValueError(
                            "CUDA was initialized during CPU-only analysis"
                        )
                    write_json(journal.root / "metrics.json", result, overwrite=True)
                    write_report(journal.root, result)
                    with (journal.root / "report.md").open("a") as report:
                        report.write(
                            "\n".join(
                                [
                                    "Phases:",
                                    "",
                                    f"- Sealed inference `{identity['inference_seal_sha256']}` on {sealed_identity['device']}: {len(seal['predictions'])} predictions, {result['runtime_s']['inference_prediction_wall_s_total']:.3f} s summed original per-view prediction wall time.",
                                    f"- CPU-only analysis: {result['runtime_s']['analysis_invocations_recorded_wall_s']:.3f} s recorded invocation wall time.",
                                    "- Allocated GPU time comes from scheduler accounting, not from these process timings.",
                                    "",
                                ]
                            )
                        )
                write_json(
                    journal.root / RECEIPT,
                    completion_receipt(journal.root, identity, seal),
                )
                sync_directory(journal.root)
        return result


def check_analysis_root(root, views, *, complete):
    """Refuse files that this phase did not write, and gaps once it completes."""
    reports = {f"{index:05d}.json" for index in range(views)}
    for name in ("physical", "physical-journal"):
        found = members(root / name)
        if not found <= reports or (complete and found != reports):
            raise ValueError(f"analysis root holds undeclared or missing {name} files")
    if not members(root) <= ANALYSIS_MEMBERS:
        raise ValueError("analysis root holds files of another phase or command")


def completion_receipt(root, identity, seal):
    """Bind the finished analysis root's results and invocation records."""
    physical = sorted((root / "physical").glob("*.json"))
    return {
        "schema_version": "1.0",
        "kind": "synthetic_analysis_receipt",
        "status": "complete",
        "phase": ANALYSIS,
        "identity": identity,
        "identity_sha256": json_sha256(identity),
        "inference_seal_sha256": identity["inference_seal_sha256"],
        "counts": seal["source"]["counts"],
        "modes": list(MODES),
        "metrics_sha256": file_sha256(root / "metrics.json"),
        "report_sha256": file_sha256(root / "report.md"),
        "invocations": record_hashes(read_invocations(root, ANALYSIS, identity)),
        "physical_reports": len(physical),
        "physical_reports_sha256": json_sha256(
            {path.name: file_sha256(path) for path in physical}
        ),
    }


def supersede_completion(root, identity, seal):
    """Verify what an analysis root already holds before another invocation.

    A finished root must still match its receipt. The receipt and results are
    then withdrawn, because the new invocation makes them stale until it ends.
    """
    check_analysis_root(root, seal["source"]["counts"]["views"], complete=False)
    prior = read_invocations(root, ANALYSIS, identity)
    results = [root / name for name in (RECEIPT, "metrics.json", "report.md")]
    if results[0].exists():
        try:
            intact = read_json(results[0]) == completion_receipt(root, identity, seal)
        except (OSError, ValueError, KeyError, TypeError):
            intact = False
        if not intact:
            raise ValueError("analysis root differs from its completion receipt")
    elif not prior and members(root) - {"identity.json", "writer.lock"}:
        raise ValueError("analysis root holds results without invocation records")
    for path in results:
        path.unlink(missing_ok=True)
    sync_directory(root)


def recorded_wall(invocations):
    return sum(row["wall_s"] for row in invocations if row["wall_s"] is not None)


def unfinished(invocations):
    return sum(row["status"] == "running" for row in invocations)
