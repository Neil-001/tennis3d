"""Qualify split synthetic evaluation phases against the single-process evaluator.

Generates a controlled simulation, trains two fixture updates of each recipe to
obtain a real checkpoint, then scores the simulation's views once in a single
process and once as interrupted, resumed inference and analysis phases. It then
repeats the analysis with worker-process physics fitting: uninterrupted, stopped
by the scheduler warning, and with a failing worker task. The run fails unless
every path agrees exactly and incomplete inference is refused.

This is execution evidence on generated fixtures with untrained weights. It reads
no pinned test data or trained model, and its timings are not full-test costs.
"""

import argparse
import os
import shutil
import signal
import sys
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

# Disable accelerator visibility before importing torch or the package.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

from tennis3d.artifacts import file_sha256, json_sha256, read_json, write_json
from tennis3d.config import load_config
from tennis3d.evaluation import synthetic_phases as phases
from tennis3d.evaluation.delivery import _synthetic_released
from tennis3d.evaluation.prediction_journal import PredictionJournal
from tennis3d.lifting.checkpoint import read_checkpoint
from tennis3d.lifting.data import ViewDataset
from tennis3d.lifting.settings import TrainConfig
from tennis3d.lifting.train import train
from tennis3d.provenance import code_identity
from tennis3d.synthetic.config import SimulationConfig
from tennis3d.synthetic.generate import generate

ROOT = Path(__file__).resolve().parents[1]
RECIPES = {
    "transformer": "configs/stage4-cpu-smoke.yaml",
    "lstm": "configs/stage4-ablation-lstm.yaml",
}
TIMING_KEYS = ("full_view_prediction_wall_s", "prediction_wall_s_total")


def fixture_checkpoint(recipe, output):
    """Two fixture updates of the recipe's architecture and optimizer on CPU."""
    config = load_config(TrainConfig, ROOT / recipe)
    config = TrainConfig.model_validate(
        {
            **config.model_dump(),
            "train_manifest": str(ROOT / "tests/fixtures/lifting/train.manifest.json"),
            "val_manifest": str(ROOT / "tests/fixtures/lifting/val.manifest.json"),
            "batch_size": 1,
            "min_frames": 20,
            "max_frames": 32,
            "max_steps": 2,
            "max_train_seconds": 120.0,
            "max_job_seconds": 300.0,
            "validation_every": 1,
            "checkpoint_every": 1,
            "log_every": 1,
            "device": "cpu",
            "artifact_label": "Split-phase qualification; fresh random weights; two fixture updates; accuracy unmeasured",
        }
    )
    checkpoint = Path(train(config, output)["best_checkpoint"])
    state = read_checkpoint(checkpoint)
    digest = file_sha256(checkpoint)
    release = SimpleNamespace(
        checkpoint=SimpleNamespace(path=str(checkpoint), sha256=digest),
        step=state["step"],
    )
    provenance = {
        "release": {"checkpoint": {"sha256": digest}, "status": "fixture"},
        "release_sha256": json_sha256({"fixture_checkpoint": digest}),
    }
    return release, provenance, state["config"]["model"]


def warn_after(module, name, count):
    """Send this process the scheduler's SIGUSR1 warning after `count` calls."""
    original, calls = getattr(module, name), []

    def wrapped(*args, **kwargs):
        calls.append(1)
        if len(calls) == count:
            os.kill(os.getpid(), signal.SIGUSR1)
        return original(*args, **kwargs)

    setattr(module, name, wrapped)
    return lambda: setattr(module, name, original)


def interrupted(run, module, name, count):
    restore = warn_after(module, name, count)
    try:
        run()
    except InterruptedError:
        return
    finally:
        restore()
    raise RuntimeError("the controlled warning did not interrupt the phase")


def warning_fit(view, numeric, metadata):
    """Worker fit that sends the coordinator the scheduler warning at one view."""
    if metadata["artifact_id"] == view:
        os.kill(os.getppid(), signal.SIGUSR1)
    return phases.fit_report(numeric, metadata)


def failing_fit(view, numeric, metadata):
    if metadata["artifact_id"] == view:
        raise RuntimeError("controlled worker failure")
    return phases.fit_report(numeric, metadata)


def stopped(run, kind):
    try:
        run()
    except kind as error:
        return str(error)
    raise RuntimeError("the controlled worker stop did not end the analysis")


def invocations(root, phase):
    return phases.read_invocations(root, phase, read_json(root / "identity.json"))


def summary(rows):
    keys = ("status", "wall_s", "process_cpu_s", "timings_s", "counts")
    return [{key: row[key] for key in keys} for row in rows]


def refused(run, output):
    try:
        run()
    except ValueError as error:
        if output.exists():
            raise RuntimeError("a refused analysis left outputs behind") from error
        return str(error)
    raise RuntimeError("analysis accepted incomplete inference")


def without_timing(exported):
    return {
        mode: {k: v for k, v in values.items() if k not in TIMING_KEYS}
        for mode, values in exported["modes"].items()
    }


def files(root):
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def qualify(name, recipe, source, counts, artifacts, workers, output):
    import tennis3d.evaluation.physical as physical_module
    import tennis3d.inference.predict as predict_module

    release, provenance, model = fixture_checkpoint(recipe, output / "training")
    arguments = (release, provenance, source)
    started = time.perf_counter()
    with PredictionJournal(output / "single", {"fixture": name}) as journal:
        single = _synthetic_released(*arguments, journal, "cpu", expected_counts=counts)
    single_wall_s = time.perf_counter() - started

    inference, analysis = output / "inference", output / "analysis"

    def infer(**options):
        return phases._infer_released(
            *arguments, inference, "cpu", expected_counts=counts, **options
        )

    def analyze(sealed, target, **options):
        return phases._analyze_released(
            *arguments, sealed, target, expected_counts=counts, **options
        )

    interrupted(infer, predict_module, "predict_view", 5)
    before = files(inference / "predictions")
    early = refused(lambda: analyze(inference, output / "early"), output / "early")
    seal = infer(resume=True)
    sealed_files = files(inference)

    damaged = output / "missing-view"
    shutil.copytree(inference, damaged)
    (damaged / seal["predictions"][-1]["file"]).unlink()
    missing = refused(lambda: analyze(damaged, output / "partial"), output / "partial")

    interrupted(
        lambda: analyze(inference, analysis), physical_module, "evaluate_physical", 3
    )
    result = analyze(inference, analysis, resume=True)

    def same_predictions():
        for row in seal["predictions"]:
            _, _, split = phases.read_prediction(inference / row["file"])
            _, _, whole = phases.read_prediction(output / "single" / row["file"])
            if not all(
                np.array_equal(split[k], whole[k], equal_nan=True) for k in split
            ):
                return False
        return True

    reports = sorted(path.name for path in (analysis / "physical").glob("*.json"))
    inference_runs = invocations(inference, phases.INFERENCE)
    analysis_runs = invocations(analysis, phases.ANALYSIS)

    # Worker-process fitting: uninterrupted, warned at one view, and with one
    # failing task. Each must reproduce the serial analysis exactly.
    clean, warned, failed = (
        output / f"analysis-workers{suffix}" for suffix in ("", "-warned", "-failed")
    )
    parallel = {"clean": analyze(inference, clean, physics_workers=workers)}
    stopped(
        lambda: analyze(
            inference,
            warned,
            physics_workers=workers,
            fit=partial(warning_fit, artifacts[2]),
        ),
        InterruptedError,
    )
    warned_committed = len(files(warned / "physical-journal"))
    warned_scored = (warned / "metrics.json").exists() or (warned / "physical").exists()
    parallel["warned"] = analyze(
        inference, warned, resume=True, physics_workers=workers
    )
    failure = stopped(
        lambda: analyze(
            inference,
            failed,
            physics_workers=workers,
            fit=partial(failing_fit, artifacts[5]),
        ),
        ValueError,
    )
    failed_committed = files(failed / "physical-journal")
    failed_scored = (failed / "metrics.json").exists()
    parallel["failed"] = analyze(
        inference, failed, resume=True, physics_workers=workers
    )
    roots = {"clean": clean, "warned": warned, "failed": failed}
    runs = {key: invocations(root, phases.ANALYSIS) for key, root in roots.items()}
    clean_run = runs["clean"][0]
    worker_checks = {
        "worker_reports_byte_equal_serial": all(
            files(root / "physical") == files(analysis / "physical")
            for root in roots.values()
        ),
        "worker_metrics_equal_serial": all(
            value["metrics"] == result["metrics"]
            and value["physical"] == result["physical"]
            and without_timing(value["exported_result_metrics"])
            == without_timing(result["exported_result_metrics"])
            for value in parallel.values()
        ),
        "worker_original_prediction_timing_reported": all(
            value["exported_result_metrics"]["modes"][mode]["prediction_wall_s_total"]
            == seal["prediction_wall_s_total"][mode]
            for value in parallel.values()
            for mode in phases.MODES
        ),
        "worker_run_fitted_every_view_once": clean_run["counts"]["physical_created"]
        == counts["views"]
        and clean_run["physics"]["workers"] == workers
        and "physics_fit" not in clean_run["timings_s"],
        "worker_warning_kept_fits_and_resumed": not warned_scored
        and [row["status"] for row in runs["warned"]] == ["interrupted", "complete"]
        and runs["warned"][0]["counts"]["physical_created"] == warned_committed
        and runs["warned"][1]["counts"]["physical_created"]
        == counts["views"] - warned_committed,
        "worker_failure_refused_and_resumed": "failed in a worker" in failure
        and not failed_scored
        and "00005.json" not in failed_committed
        and [row["status"] for row in runs["failed"]] == ["failed", "complete"]
        and runs["failed"][1]["counts"]["physical_created"]
        == counts["views"] - len(failed_committed),
        "worker_analysis_without_cuda": not torch.cuda.is_initialized()
        and all(
            row["cuda_initialized"] is False for rows in runs.values() for row in rows
        ),
        "worker_analysis_left_sealed_root_unchanged": files(inference) == sealed_files,
    }
    worker_cpu_s = clean_run["timings_s"]["physics_fit_worker_cpu"]
    checks = {
        "result_schema": (result["schema_version"], result["kind"], result["status"])
        == ("1.0", "model_only_synthetic", "measured"),
        "unconditional_metrics_equal": result["metrics"] == single["metrics"],
        "exported_metrics_equal_except_timing": without_timing(
            result["exported_result_metrics"]
        )
        == without_timing(single["exported_result_metrics"]),
        "physical_summary_equal": result["physical"] == single["physical"],
        "all_physical_reports_byte_equal": reports
        == [f"{index:05d}.json" for index in range(counts["views"])]
        and all(
            (analysis / "physical" / report).read_bytes()
            == (output / "single/physical" / report).read_bytes()
            for report in reports
        ),
        "all_predictions_bit_equal": len(seal["predictions"]) == 2 * counts["views"]
        and same_predictions(),
        "original_prediction_timing_reported": all(
            result["exported_result_metrics"]["modes"][mode]["prediction_wall_s_total"]
            == seal["prediction_wall_s_total"][mode]
            for mode in phases.MODES
        ),
        "interrupted_predictions_kept_their_bytes": len(before) == 5
        and before.items() <= files(inference / "predictions").items(),
        "inference_resumed": [row["status"] for row in inference_runs]
        == ["interrupted", "complete"]
        and inference_runs[1]["counts"]["predictions_created"]
        == 2 * counts["views"] - 5,
        "analysis_resumed": [row["status"] for row in analysis_runs]
        == ["interrupted", "complete"]
        and analysis_runs[0]["counts"]["physical_created"]
        + analysis_runs[1]["counts"]["physical_created"]
        == counts["views"],
        "analysis_created_no_prediction": all(
            row["counts"]["predictions_created"] == 0 for row in analysis_runs
        )
        and not (analysis / "predictions").exists(),
        "analysis_left_sealed_root_unchanged": files(inference) == sealed_files,
        "analysis_without_cuda": not torch.cuda.is_initialized()
        and result["phases"]["analysis"]["cuda_initialized"] is False,
        "unsealed_inference_refused": "incomplete" in early,
        "missing_view_refused": "incomplete" in missing,
        **worker_checks,
    }
    return {
        "recipe": recipe,
        "model": model,
        "checkpoint_sha256": release.checkpoint.sha256,
        "checks": checks,
        "refusals": {
            "unsealed": early,
            "missing_view": missing,
            "worker_failure": failure,
        },
        "seal_file_sha256": file_sha256(inference / phases.SEAL),
        "metrics_sha256": {
            "single": json_sha256(single["metrics"]),
            "split": json_sha256(result["metrics"]),
        },
        "fixture_timing_s": {
            "single_process_wall": single_wall_s,
            "single_process_prediction_total": sum(
                values["prediction_wall_s_total"]
                for values in single["exported_result_metrics"]["modes"].values()
            ),
            "inference_invocations": summary(inference_runs),
            "analysis_invocations": summary(analysis_runs),
            "worker_analysis": {
                "note": "Measured on this fixture and hardware only. Not a full-test speedup.",
                "workers": workers,
                "physics": clean_run["physics"],
                "uninterrupted_invocation": summary([clean_run])[0],
                "serial_analysis_wall": sum(row["wall_s"] for row in analysis_runs),
                "serial_analysis_process_cpu": sum(
                    row["process_cpu_s"] for row in analysis_runs
                ),
                "worker_fit_cpu": worker_cpu_s,
                "cpu_seconds_per_wall_second": (
                    worker_cpu_s + clean_run["process_cpu_s"]
                )
                / clean_run["wall_s"],
                "warned_invocations": summary(runs["warned"]),
                "failed_invocations": summary(runs["failed"]),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--recipe", action="append", choices=sorted(RECIPES), dest="recipes"
    )
    parser.add_argument("--rallies", type=int, default=6)
    parser.add_argument("--physics-workers", type=int, default=4, choices=range(2, 9))
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("--output must be outside the repository")
    if output.exists():
        parser.error("--output must be a new, nonexistent directory")
    if args.rallies < 3:
        parser.error("--rallies must be at least 3 for the controlled worker stops")
    os.chdir(ROOT)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True)
    simulation = SimulationConfig(
        seed=20261008,
        train_rallies=1,
        val_rallies=1,
        test_rallies=args.rallies,
        pool_size=24,
        views_per_rally=2,
    )
    generate(output / "simulation", simulation)
    source = output / "simulation/test.manifest.json"
    with ViewDataset(source) as dataset:
        frames = [row["frames"] for row in dataset.records]
        artifacts = [
            f"{row['rally_id']}-view-{row['view_id']}" for row in dataset.records
        ]
        counts = {
            "rallies": len({row["rally_id"] for row in dataset.records}),
            "views": len(dataset),
            "frames": sum(frames),
        }
    report = {
        "schema_version": "1.0",
        "kind": "synthetic_phase_qualification",
        "claim": "Execution equivalence on a generated fixture with untrained weights. No accuracy, pinned-test or full-cost result.",
        "code": {
            key: code_identity()[key] for key in ("code_revision", "source_sha256")
        },
        "hardware": phases.hardware("cpu"),
        "torch_threads": torch.get_num_threads(),
        "simulation": simulation.model_dump(mode="json"),
        "source_manifest_sha256": file_sha256(source),
        "counts": counts,
        "view_frames": frames,
        "multi_window_views": sum(length > 250 for length in frames),
        "recipes": {},
    }
    for name in args.recipes or sorted(RECIPES):
        report["recipes"][name] = qualify(
            name,
            RECIPES[name],
            source,
            counts,
            artifacts,
            args.physics_workers,
            output / name,
        )
    report["passed"] = all(
        all(values["checks"].values()) for values in report["recipes"].values()
    )
    write_json(output / "qualification.json", report)
    print(read_json(output / "qualification.json")["passed"])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
