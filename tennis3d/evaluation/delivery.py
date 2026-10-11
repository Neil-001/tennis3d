"""Reproducible Stage 4 evaluation, gated by an explicit selected-best receipt."""

import argparse
import html
import json
import math
import os
import shutil
import time
from importlib import import_module
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from tennis3d.artifacts import _loads, file_sha256, json_sha256, read_json, write_json
from tennis3d.evaluation.inventory import read_inventory, verify_clip
from tennis3d.schemas import SHA256, Observation, StrictModel

MAIN_MANIFEST = "b32fe9cc97226c9d67468bcf7724f7d3ffc8924541798796744f9ba62bba6707"
MAIN_PIN = "c2e782053694cfde305362e670ebbb4f4e70884c"


class EvidenceFile(StrictModel):
    path: str
    sha256: SHA256

    def read(self):
        path = Path(self.path)
        if file_sha256(path) != self.sha256:
            raise ValueError(f"release evidence changed: {path.name}")
        return path


class CheckpointRelease(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    status: Literal["selected_best_released"]
    released_by: str = Field(min_length=1)
    released_utc: str
    checkpoint: EvidenceFile
    run_id: str
    step: int = Field(ge=1)
    selection: Literal["minimum_reserved_validation_corrupted_rally_mean"]
    validation_history: EvidenceFile
    final_budget: EvidenceFile
    exposure_receipt: EvidenceFile
    publication: Literal["pending", "authorized"] = "pending"


def checked_release(path):
    """Verify validation-only selection and actual final compute/exposure evidence."""
    from tennis3d.lifting.checkpoint import read_checkpoint

    release = CheckpointRelease.model_validate(read_json(path))
    checkpoint = release.checkpoint.read()
    state = read_checkpoint(checkpoint)
    rows = [
        read_json_line(line)
        for line in release.validation_history.read().read_text().splitlines()
        if line.strip()
    ]
    if not rows or any(row["run_id"] != release.run_id for row in rows):
        raise ValueError("validation history must belong to the released run")
    for row in rows:
        if (
            row["source_identity"] != state["dataset_identity"]["val"]
            or row["manifest_sha256"] != state["manifest_sha256"]["val"]
        ):
            raise ValueError(
                "every history row must use the released reserved-validation identity"
            )
        if any(
            type(row[key]) is not int or row[key] < 0
            for key in ("step", "sampled_views")
        ):
            raise ValueError(
                "validation history needs actual nonnegative step/view counters"
            )
        if any(
            type(row[key]) not in (int, float)
            or not math.isfinite(row[key])
            or row[key] < 0
            for key in ("train_seconds", "job_seconds")
        ):
            raise ValueError("validation history needs finite charged compute")
    scores = [row["modes"]["corrupted"]["val_xyz_m"] for row in rows]
    if any(
        type(score) not in (float, int) or not math.isfinite(score) for score in scores
    ):
        raise ValueError("validation selection needs finite recorded scores")
    chosen = rows[min(range(len(rows)), key=lambda index: scores[index])]
    if (
        state["run_id"] != release.run_id
        or state["step"] != release.step
        or chosen["step"] != release.step
        or state["best_step"] != release.step
        or state["best_score"] != chosen["modes"]["corrupted"]["val_xyz_m"]
        or chosen["source_identity"] != state["dataset_identity"]["val"]
        or chosen["manifest_sha256"] != state["manifest_sha256"]["val"]
    ):
        raise ValueError("checkpoint is not the validation-selected best of this run")
    budget = read_json(release.final_budget.read())
    exposure = read_json(release.exposure_receipt.read())
    if budget["run_id"] != release.run_id or exposure["run_id"] != release.run_id:
        raise ValueError("final compute/exposure evidence belongs to another run")
    if (
        any(
            type(exposure[key]) is not int or exposure[key] < 0
            for key in ("final_step", "sampled_views")
        )
        or type(budget["step"]) is not int
    ):
        raise ValueError("final compute/exposure needs integer step/view counters")
    if budget["step"] != exposure["final_step"]:
        raise ValueError("final budget step and exposure final step differ")
    # Repeated/replayed validation steps are legitimate after recovery. Compare
    # maxima instead of requiring monotonically increasing optimizer steps.
    if exposure["final_step"] < max(
        release.step, *(row["step"] for row in rows)
    ) or exposure["sampled_views"] < max(
        state["sampler"]["draws"], *(row["sampled_views"] for row in rows)
    ):
        raise ValueError(
            "final exposure cannot precede any recorded validation or selected checkpoint"
        )
    if any(
        type(budget[key]) not in (float, int)
        or not math.isfinite(budget[key])
        or budget[key] < max(state["budget"][key], *(row[key] for row in rows))
        for key in ("train_seconds", "job_seconds")
    ):
        raise ValueError(
            "final charged compute cannot precede any recorded validation or selected checkpoint"
        )
    return (
        release,
        state,
        {
            "release": release.model_dump(),
            "release_sha256": file_sha256(path),
            "checkpoint_training_provenance": state["provenance"],
            "checkpoint_config": state["config"],
            "dataset_identity": state["dataset_identity"],
            "selected_checkpoint_budget": state["budget"],
            "selected_checkpoint_sampler": {
                key: state["sampler"][key]
                for key in ("draws", "rejected", "epoch", "cursor")
            },
            "actual_final_budget": budget,
            "actual_final_exposure": exposure,
            "weights": "strictly loaded EMA module weights; random initialization recorded by training provenance",
        },
    )


def read_json_line(line):
    # Same strict JSON decoder as artifact files, without a temporary file.
    from tennis3d.artifacts import _loads

    return _loads(line)


def plan(pins_path, inventory_path, output):
    """Create metadata-only descriptors. This function never opens data arrays."""
    from tennis3d.datasets.source import SourceSpec

    pins = read_json(pins_path)
    main = pins["main"]
    if pins["status"] != "verified-release" or main["commit"] != MAIN_PIN:
        raise ValueError("final evaluation requires the audited immutable main release")
    output = fresh_output(output)
    spec = SourceSpec.model_validate(
        {
            "manifest": main["manifest_artifact"],
            "split": "test",
            "repo_id": pins["repo_id"],
            "revision": MAIN_PIN,
            "cache_dir": "test-cache",
            "cache_bytes": main["reader_cache_bytes"],
            "max_group_bytes": main["max_group_bytes"],
        }
    )
    write_json(output / "test.source.json", spec.model_dump())
    inventory = read_inventory(inventory_path)
    result = {
        "schema_version": "1.0",
        "status": "awaiting_selected_checkpoint_release",
        "test_arrays_opened": False,
        "pins_sha256": file_sha256(pins_path),
        "source_descriptor_sha256": file_sha256(output / "test.source.json"),
        "expected_test_counts": main["counts"]["test"],
        "reader": "ViewDataset/V2Source, immutable membership/order identity, checksummed bounded row groups",
        "reader_cache_bytes": spec.cache_bytes,
        "max_group_bytes": spec.max_group_bytes,
        "modes": ["corrupted", "clean"],
        "prediction": {
            "max_frames": 250,
            "stride": 125,
            "merge": "centre weights",
            "coverage": "every label_valid AND camera_valid frame before export filtering",
        },
        "physics": "merged physical evaluator on corrupted predictions, truth used only for discrimination controls",
        "inventory_sha256": file_sha256(inventory_path),
        "real_inventory": inventory.summary(),
        "dependencies": [
            "selected-best release receipt",
            "merged physical evaluator",
            "eligible audited source videos",
            "normalized labels for tracking/landing accuracy",
        ],
        "constraints": [
            "No tuning on test or real outputs",
            "No new allocation while full training runs",
            "Owned cache never production cache",
            "Unavailable labels or independent real XYZ stay unavailable",
        ],
    }
    write_json(output / "plan.json", result)
    return result


def fresh_output(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("output directory already exists; choose a new run")
    output.mkdir(parents=True)
    return output


def require_final_source(source_path):
    """Reject anything but the complete pinned main test membership."""
    spec = read_json(source_path)
    if (
        spec.get("revision") != MAIN_PIN
        or spec.get("split") != "test"
        or spec.get("exclude_rally_ids")
        or spec.get("subset_sha256")
        or spec.get("manifest", {}).get("sha256") != MAIN_MANIFEST
        or spec.get("repo_id") != "N-il/tennis-3d-synthetic"
        or spec.get("cache_bytes", 1 << 31) > 1 << 30
        or spec.get("max_group_bytes", 1 << 31) > 1 << 28
    ):
        raise ValueError(
            "final synthetic evaluation requires complete mainpin test membership"
        )


def synthetic(release_path, source_path, output, *, device="cpu", resume=False):
    release, _, provenance = checked_release(release_path)
    require_final_source(source_path)
    from tennis3d.evaluation.prediction_journal import PredictionJournal
    from tennis3d.lifting.runtime import environment, resolve_device
    from tennis3d.provenance import code_identity

    device = resolve_device(device)
    identity = {
        "environment": environment(device),
        "release_sha256": provenance["release_sha256"],
        "source_descriptor_sha256": file_sha256(source_path),
        "source_sha256": code_identity()["source_sha256"],
        "device": device,
    }
    with PredictionJournal(output, identity, resume=resume) as journal:
        return _synthetic_released(release, provenance, source_path, journal, device)


def _synthetic_released(
    release,
    provenance,
    source_path,
    journal,
    device,
    *,
    expected_counts=None,
):
    summarize_physical = import_module(
        "tennis3d.evaluation.physical_controls"
    ).summarize_physical
    from tennis3d.lifting.checkpoint import load_lifter
    from tennis3d.lifting.data import ViewDataset
    from tennis3d.lifting.runtime import environment, resolve_device, synchronize

    output = journal.root
    device = resolve_device(device)
    model, _ = load_lifter(release.checkpoint.path, device)
    started = time.perf_counter()
    with ViewDataset(source_path) as dataset:
        metrics, exported, physical = score_synthetic(
            model, release, dataset, journal, device, expected_counts=expected_counts
        )
    synchronize(device)
    if file_sha256(release.checkpoint.path) != release.checkpoint.sha256:
        raise ValueError("checkpoint changed during final evaluation")
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
        "physical": summarize_physical(read_json(path) for path in physical),
        "runtime_s": {
            "full_reader_preflight_predictions_metrics_physics": time.perf_counter()
            - started
        },
        "environment": environment(device),
        "limitations": [
            "Model-only inputs; no video detector",
            "Clean and corrupted modes use frozen views; no new stress distribution generated",
            "No independently measured full-training/convergence baseline or new stress comparison; old compact-model scores are not comparable evidence",
            "Prediction timing sums retained original per-view inference durations; resumed invocation wall time also includes replayed reader/preflight/scoring work",
        ],
    }
    write_json(output / "metrics.json", result, overwrite=True)
    write_report(output, result)
    return result


def full_test_counts(dataset, expected_counts=None):
    counts = {
        "rallies": len({row["rally_id"] for row in dataset.records}),
        "views": len(dataset),
        "frames": sum(row["frames"] for row in dataset.records),
    }
    if counts != (
        expected_counts or {"rallies": 5000, "views": 10000, "frames": 1829263}
    ):
        raise ValueError("full test membership differs from the immutable release")
    return counts


def physical_fit_inputs(release, dataset, record, arrays, inputs, xyz, covered):
    """Numeric arrays and metadata of one corrupted view's physical evaluation."""
    numeric = {
        "t_s": arrays["t_s"],
        "xyz_m": xyz,
        "available": covered,
        "input_observed": inputs["ball_observed"],
        "truth_xyz_m": arrays["xyz_m"],
        "label_valid": arrays["label_valid"] & arrays["camera_valid"],
        **{
            key: arrays[key]
            for key in ("event_t_s", "event_code", "event_end_t_s")
            if key in arrays
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


def score_synthetic(model, release, dataset, journal, device, *, expected_counts=None):
    """Score both frozen modes of an open dataset through the journal.

    Returns the unconditional metrics, the unfinished exported-result metrics
    and the ordered per-view physical report paths. Shared by the single-process
    command and the CPU analysis phase so their scoring cannot diverge.
    """
    evaluate_physical = import_module("tennis3d.evaluation.physical").evaluate_physical
    from tennis3d.evaluation.synthetic import evaluate_model

    output = journal.root
    physical = []
    exported = ExportedMetrics()

    def on_view(record, arrays, inputs, xyz, covered, mode, diagnostics):
        exported.add(arrays, inputs, xyz, covered, mode, diagnostics)
        if mode != "corrupted":
            return
        numeric, metadata = physical_fit_inputs(
            release, dataset, record, arrays, inputs, xyz, covered
        )
        report = journal.physical_report(
            len(physical), lambda: evaluate_physical(numeric, metadata=metadata)
        )
        # Numeric ordinal filenames avoid interpreting dataset identity as paths.
        path = output / "physical" / f"{len(physical):05d}.json"
        if path.exists():
            if read_json(path) != report:
                raise ValueError("published physical report differs from journal")
        else:
            write_json(path, report)
        physical.append(path)

    dataset.verify_identity()
    full_test_counts(dataset, expected_counts)
    metrics = evaluate_model(
        model,
        dataset,
        device=device,
        modes=("corrupted", "clean"),
        on_view=on_view,
        prediction_provider=journal.prediction,
    )
    dataset.verify_identity()
    return metrics, exported, physical


class ExportedMetrics:
    """Score fixed production quality rules with retained supported denominators."""

    def __init__(self):
        from collections import Counter

        from tennis3d.inference.quality import QualityConfig

        self.config = QualityConfig()
        self.modes: dict[str, dict] = {
            mode: {
                "errors": [],
                "supported": 0,
                "source": 0,
                "reasons": Counter(),
                "view_wall_s": [],
            }
            for mode in ("corrupted", "clean")
        }

    def add(self, arrays, inputs, xyz, covered, mode, diagnostics):
        from tennis3d.inference.quality import quality_reasons

        observation: list[Observation] = [
            "detected" if value else "missing" for value in inputs["ball_observed"]
        ]
        camera_xyz = (
            np.einsum("tij,tj->ti", arrays["camera_R"], xyz) + arrays["camera_t"]
        )
        reasons = quality_reasons(
            arrays["t_s"],
            observation,
            arrays["camera_valid"],
            xyz,
            covered,
            diagnostics["disagreement_m"],
            self.config,
            positive_depth=camera_xyz[:, 2] > 0,
        )
        expected = arrays["label_valid"] & arrays["camera_valid"]
        available = expected & np.array([not row for row in reasons])
        values = self.modes[mode]
        values["errors"].append(xyz[available] - arrays["xyz_m"][available])
        values["supported"] += int(expected.sum())
        values["source"] += len(expected)
        values["view_wall_s"].append(diagnostics["prediction_wall_s"])
        values["reasons"].update(reason for row in reasons for reason in row)

    def finish(self):
        from tennis3d.evaluation.metrics import summarize
        from tennis3d.evaluation.real import error_summary

        modes = {}
        for mode, values in self.modes.items():
            summary = summarize(np.concatenate(values["errors"]))
            modes[mode] = {
                **summary,
                "supported_frames": values["supported"],
                "source_frames": values["source"],
                "supported_coverage": summary["count"] / values["supported"],
                "source_coverage": summary["count"] / values["source"],
                "reason_counts_all_source_rows": dict(values["reasons"]),
                "full_view_prediction_wall_s": error_summary(values["view_wall_s"]),
                "prediction_wall_s_total": sum(values["view_wall_s"]),
            }
        return {
            "quality_config": self.config.model_dump(),
            "definitions": "Errors conditional on exported availability; coverage uses fixed label_valid AND camera_valid and all-source denominators. Unconditional results remain separate. Synthetic observations have only detected/missing states.",
            "modes": modes,
        }


def real(release_path, inventory_path, config_path, output, *, device="cpu"):
    from tennis3d.config import load_config

    summarize_physical = import_module(
        "tennis3d.evaluation.physical_controls"
    ).summarize_physical
    from tennis3d.inference.settings import ReconstructionConfig

    release, _, provenance = checked_release(release_path)
    inventory = read_inventory(inventory_path)
    config = load_config(
        ReconstructionConfig,
        config_path,
        overrides={
            "detection": {"device": device},
            "lifting": {
                "device": device,
                "checkpoint": release.checkpoint.path,
                "model_label": "Final selected EMA; dense real XYZ accuracy unmeasured",
            },
        },
    )
    output = fresh_output(output)
    write_json(output / "inventory.json", inventory.model_dump())
    write_json(output / "config.json", config.model_dump(mode="json"))
    records, physics = [], []
    for clip in inventory.clips:
        record = {
            "clip_id": clip.clip_id,
            "recording_group": clip.recording_group,
            "source": clip.model_dump(),
        }
        if not clip.eligible:
            record.update(status="ineligible", failure_reasons=clip.barriers)
        else:
            try:
                diagnostics, fit = evaluate_real_clip(
                    clip, config, output / "clips" / clip.clip_id
                )
                record.update(diagnostics)
                if fit is not None:
                    physics.append(fit)
                if diagnostics.get("labelled_lifting"):
                    physics.append(
                        read_json(
                            output / diagnostics["labelled_lifting"]["physical_path"]
                        )
                    )
            except (ValueError, OSError, RuntimeError, ImportError) as error:
                record.update(
                    status="failed",
                    failure_reasons=[f"{type(error).__name__}: {error}"],
                )
        records.append(record)
        write_json(
            output / "progress.json",
            {
                "processed": len(records),
                "preselected": len(inventory.clips),
                "records": records,
            },
            overwrite=True,
        )
    result = {
        "schema_version": "1.0",
        "kind": "end_to_end_real",
        "status": (
            "incomplete_inventory"
            if not inventory.summary()["required_gallery_ready"]
            else "measured"
            if any(record.get("result_path") for record in records)
            else "no_results"
        ),
        "inventory_sha256": file_sha256(inventory_path),
        "inventory": inventory.summary(),
        "provenance": provenance,
        "config_sha256": json_sha256(config.model_dump(mode="json")),
        "records": records,
        "physical": summarize_physical(physics) if physics else None,
        "limitations": inventory.limitations
        + [
            "Dense real XYZ truth unavailable",
            "Unknown detector training overlap prevents held-out detector claims",
            "Detection-reference reprojection shares detector errors",
            "Physical consistency is not independent XYZ truth",
        ],
    }
    if file_sha256(release.checkpoint.path) != release.checkpoint.sha256:
        raise ValueError("checkpoint changed during real evaluation")
    write_json(output / "metrics.json", result)
    write_report(output, result)
    write_gallery(output, result)
    return result


def evaluate_real_clip(clip, config, run):
    from tennis3d.evaluation.annotations import AnnotationManifest
    from tennis3d.evaluation.landing import landing_metrics

    evaluate_physical = import_module("tennis3d.evaluation.physical").evaluate_physical
    from tennis3d.evaluation.real import pixel_metrics
    from tennis3d.geometry.camera import Camera, ImageSize
    from tennis3d.inference.inputs import camera_inputs, detection_inputs
    from tennis3d.inference.lift import prediction_inputs
    from tennis3d.inference.pipeline import reconstruct
    from tennis3d.schemas import read_result

    timeline = verify_clip(clip)
    if clip.source_frame_start != 0:
        raise ValueError(
            "runner requires a preselected complete clip file, not an implicit source crop"
        )
    assert (
        clip.source_path is not None
        and clip.width is not None
        and clip.height is not None
    )
    summary = reconstruct(Path(clip.source_path), run, config)
    if not (run / "result.json").is_file():
        return {
            "status": "failed",
            "pipeline": summary,
            "failure_reasons": summary.get("failure_reasons", []),
        }, None
    result = read_result(run / "result.json")
    detection, frames, track = detection_inputs(run)
    calibration, cameras = camera_inputs(run, detection, frames)
    lifting, raw = prediction_inputs(run, detection, frames)
    from fractions import Fraction

    pts = [
        Fraction(row["pts"] * row["time_base"][0], row["time_base"][1])
        for row in timeline["frames"]
    ]
    expected = np.array([float(value - pts[0]) for value in pts])
    if (
        result.source.sha256 != clip.source_sha256
        or result.source.frame_count != clip.frame_count
        or (result.source.width, result.source.height) != (clip.width, clip.height)
        or not np.allclose(frames["t_s"], expected, atol=1e-9, rtol=0)
        or not np.array_equal(frames["pts"], [row["pts"] for row in timeline["frames"]])
        or not np.array_equal(
            frames["time_base"], [row["time_base"] for row in timeline["frames"]]
        )
    ):
        raise ValueError(
            "pipeline does not preserve the frozen source identity/timeline"
        )
    labels = (
        AnnotationManifest.model_validate_json(Path(clip.labels_path).read_text())
        if clip.labels_path
        else None
    )
    camera_objects = [
        Camera(
            cameras["K"][i],
            cameras["R"][i],
            cameras["t"][i],
            ImageSize(clip.width, clip.height),
        )
        if valid
        else None
        for i, valid in enumerate(cameras["camera_valid"])
    ]
    metadata = {
        "artifact_id": clip.clip_id,
        "mode": "end_to_end",
        "domain": "real",
        "split": "test",
        "time_basis": "source_pts",
        "timing_note": clip.timing_note,
        "source_identity": {
            "source_sha256": clip.source_sha256,
            "width": clip.width,
            "height": clip.height,
            "frame_count": clip.frame_count,
            "recording_group": clip.recording_group,
            "rights": clip.rights_evidence,
        },
        "prediction_identity": {
            "checkpoint_sha256": lifting["checkpoint_sha256"],
            "weights": "EMA",
            "filtering": "raw predictions before export quality rules",
        },
        "segmentation_source": "annotated" if labels else "automatic",
        "segmentation_method": "contiguous same-event labelled source indices"
        if labels
        else "velocity_jump_v1",
        "event_grouping": "contiguous same-event source indices"
        if labels
        else "prediction-derived velocity jumps; engineering thresholds fixed before outputs; no contact truth",
        "image_reference": {
            "kind": "annotations" if labels else "detections",
            "identity": labels.labels_sha256
            if labels
            else detection["artifacts"]["ball_track.npz"],
            "dependence": "independent manual annotation"
            if labels
            else "same input detector; dependent consistency measure",
        },
        "camera_source": calibration["mode"],
        "bounce_convention": "ball_center",
    }
    numeric = {
        "t_s": frames["t_s"],
        "xyz_m": raw["xyz_m"],
        "available": raw["covered"],
        "input_observed": track["ball_observed"],
        **(
            {
                "reference_uv_px": track["ball_uv"],
                "reference_valid": track["ball_observed"],
            }
            if labels is None
            else {}
        ),
        "camera_K": cameras["K"],
        "camera_R": cameras["R"],
        "camera_t": cameras["t"],
        "camera_valid": cameras["camera_valid"],
    }
    fit = evaluate_physical(
        numeric, metadata=metadata, labels=labels, cameras=camera_objects
    )
    write_json(run / "physical.json", fit)
    record = {
        "status": result.status,
        "failure_reasons": list(
            dict.fromkeys(
                [
                    *summary.get("failure_reasons", []),
                    *result.failure_reasons,
                    *read_json(run / "metrics.json").get("failure_reasons", {}),
                ]
            )
        ),
        "pipeline": summary,
        "execution": read_json(run / "metrics.json"),
        "physical_path": f"clips/{clip.clip_id}/physical.json",
        "result_path": f"clips/{clip.clip_id}/result.json",
        "tracking": None,
        "landing": None,
        "labelled_lifting": None,
    }
    if labels:
        if (labels.clip_id, labels.recording_group, labels.split) != (
            clip.clip_id,
            clip.recording_group,
            "test",
        ):
            raise ValueError(
                "annotation split/clip/recording differs from frozen inventory"
            )
        record["tracking"] = pixel_metrics(
            labels,
            track["ball_uv"],
            track["ball_observed"],
            source_sha256=clip.source_sha256,
            image_size=(clip.width, clip.height),
            tolerances_px=[5.0, 10.0, 20.0],
            t_s=frames["t_s"],
        )
        exported = np.array(
            [
                row.xyz_m if row.xyz_m is not None else [np.nan] * 3
                for row in result.frames
            ]
        )
        available = np.array([row.xyz_m is not None for row in result.frames])
        record["landing"] = landing_metrics(
            labels,
            frames["t_s"],
            exported,
            available,
            camera_objects,
            source_sha256=clip.source_sha256,
            camera_source=calibration["mode"],
            convention="ball_center",
        )
        record["labelled_lifting"] = labelled_lifting(
            labels, numeric, cameras, frames, config, metadata, camera_objects, run
        )
    return record, fit


def labelled_lifting(
    labels, numeric, cameras, frames, config, metadata, camera_objects, run
):
    from tennis3d.evaluation.landing import landing_metrics

    evaluate_physical = import_module("tennis3d.evaluation.physical").evaluate_physical
    from tennis3d.inference.predict import predict_view
    from tennis3d.lifting.checkpoint import load_lifter

    uv = np.full((len(frames["t_s"]), 2), np.nan)
    observed = np.zeros(len(uv), dtype=bool)
    for row in labels.annotations:
        if row.uv is not None:
            uv[row.frame_index], observed[row.frame_index] = row.uv, True
    model, _ = load_lifter(config.lifting.checkpoint, config.lifting.device)
    xyz, available, _ = predict_view(
        model,
        {
            "t_s": frames["t_s"],
            "image_size": frames["image_size"],
            "ball_uv": uv,
            "ball_observed": observed,
            "court_uv": cameras["court_uv"],
            "camera_valid": cameras["camera_valid"],
        },
        device=config.lifting.device,
    )
    fit = evaluate_physical(
        {**numeric, "xyz_m": xyz, "available": available, "input_observed": observed},
        metadata={
            **metadata,
            "mode": "manual_2D_input",
            "prediction_identity": {
                **metadata["prediction_identity"],
                "input": "manual_2D_labels",
            },
        },
        labels=labels,
        cameras=camera_objects,
    )
    write_json(run / "labelled-physical.json", fit)
    return {
        "kind": "manual_2D_input_lifting_diagnostic",
        "availability": float(available.mean()),
        "landing": landing_metrics(
            labels,
            frames["t_s"],
            xyz,
            available,
            camera_objects,
            source_sha256=labels.source_sha256,
            camera_source=metadata["camera_source"],
            convention="ball_center",
        ),
        "physical_path": f"clips/{labels.clip_id}/labelled-physical.json",
        "limitation": "Manual 2D inputs and estimated cameras; no independent airborne XYZ truth",
    }


def write_report(output, result):
    lines = [
        "# Stage 4 evaluation",
        "",
        f"Result kind: {result['kind']}. Status: {result['status']}.",
        "",
        f"Checkpoint SHA-256: `{result['provenance']['release']['checkpoint']['sha256']}`. Weights: selected EMA.",
        "",
        "All detailed numeric results and measurement conditions are in [metrics.json](metrics.json).",
        "",
    ]
    if result["kind"] == "model_only_synthetic":
        lines += [
            "| mode | frames | mean XYZ, m | RMSE, m | p95, m | coverage |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for mode, values in result["metrics"]["modes"].items():
            lines.append(
                f"| {mode} | {values['count']} | {values['mean_m']:.6f} | {values['rmse_m']:.6f} | {values['p95_m']:.6f} | {values['valid_coverage']:.6f} |"
            )
    else:
        lines += [
            "| preselected clip | status | exported coverage | labels |",
            "| --- | --- | ---: | --- |",
        ]
        for row in result["records"]:
            coverage = (
                row.get("execution", {}).get("coverage", {}).get("valid_frame_fraction")
            )
            lines.append(
                f"| {row['clip_id']} | {row['status']} | {coverage if coverage is not None else 'unavailable'} | {row['source']['label_status']} |"
            )
    lines += [
        "",
        "Physical consistency uses the frozen evaluator and its recorded segmentation/exclusions. It does not establish real XYZ accuracy.",
        "",
        "Unresolved limits:",
        "",
        *[f"- {limit}" for limit in result["limitations"]],
        "",
    ]
    (Path(output) / "report.md").write_text("\n".join(lines))


def write_gallery(output, result):
    rows = []
    for row in result["records"]:
        label = html.escape(row["clip_id"])
        status = html.escape(row["status"])
        link = (
            f'<a href="{html.escape(row["result_path"], quote=True)}">Viewer result JSON</a>'
            if row.get("result_path")
            else "Result unavailable"
        )
        rows.append(
            f"<tr><td>{label}</td><td>{status}</td><td>{link}</td><td>{html.escape('; '.join(row.get('failure_reasons', [])))}</td></tr>"
        )
    (Path(output) / "gallery.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Frozen real gallery</title><h1>Frozen real gallery</h1><p>Open the standalone viewer and load each result JSON. Every preselected clip and failure is retained. Dense real XYZ accuracy is unmeasured. Original media and detector weights are excluded.</p><table><tr><th>Clip</th><th>Status</th><th>Artifact</th><th>Failure</th></tr>'
        + "".join(rows)
        + "</table></html>"
    )


def package(
    synthetic_path,
    real_path,
    viewer_path,
    output,
    *,
    contacts_path=None,
    stress_path=None,
    stress_inventory_path=None,
):
    """Package measured JSON/derived trajectories; never copy original media/weights."""
    metrics_bytes = {
        "synthetic": (Path(synthetic_path) / "metrics.json").read_bytes(),
        "real": (Path(real_path) / "metrics.json").read_bytes(),
    }
    if stress_path is not None:
        metrics_bytes["resolution-stress"] = (
            Path(stress_path) / "metrics.json"
        ).read_bytes()
    synthetic_result, real_result = (
        _loads(metrics_bytes["synthetic"]),
        _loads(metrics_bytes["real"]),
    )
    if (
        synthetic_result["kind"] != "model_only_synthetic"
        or real_result["kind"] != "end_to_end_real"
    ):
        raise ValueError("package requires measured synthetic and real run directories")
    if (
        synthetic_result["provenance"]["release"]
        != real_result["provenance"]["release"]
    ):
        raise ValueError("package runs use different checkpoint releases")
    from tennis3d.evaluation.review_bundle import (
        checked_supplements,
        copy_contact_reports,
        copy_real_reports,
        verify_contact_inputs,
        write_playable_gallery,
    )

    contacts, stress, contact_files = checked_supplements(
        real_path,
        real_result,
        contacts_path,
        stress_path,
        stress_inventory_path,
        real_metrics_bytes=metrics_bytes["real"],
        stress_metrics_bytes=metrics_bytes.get("resolution-stress"),
    )
    output = fresh_output(output)

    def verify_packaged_metrics(names):
        for name in names:
            if (output / name / "metrics.json").read_bytes() != metrics_bytes[name]:
                raise ValueError("packaged evaluation metrics changed after validation")

    summarize_physical = import_module(
        "tennis3d.evaluation.physical_controls"
    ).summarize_physical

    def physical_reports():
        for path in sorted((Path(synthetic_path) / "physical").glob("*.json")):
            target = output / "synthetic" / "physical" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            yield read_json(path)
        from tennis3d.evaluation.inventory import Clip

        for row in real_result["records"]:
            paths = [(row.get("physical_path"), "physical.json", "end_to_end")]
            labelled = row.get("labelled_lifting")
            if labelled:
                paths.append(
                    (
                        labelled.get("physical_path"),
                        "labelled-physical.json",
                        "manual_2D_input",
                    )
                )
            for relative, filename, mode in paths:
                if relative is None:
                    continue
                clip = Clip.model_validate(row["source"])
                expected = f"clips/{clip.clip_id}/{filename}"
                if row["clip_id"] != clip.clip_id or relative != expected:
                    raise ValueError("physical report path differs from clip identity")
                source = Path(real_path) / expected
                report = read_json(source)
                metadata = report["metadata"]
                if (
                    metadata["artifact_id"] != clip.clip_id
                    or metadata.get("mode") != mode
                    or metadata["source_identity"].get("source_sha256")
                    != clip.source_sha256
                    or metadata["prediction_identity"].get("checkpoint_sha256")
                    != real_result["provenance"]["release"]["checkpoint"]["sha256"]
                ):
                    raise ValueError(
                        "physical report source/checkpoint/mode differs from evaluation"
                    )
                target = output / "real" / expected
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                yield report

    write_json(
        output / "physical-controls.json", summarize_physical(physical_reports())
    )
    for name, source in (
        ("synthetic", Path(synthetic_path)),
        ("real", Path(real_path)),
    ):
        for filename in ("metrics.json", "report.md", "gallery.html"):
            if (source / filename).is_file():
                target = output / name / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / filename, target)
    # Generated trajectories are included in this private review package.
    # Original footage and detector weights are never bundled.
    for row in real_result["records"]:
        if row.get("result_path"):
            from tennis3d.evaluation.inventory import Clip
            from tennis3d.schemas import read_result

            clip = Clip.model_validate(row["source"])
            expected_path = f"clips/{clip.clip_id}/result.json"
            if row["clip_id"] != clip.clip_id or row["result_path"] != expected_path:
                raise ValueError(
                    "gallery result path differs from frozen clip identity"
                )
            source = Path(real_path) / expected_path
            result = read_result(source)
            if (
                result.source.sha256 != clip.source_sha256
                or result.provenance.lifter_checkpoint_sha256
                != real_result["provenance"]["release"]["checkpoint"]["sha256"]
            ):
                raise ValueError(
                    "gallery result source/checkpoint differs from evaluation"
                )
            target = output / "real" / expected_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    if contacts is not None:
        verify_contact_inputs(real_path, contacts, packaged_real=output / "real")
    verify_packaged_metrics(("synthetic", "real"))
    shutil.copyfile(viewer_path, output / "viewer.html")
    write_playable_gallery(output / "real", real_result, viewer_path)
    if contacts is not None:
        copy_contact_reports(contacts_path, output / "contacts", contact_files)
    stress_files = {}
    if stress is not None:
        assert stress_path is not None
        stress_files = copy_real_reports(
            stress_path,
            output / "resolution-stress",
            stress,
            synthetic_result["provenance"]["release"]["checkpoint"]["sha256"],
        )
        verify_packaged_metrics(("resolution-stress",))
        write_playable_gallery(
            output / "resolution-stress",
            stress,
            viewer_path,
            title="Separate resolution-stress gallery",
        )
        frozen_path = (
            Path(stress_inventory_path)
            if stress_inventory_path is not None
            else Path(stress_path) / "inventory.json"
        )
        frozen_target = output / "resolution-stress" / "inventory.frozen.json"
        shutil.copyfile(frozen_path, frozen_target)
        if file_sha256(frozen_target) != stress["inventory_sha256"]:
            raise ValueError("stress frozen inventory changed during packaging")
    if contacts is not None:
        verify_contact_inputs(real_path, contacts, packaged_real=output / "real")
    combined = [
        "# Final delivery review",
        "",
        "Selected EMA checkpoint: `"
        + synthetic_result["provenance"]["release"]["checkpoint"]["sha256"]
        + "`.",
        "",
        "See [synthetic report](synthetic/report.md), [real report](real/report.md) and [fixed gallery](real/gallery.html).",
        "",
        "Physical residual distributions and synthetic discrimination are in [physical-controls.json](physical-controls.json) for real and synthetic cohorts side by side. Real image references from detections are dependent; landing references require audited annotations and calibration. Dense airborne real XYZ is unmeasured.",
        "",
        "Public model release and representative gallery acceptance remain separate release decisions.",
        "",
    ]
    if contacts is not None:
        combined += [
            "See [manual contact evaluation](contacts/report.md) for all frozen references, raw/export cohorts and separately segmented manual/automatic ODE consistency. Contacts are unclassified heuristic episodes; reference-type recall does not establish typed precision.",
            "",
        ]
    if stress is not None:
        combined += [
            "The [resolution-stress report](resolution-stress/report.md) and [stress gallery](resolution-stress/gallery.html) retain their separate frozen inventory. These results are not pooled with high-resolution clips and are not a paired resolution comparison.",
            "",
        ]
    (output / "report.md").write_text("\n".join(combined))
    verify_packaged_metrics(metrics_bytes)
    if any(
        file_sha256(output / "resolution-stress" / relative) != digest
        for relative, digest in stress_files.items()
    ):
        raise ValueError("packaged stress report changed after validation")
    files = {
        str(path.relative_to(output)): file_sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    write_json(
        output / "package.json",
        {
            "schema_version": "1.0",
            "files": files,
            "checkpoint_sha256": synthetic_result["provenance"]["release"][
                "checkpoint"
            ]["sha256"],
            "status": "private_reviewable_evaluation_package",
            "model_publication": "requires a separate release",
            "original_media_and_detector_weights": "excluded",
            "real_results": "bundled derived JSON for private review; publication rights must be established",
            "gallery_ready": real_result["inventory"]["required_gallery_ready"],
            "contact_supplement": contacts is not None,
            "separate_resolution_stress_supplement": stress is not None,
            "offline_playback": "per-clip viewer.html embeds the unchanged validated result; no server or network required",
        },
    )
    return {"output": str(output), "files": files}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument(
        "--pins", type=Path, default=Path("configs/release-pins-main-v1.json")
    )
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    for name in ("synthetic", "real"):
        p = sub.add_parser(name)
        p.add_argument("--release", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        if name == "synthetic":
            p.add_argument("--source", type=Path, required=True)
            p.add_argument("--resume", action="store_true")
        else:
            p.add_argument("--inventory", type=Path, required=True)
            p.add_argument(
                "--config",
                type=Path,
                default=Path("configs/stage4-reconstruction.yaml"),
            )
    p = sub.add_parser(
        "synthetic-inference",
        help="save and seal every prediction; the only phase that may hold a GPU",
    )
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--gpu-reservation", type=Path)
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser(
        "synthetic-analysis",
        help="score a sealed inference root on CPU without a GPU allocation",
    )
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--inference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inference-allocation-receipt", type=Path, action="append")
    p.add_argument(
        "--physics-workers",
        type=int,
        default=1,
        choices=range(1, 9),
        help="processes fitting physics; reports and metrics equal the serial run",
    )
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("package")
    for name in ("synthetic", "real", "viewer", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--contacts", type=Path)
    p.add_argument("--resolution-stress", type=Path)
    p.add_argument("--resolution-stress-inventory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = plan(args.pins, args.inventory, args.output)
        elif args.command == "synthetic":
            result = synthetic(
                args.release,
                args.source,
                args.output,
                device=args.device,
                resume=args.resume,
            )
        elif args.command == "synthetic-inference":
            from tennis3d.evaluation.synthetic_phases import infer

            seal = infer(
                args.release,
                args.source,
                args.output,
                device=args.device,
                resume=args.resume,
                reservation_path=args.gpu_reservation,
            )
            # The seal lists every prediction; print its identity, not 20,000 rows.
            result = {k: v for k, v in seal.items() if k != "predictions"}
            result["prediction_count"] = len(seal["predictions"])
        elif args.command == "synthetic-analysis":
            # Hide every GPU before torch can see one; analysis never needs CUDA.
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            from tennis3d.evaluation.synthetic_phases import analyze

            result = analyze(
                args.release,
                args.source,
                args.inference,
                args.output,
                resume=args.resume,
                allocation_receipts=args.inference_allocation_receipt or (),
                physics_workers=args.physics_workers,
            )
        elif args.command == "real":
            result = real(
                args.release,
                args.inventory,
                args.config,
                args.output,
                device=args.device,
            )
        else:
            result = package(
                args.synthetic,
                args.real,
                args.viewer,
                args.output,
                contacts_path=args.contacts,
                stress_path=args.resolution_stress,
                stress_inventory_path=args.resolution_stress_inventory,
            )
        print(json.dumps(result, allow_nan=False, indent=2))
        return 0
    except (
        ValueError,
        OSError,
        RuntimeError,
        ImportError,
        KeyError,
        TypeError,
    ) as error:
        print(f"Evaluation unavailable: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
