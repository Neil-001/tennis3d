"""Fixed complete-view synthetic XYZ evaluation, independent of export filters."""

import time
from contextlib import nullcontext

import numpy as np

from tennis3d.artifacts import file_sha256
from tennis3d.evaluation.metrics import XYZMetrics
from tennis3d.inference.predict import predict_view
from tennis3d.inference.windows import make_windows
from tennis3d.lifting.checkpoint import load_lifter
from tennis3d.lifting.data import ViewDataset
from tennis3d.lifting.runtime import environment, resolve_device, synchronize


def input_mode(arrays: dict, mode: str) -> dict:
    if mode == "corrupted":
        return arrays
    if mode != "clean":
        raise ValueError("input mode must be clean or corrupted")
    # The clean mode removes detector/calibration corruption, but never invents
    # observations outside the image. Camera/label support is identical.
    return {
        **arrays,
        "ball_uv": arrays["ball_clean_uv"],
        "court_uv": arrays["court_clean_uv"],
        "ball_observed": arrays["ball_in_frame"],
    }


def preflight(dataset: ViewDataset, modes=("corrupted", "clean")):
    """Fail before scoring if any frozen evaluation window has no evidence."""
    dataset.verify_identity()
    checked = dataset._evaluation_preflight_modes
    pending = tuple(mode for mode in modes if mode not in checked)
    if not pending:
        return
    for index in range(len(dataset)):
        arrays = dataset[index]
        supported = arrays["camera_valid"] & arrays["label_valid"]
        if not supported.any():
            raise ValueError("evaluation view has no supported labels")
        for mode in pending:
            inputs = input_mode(arrays, mode)
            for w in make_windows(inputs["camera_valid"]):
                if not inputs["ball_observed"][w.start : w.stop].any():
                    raise ValueError(
                        f"{mode} view {dataset.records[index]['view_id']} has an all-missing evaluation window"
                    )

    dataset.verify_identity()
    dataset._evaluation_preflight_modes = checked | set(pending)


def evaluate_model(
    model,
    manifest,
    *,
    device="cpu",
    modes=("corrupted", "clean"),
    on_view=None,
    prediction_provider=None,
) -> dict:
    """Score every declared view/frame once after 250/125 centre-weight merging."""
    context = (
        nullcontext(manifest)
        if isinstance(manifest, ViewDataset)
        else ViewDataset(manifest)
    )
    with context as dataset:
        return _evaluate_dataset(
            model,
            dataset,
            device=device,
            modes=modes,
            on_view=on_view,
            prediction_provider=prediction_provider,
        )


def _evaluate_dataset(
    model, dataset, *, device, modes, on_view=None, prediction_provider=None
):
    manifest_hash = dataset.manifest_sha256
    preflight(dataset, modes)
    metrics = {mode: XYZMetrics() for mode in modes}
    # Per mode and reference: [error sum in pixels, frame count].
    reprojection = {mode: {"clean": [0.0, 0], "observed": [0.0, 0]} for mode in modes}
    for index in range(len(dataset)):
        arrays = dataset[index]
        for mode in modes:
            inputs = input_mode(arrays, mode)

            def compute(inputs=inputs):
                if on_view is not None:
                    synchronize(device)
                started = time.perf_counter()
                xyz, covered, disagreement = predict_view(
                    model, inputs, device=device, require_complete=True
                )
                if on_view is not None:
                    synchronize(device)
                return xyz, covered, disagreement, time.perf_counter() - started

            xyz, covered, disagreement, prediction_wall_s = (
                compute()
                if prediction_provider is None
                else prediction_provider(
                    index, dataset.records[index], arrays, mode, compute
                )
            )
            metrics[mode].add(
                dataset.records[index], arrays, xyz, covered, inputs["ball_observed"]
            )
            camera_xyz = (
                np.einsum("tij,tj->ti", arrays["camera_R"], xyz) + arrays["camera_t"]
            )
            # Projection against clean camera and clean observations is separate
            # from detector consistency under the supplied context intrinsics.
            for name, intrinsics, reference, mask in (
                (
                    "clean",
                    arrays["camera_K"],
                    arrays["ball_clean_uv"],
                    arrays["label_valid"],
                ),
                (
                    "observed",
                    arrays["camera_K"]
                    if mode == "clean"
                    else arrays["estimated_camera_K"],
                    inputs["ball_uv"],
                    inputs["ball_observed"],
                ),
            ):
                homogeneous = np.einsum("tij,tj->ti", intrinsics, camera_xyz)
                valid = (
                    covered
                    & arrays["camera_valid"]
                    & mask
                    & (camera_xyz[:, 2] > 0)
                    & np.isfinite(reference).all(-1)
                )
                uv = homogeneous[valid, :2] / homogeneous[valid, 2, None]
                error = np.linalg.norm(uv - reference[valid], axis=-1)
                if not np.isfinite(error).all():
                    raise ValueError("nonfinite positive-depth reprojection")
                reprojection[mode][name][0] += float(error.sum())
                reprojection[mode][name][1] += int(valid.sum())
            if on_view is not None:
                on_view(
                    dataset.records[index],
                    arrays,
                    inputs,
                    xyz,
                    covered,
                    mode,
                    {
                        "disagreement_m": disagreement,
                        "prediction_wall_s": prediction_wall_s,
                    },
                )
    results = {}
    for mode, metric in metrics.items():
        results[mode] = metric.finish()
        results[mode]["reprojection"] = {
            name: {"count": count, "mean_px": total / count if count else None}
            for name, (total, count) in reprojection[mode].items()
        }
    dataset.verify_identity()
    return {
        "schema_version": "1.0",
        "manifest_sha256": manifest_hash,
        "source_identity": dataset.identity,
        "split": dataset.manifest["split"],
        "window": {
            "max_frames": 250,
            "stride": 125,
            "weight": "min(i+1,L-i) for i=0..L-1, L=real window length",
        },
        "modes": results,
    }


def evaluate_checkpoint(
    checkpoint, manifest, *, device="auto", modes=("corrupted", "clean")
):
    device = resolve_device(device)
    checkpoint_hash = file_sha256(checkpoint)
    model, state = load_lifter(checkpoint, device)
    result = evaluate_model(model, manifest, device=device, modes=modes)
    if file_sha256(checkpoint) != checkpoint_hash:
        raise ValueError("checkpoint changed during evaluation")
    result.update(
        checkpoint_sha256=checkpoint_hash,
        checkpoint_step=state["step"],
        artifact_label=state["config"]["artifact_label"],
        environment=environment(device),
    )
    return result
