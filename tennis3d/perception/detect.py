"""Streaming detector stage. Large raw heatmaps are individual atomic numeric files."""

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from tennis3d.artifacts import (
    ArtifactProvenance,
    atomic_file,
    file_sha256,
    write_json,
    write_npz,
)
from tennis3d.ball.associate import DETECTED, associate
from tennis3d.config import record_config
from tennis3d.perception.assets import manifest, verify
from tennis3d.perception.network import load_network
from tennis3d.perception.postprocess import ball_map, circles
from tennis3d.perception.provenance import (
    OPENVINO_DEVICES,
    code_identity,
    detection_runtime,
    runtime_versions,
)
from tennis3d.video.decode import decode
from tennis3d.video.duplicates import DuplicateDetector, confirm_repetition


def device_name(requested):
    if requested in OPENVINO_DEVICES:
        return requested
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return (
        "cuda"
        if requested == "cuda" or requested == "auto" and torch.cuda.is_available()
        else "cpu"
    )


def torch_inference(networks, device):
    def court(frame):
        logits = networks["court"](torch.from_numpy(frame).to(device))
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite court network output")
        return (logits[0].sigmoid().cpu().numpy() * 255).astype(np.uint8)

    def ball(frames):
        logits = networks["ball"](torch.from_numpy(frames).to(device))
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite ball network output")
        return logits[0].argmax(dim=0).cpu().numpy().astype(np.uint8)

    return {"court": court, "ball": ball}


def load_inference(assets, device):
    """Per-network callables from float32 NCHW arrays to uint8 host arrays."""
    accelerated = device in OPENVINO_DEVICES
    networks = {
        name: load_network(
            assets.assets[name].path, name, "cpu" if accelerated else device
        )
        for name in ("ball", "court")
    }
    if accelerated:
        from tennis3d.perception.accelerate import load_inference as compile_networks

        return compile_networks(networks, device)
    return torch_inference(networks, device), None


def detect(video: Path, output: Path, config, *, overwrite=False, progress=None):
    from tennis3d.inference.progress import report

    report(progress, "detect", unit="frames")
    if not video.is_file():
        raise FileNotFoundError(video)
    assets = manifest(config.assets_manifest)
    for name in ("ball", "court"):
        verify(assets.assets[name])
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Run directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config_hash = record_config(
        output / "detect.config.json", config, overwrite=overwrite
    )
    source_hash = file_sha256(video)
    device = device_name(config.device)
    torch.set_num_threads(config.threads)
    cv2.setNumThreads(config.threads)
    infer, accelerator = load_inference(assets, device)
    provenance = ArtifactProvenance(
        source_sha256=source_hash,
        config_sha256=config_hash,
        upstream_sha256={
            name: assets.assets[name].sha256 for name in ("ball", "court")
        },
        code_revision=code_identity(),
    )
    history = deque(maxlen=3)
    duplicate_detector = DuplicateDetector(
        config.duplicate_ssim, config.duplicate_motion_error, config.cut_mean_difference
    )
    columns = {
        name: []
        for name in (
            "frame_index",
            "t_s",
            "pts",
            "time_base",
            "duplicate_of",
            "ssim",
            "motion_error",
            "cut",
        )
    }
    exact_duplicates, courts, scores, found_per_frame = [], [], [], []
    origin, size, failure = None, (0, 0), None
    failure_kind = None
    started = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        with torch.inference_mode():
            for frame in decode(video):
                report(progress, "detect", frame.index, unit="frames")
                bgr = frame.bgr
                height, width = bgr.shape[:2]
                size = (width, height)
                scale = np.array([width / 640, height / 360])
                if origin is None:
                    origin = float(frame.source_time)
                duplicate, ssim, motion_error, cut = duplicate_detector.update(
                    bgr, frame.index
                )
                exact_duplicates.append(duplicate_detector.exact_match)
                for name, value in (
                    ("frame_index", frame.index),
                    ("t_s", frame.t_s),
                    ("pts", frame.pts),
                    ("time_base", frame.time_base.as_integer_ratio()),
                    ("duplicate_of", duplicate),
                    ("ssim", ssim),
                    ("motion_error", motion_error),
                    ("cut", cut),
                ):
                    columns[name].append(value)
                resized = cv2.resize(bgr, (640, 360), interpolation=cv2.INTER_LINEAR)
                history.append(resized.astype(np.float32) / 255)
                heatmaps = infer["court"](history[-1].transpose(2, 0, 1).copy()[None])
                raw = np.full((14, 2), np.nan)
                score = np.zeros(14)
                court_candidates, court_offsets = [], [0]
                for index, heatmap in enumerate(heatmaps[:14]):
                    found = circles(heatmap, court=True)
                    court_candidates.extend(found.tolist())
                    court_offsets.append(len(court_candidates))
                    if len(found):
                        raw[index] = found[0, :2] * scale
                        score[index] = found[0, 3]
                courts.append(raw)
                scores.append(score)
                labels = np.zeros((360, 640), dtype=np.uint8)
                found = np.empty((0, 4))
                if len(history) == 3:
                    stacked = np.concatenate(list(reversed(history)), axis=2)
                    labels = infer["ball"](stacked.transpose(2, 0, 1).copy()[None])
                    found = circles(ball_map(labels))
                found_per_frame.append(found)
                if config.save_heatmaps:
                    write_npz(
                        output / "raw" / f"{frame.index:08d}.npz",
                        {
                            "ball_intensity_class": labels,
                            "ball_has_context": np.array(len(history) == 3),
                            "court_sigmoid_uint8": heatmaps,
                            "court_candidates_network": np.asarray(
                                court_candidates
                            ).reshape(-1, 4),
                            "court_candidate_offsets": np.array(
                                court_offsets, dtype=np.int64
                            ),
                        },
                        provenance,
                        overwrite=overwrite,
                    )
                if config.overlay_every and frame.index % config.overlay_every == 0:
                    for index, point in enumerate(raw):
                        if np.isfinite(point).all():
                            p = tuple(np.rint(point).astype(int))
                            cv2.circle(bgr, p, 6, (0, 255, 0), 2)
                            cv2.putText(
                                bgr,
                                str(index),
                                p,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                1,
                            )
                    cv2.putText(
                        bgr,
                        "Raw detector preview (before duplicate confirmation)",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )
                    # Before cadence confirmation, which needs the whole shot.
                    uv, state = associate(found, duplicate, config.candidate_merge_px)
                    if state == DETECTED:
                        uv = tuple(np.rint(uv * scale).astype(int))
                        cv2.circle(bgr, uv, 10, (0, 0, 255), 2)
                    with atomic_file(
                        output / "diagnostics" / f"detect-{frame.index:08d}.jpg",
                        overwrite=overwrite,
                    ) as stream:
                        stream.write(cv2.imencode(".jpg", bgr)[1].tobytes())
    except (ValueError, RuntimeError, OSError) as error:
        failure = str(error)
        failure_kind = "runtime" if isinstance(error, RuntimeError) else "input"
    if device == "cuda":
        torch.cuda.synchronize()
    n = len(columns["t_s"])
    report(progress, "detect", n, n)
    # If inference failed after decoding a timestamp, retain that frame as missing.
    courts += [np.full((14, 2), np.nan)] * (n - len(courts))
    scores += [np.zeros(14)] * (n - len(scores))
    found_per_frame += [np.empty((0, 4))] * (n - len(found_per_frame))
    duplicate_of = confirm_repetition(columns["duplicate_of"], exact_duplicates)
    scale = np.array([size[0] / 640, size[1] / 360])
    tracks, states = [], []
    for found, duplicate in zip(found_per_frame, duplicate_of):
        uv, state = associate(found, duplicate, config.candidate_merge_px)
        tracks.append(uv * scale)
        states.append(state)
    # Radius remains in network pixels; x/y become original pixels.
    candidates = np.concatenate(found_per_frame or [np.empty((0, 4))])
    candidates[:, :2] *= scale
    offsets = np.cumsum([0] + [len(found) for found in found_per_frame], dtype=np.int64)
    frames = {
        "frame_index": np.array(columns["frame_index"], dtype=np.int64),
        "t_s": np.array(columns["t_s"], dtype=np.float64),
        "pts": np.array(columns["pts"], dtype=np.int64),
        "time_base": np.array(columns["time_base"], dtype=np.int64).reshape(n, 2),
        "duplicate_of": duplicate_of,
        "ssim": np.array(columns["ssim"], dtype=np.float64),
        "motion_error": np.array(columns["motion_error"], dtype=np.float64),
        "cut": np.array(columns["cut"], dtype=bool),
        "image_size": np.array(size, dtype=np.int64),
        "time_origin_s": np.array(origin or 0.0),
    }
    write_npz(output / "frames.npz", frames, provenance, overwrite=overwrite)
    raw_uv = np.asarray(courts).reshape(n, 14, 2)
    write_npz(
        output / "court.npz",
        {
            "raw_uv": raw_uv,
            "observed": np.all(np.isfinite(raw_uv), axis=-1),
            "score": np.asarray(scores).reshape(n, 14),
        },
        provenance,
        overwrite=overwrite,
    )
    write_npz(
        output / "ball_candidates.npz",
        {
            "candidates": candidates,
            "offsets": offsets,
        },
        provenance,
        overwrite=overwrite,
    )
    write_npz(
        output / "ball_track.npz",
        {
            "ball_uv": np.asarray(tracks).reshape(n, 2),
            "observation": np.asarray(states, dtype=np.int8),
            "ball_observed": np.asarray(states) == DETECTED,
        },
        provenance,
        overwrite=overwrite,
    )
    summary = {
        "stage": "detect",
        "status": "failed"
        if failure or not n or frames["cut"].any()
        else "partial"
        if any(s != DETECTED for s in states)
        else "ok",
        "source": str(video.resolve()),
        "source_sha256": source_hash,
        "frame_count": n,
        "detected_frame_fraction": states.count(DETECTED) / n if n else 0,
        "failure": failure,
        "failure_kind": failure_kind,
        "exit_code": 3
        if failure_kind == "runtime"
        else 2
        if failure or not n or frames["cut"].any()
        else 0,
        "shot_cut": bool(frames["cut"].any()),
        "device": device,
        "accelerator": accelerator,
        "elapsed_s": time.perf_counter() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated()
        if device == "cuda"
        else None,
        "runtime": runtime_versions(detection_runtime(device)),
        "assets": assets.model_dump(mode="json"),
        "config_sha256": config_hash,
        "artifacts": {
            name: file_sha256(output / name)
            for name in (
                "frames.npz",
                "court.npz",
                "ball_candidates.npz",
                "ball_track.npz",
            )
        },
    }
    write_json(output / "detect.json", summary, overwrite=overwrite)
    return summary
