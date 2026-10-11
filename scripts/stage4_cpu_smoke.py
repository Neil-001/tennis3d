"""Run two fixture updates with the production architecture, including resume.

Use a fresh local environment with the lifting extra. Add perception only for
the optional full-video CLI check, which uses existing assets without downloads.
The output must be a new directory outside this checkout. Real-video execution
requires the caller to freeze its evaluation inventory before invoking this tool.
"""

import argparse
import hashlib
import importlib.metadata
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

# Disable accelerator visibility before importing torch or the package.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

from tennis3d.artifacts import (
    canonical_json,
    file_sha256,
    read_json,
    read_npz,
    write_json,
)
from tennis3d.config import load_config
from tennis3d.evaluation.synthetic import evaluate_checkpoint
from tennis3d.lifting.checkpoint import read_checkpoint
from tennis3d.lifting.model import Lifter
from tennis3d.lifting.settings import TrainConfig
from tennis3d.lifting.train import TrainControl, train

ROOT = Path(__file__).resolve().parents[1]
LABEL = "Stage4 CPU smoke; real XYZ accuracy unmeasured"


def state_hash(value):
    """Hash tensor values and nested state, independently of torch archive bytes."""
    digest = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(
                canonical_json(["tensor", str(array.dtype), list(array.shape)])
            )
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=str):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            digest.update(str(len(item)).encode())
            for child in item:
                add(child)
        else:
            digest.update(canonical_json(item))

    add(value)
    return digest.hexdigest()


def fresh_hash(config):
    random.seed(config.seed)
    np.random.seed(config.seed % 2**32)
    torch.manual_seed(config.seed)
    return state_hash(Lifter(config.model).state_dict())


def optimizer_settings(state):
    groups = []
    for group in state["optimizer"]["param_groups"]:
        values = {k: v for k, v in group.items() if k not in {"params", "lr"}}
        values["lr_schedule"] = {
            k: v for k, v in values["lr_schedule"].items() if k != "step"
        }
        groups.append(values)
    return groups


def recovery_hash(state):
    return state_hash(
        {key: state[key] for key in ("model", "ema", "optimizer", "sampler", "rng")}
    )


def cli(arguments, output, name):
    command = [sys.executable, "-m", "tennis3d", *map(str, arguments)]
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    (output / f"{name}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output / f"{name}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    evidence = {"command": command, "exit_code": completed.returncode}
    write_json(output / f"{name}.command.json", evidence)
    return evidence


def video_smoke(video, assets, checkpoint, output):
    from tennis3d.inference.settings import ReconstructionConfig
    from tennis3d.perception.assets import manifest, verify
    from tennis3d.schemas import Result
    from tennis3d.video.decode import decode

    before = {"video": file_sha256(video), "assets_manifest": file_sha256(assets)}
    declared_assets = manifest(assets)
    for asset in declared_assets.assets.values():
        verify(asset)
    config = ReconstructionConfig.model_validate(
        {
            "detection": {
                "device": "cpu",
                "threads": 2,
                "assets_manifest": assets,
                "save_heatmaps": False,
            },
            "lifting": {
                "checkpoint": checkpoint,
                "device": "cpu",
                "threads": 2,
                "model_label": LABEL,
            },
        }
    )
    config_path = output / "reconstruction.config.json"
    write_json(config_path, config.model_dump(mode="json"))
    run = output / "real-development"
    invocation = cli(
        [
            "reconstruct",
            video,
            "--output",
            run,
            "--config",
            config_path,
            "--device",
            "cpu",
        ],
        output,
        "reconstruct",
    )
    # Exit 4 still represents a completed pipeline with no usable reconstruction.
    if invocation["exit_code"] not in (0, 4):
        raise RuntimeError(
            f"reconstruction failed; inspect {output}/reconstruct.stdout.txt"
        )
    pts, bases, times = [], [], []
    for frame in decode(video):
        pts.append(frame.pts)
        bases.append(frame.time_base.as_integer_ratio())
        times.append(frame.t_s)
    if not times:
        raise ValueError("source video has no decoded frames")
    frames, _ = read_npz(run / "frames.npz")
    raw, _ = read_npz(run / "predictions_raw.npz")
    result = Result.model_validate(read_json(run / "result.json"))
    expected_index = np.arange(len(times), dtype=np.int64)
    for name, expected in (
        ("frame_index", expected_index),
        ("pts", np.asarray(pts)),
        ("time_base", np.asarray(bases)),
        ("t_s", np.asarray(times)),
    ):
        if not np.array_equal(frames[name], expected):
            raise ValueError(f"source {name} differs from detection timeline")
    for name, expected in (("frame_index", expected_index), ("t_s", np.asarray(times))):
        if not np.array_equal(raw[name], expected):
            raise ValueError(f"raw prediction {name} differs from source timeline")
    if result.source.frame_count != len(times) or any(
        frame.frame_index != index or frame.t_s != times[index]
        for index, frame in enumerate(result.frames)
    ):
        raise ValueError("result JSON does not retain every source row and timestamp")
    for filename, key in (
        ("ball_track.npz", "observation"),
        ("court.npz", "raw_uv"),
        ("cameras.npz", "camera_valid"),
    ):
        arrays, _ = read_npz(run / filename)
        if len(arrays[key]) != len(times):
            raise ValueError(f"{filename} does not retain every source row")
    if before != {"video": file_sha256(video), "assets_manifest": file_sha256(assets)}:
        raise ValueError("source video or asset manifest changed")
    for asset in declared_assets.assets.values():
        verify(asset)
    return {
        **invocation,
        "source_path": str(video),
        "source_sha256": before["video"],
        "asset_manifest_sha256": before["assets_manifest"],
        "source_frame_count": len(times),
        "all_source_pts_retained": True,
        "all_stage_timeline_lengths_retained": True,
        "source_last_t_s": times[-1],
        "result_status": result.status,
        "metrics_path": str(run / "metrics.json"),
        "real_xyz_accuracy": "unmeasured; unlabelled development execution only",
        "model_label": LABEL,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/stage4-cpu-smoke.yaml"
    )
    parser.add_argument("--video", type=Path)
    parser.add_argument("--assets-manifest", type=Path)
    args = parser.parse_args()
    if bool(args.video) != bool(args.assets_manifest):
        parser.error("--video and --assets-manifest must be supplied together")
    output = args.output.resolve()
    smoke_config_path = args.config.resolve()
    video = args.video.resolve() if args.video else None
    assets = args.assets_manifest.resolve() if args.assets_manifest else None
    if output == ROOT or ROOT in output.parents:
        parser.error("--output must be outside the repository")
    if output.exists():
        parser.error("--output must be a new, nonexistent directory")
    if sys.prefix == sys.base_prefix or ROOT in Path(sys.prefix).resolve().parents:
        parser.error("use a fresh virtual environment outside this checkout")
    os.chdir(ROOT)
    config = load_config(TrainConfig, smoke_config_path)
    production_path = ROOT / "configs/stage4-preparation-main-v1/full-run.yaml"
    production_hash = file_sha256(production_path)
    production = load_config(TrainConfig, production_path)
    if config.model != production.model or any(
        getattr(config, key) != getattr(production, key)
        for key in ("seed", "learning_rate", "ema_decay", "gradient_clip")
    ):
        raise ValueError(
            "smoke must retain selected architecture and optimizer settings"
        )
    expected_fixtures = {
        "train_manifest": ROOT / "tests/fixtures/lifting/train.manifest.json",
        "val_manifest": ROOT / "tests/fixtures/lifting/val.manifest.json",
    }
    for key, expected in expected_fixtures.items():
        if Path(getattr(config, key)).resolve() != expected:
            raise ValueError("smoke accepts only the genuine tiny train/val fixtures")
    if (
        config.device != "cpu"
        or config.max_steps != 2
        or "smoke" not in config.artifact_label.lower()
    ):
        raise ValueError(
            "smoke requires CPU, exactly two total updates and a smoke label"
        )
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    initial = fresh_hash(config)
    if fresh_hash(config) != initial:
        raise RuntimeError("seeded fresh model initialization is not deterministic")
    output.mkdir(parents=True, exist_ok=False)
    first_output, resumed_output = output / "fresh", output / "resumed"
    observed = {}

    def capture(event, directory, key):
        if event["phase"] == "training" and "loss" not in event:
            state = read_checkpoint(directory / "checkpoints/last.pt")
            observed[key] = {
                "step": state["step"],
                "model_hash": state_hash(state["model"]),
                "recovery_hash": recovery_hash(state),
            }

    first = train(
        config,
        first_output,
        control=TrainControl(
            stop_after_steps=1,
            on_update=lambda e: capture(e, first_output, "fresh"),
        ),
    )
    prior = read_checkpoint(first["last_checkpoint"])
    if first["step"] != 1 or observed["fresh"]["model_hash"] != initial:
        raise RuntimeError(
            "fresh update did not start from the declared random initialization"
        )
    resumed = train(
        config,
        resumed_output,
        resume=first["last_checkpoint"],
        control=TrainControl(
            stop_after_steps=1,
            on_update=lambda e: capture(e, resumed_output, "resumed"),
        ),
    )
    final = read_checkpoint(resumed["last_checkpoint"])
    if resumed["step"] != 2 or observed["resumed"]["recovery_hash"] != recovery_hash(
        prior
    ):
        raise RuntimeError(
            "resume did not restore exact model/EMA/optimizer/sampler/RNG state"
        )
    if (
        optimizer_settings(prior) != optimizer_settings(final)
        or prior["config"] != final["config"]
    ):
        raise RuntimeError("resume changed the architecture or optimizer settings")
    if any(g["lr_schedule"]["step"] != 2 for g in final["optimizer"]["param_groups"]):
        raise RuntimeError("optimizer schedule did not retain its update counter")
    selected = Path(resumed["best_checkpoint"])
    evaluation = evaluate_checkpoint(selected, config.val_manifest, device="cpu")
    if evaluation["split"] != "val":
        raise RuntimeError("selected EMA evaluation must use validation only")
    write_json(output / "selected-ema-validation.json", evaluation)
    doctor = cli(
        ["doctor", "--device", "cpu", "--checkpoint", selected], output, "doctor"
    )
    if doctor["exit_code"] != 0:
        raise RuntimeError("selected EMA doctor CLI failed")
    report = {
        "schema_version": "1.0",
        "artifact_label": config.artifact_label,
        "accuracy_claim": "none; fixture execution evidence only",
        "real_xyz_accuracy": "unmeasured",
        "total_optimizer_updates": 2,
        "initialization": {
            "kind": "fresh random",
            "seed": config.seed,
            "model_tensor_sha256": initial,
            "repeat_equal": True,
        },
        "resume": {
            "exact_state_restored": True,
            "settings_preserved": True,
            "before_update": observed["resumed"],
            "from_step": 1,
            "to_step": 2,
            "optimizer_settings": optimizer_settings(final),
        },
        "production_config_sha256": production_hash,
        "smoke_config_sha256": file_sha256(smoke_config_path),
        "fixture_manifest_sha256": {
            k: file_sha256(v) for k, v in expected_fixtures.items()
        },
        "selected_checkpoint": str(selected),
        "selected_checkpoint_sha256": file_sha256(selected),
        "selected_step": evaluation["checkpoint_step"],
        "weights": "EMA",
        "environment": {
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "python": platform.python_version(),
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("tennis3d", "numpy", "torch", "pydantic", "PyYAML")
            },
            "threads": torch.get_num_threads(),
            "device": "cpu",
            "cuda_initialized": torch.cuda.is_initialized(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "doctor": doctor,
        "real_development": {
            "status": "not requested",
            "real_xyz_accuracy": "unmeasured",
        },
    }
    if video:
        report["real_development"] = video_smoke(
            video,
            assets,
            selected,
            output,
        )
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU smoke unexpectedly initialized CUDA")
    if file_sha256(production_path) != production_hash:
        raise RuntimeError("production config changed during the smoke run")
    write_json(output / "smoke-report.json", report)
    print(canonical_json(report).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
