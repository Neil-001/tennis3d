"""Stage composition on controlled observations, with a real tiny EMA model."""

import importlib.metadata
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tennis3d.artifacts import (
    ArtifactProvenance,
    file_sha256,
    json_sha256,
    read_json,
    read_npz,
    write_json,
    write_npz,
)
from tennis3d.cli import main
from tennis3d.court.model import court_landmarks
from tennis3d.geometry.camera import Camera, ImageSize
from tennis3d.geometry.projection import project
from tennis3d.inference.export import export
from tennis3d.inference.inputs import camera_inputs, detection_inputs, hashes
from tennis3d.inference.lift import lift
from tennis3d.inference.quality import QualityConfig
from tennis3d.inference.settings import LiftConfig
from tennis3d.lifting.model import Lifter, ModelConfig
from tennis3d.perception.config import CalibrationConfig, DetectConfig
from tennis3d.perception.provenance import (
    CALIBRATION_RUNTIME,
    DETECTION_RUNTIME,
    code_identity,
    runtime_versions,
)
from tennis3d.schemas import read_result


def prepared_run(run: Path, *, cut=False):
    run.mkdir()
    n = 520
    pts = np.r_[1000, 1000 + np.cumsum(np.resize([4, 5], n - 1))]
    times = (pts - pts[0]) / 1000.0
    states = np.ones(n, dtype=np.int8)
    states[:2] = 0
    states[3] = 3
    states[300:500] = 0
    duplicate = np.full(n, -1, dtype=np.int64)
    duplicate[3] = 2
    observed = states == 1
    ball = np.full((n, 2), 200.0)
    ball[~observed] = np.nan
    frames = {
        "frame_index": np.arange(n),
        "t_s": times,
        "pts": pts,
        "time_base": np.tile([1, 1000], (n, 1)),
        "time_origin_s": np.array(1.0),
        "image_size": np.array([640, 480]),
        "duplicate_of": duplicate,
        "cut": np.zeros(n, dtype=bool),
    }
    frames["cut"][400] = cut
    config = DetectConfig().model_dump(mode="json")
    provenance = ArtifactProvenance(
        source_sha256="a" * 64,
        config_sha256=json_sha256(config),
        upstream_sha256={"ball": "b" * 64, "court": "c" * 64},
        code_revision=code_identity(),
    )
    write_json(run / "detect.config.json", config)
    write_npz(run / "frames.npz", frames, provenance)
    write_npz(run / "court.npz", {"raw_uv": np.zeros((n, 14, 2))}, provenance)
    write_npz(run / "ball_candidates.npz", {"candidates": np.empty((0, 4))}, provenance)
    write_npz(
        run / "ball_track.npz",
        {"ball_uv": ball, "observation": states, "ball_observed": observed},
        provenance,
    )
    write_json(
        run / "detect.json",
        {
            "source": "controlled.mp4",
            "source_sha256": "a" * 64,
            "frame_count": n,
            "config_sha256": provenance.config_sha256,
            "assets": {
                "assets": {"ball": {"sha256": "b" * 64}, "court": {"sha256": "c" * 64}}
            },
            "artifacts": hashes(
                run,
                ("frames.npz", "court.npz", "ball_candidates.npz", "ball_track.npz"),
            ),
            "status": "failed" if cut else "partial",
            "exit_code": 2 if cut else 0,
            "failure": None,
            "shot_cut": cut,
            "device": "cpu",
            "runtime": runtime_versions(DETECTION_RUNTIME),
            "elapsed_s": 0.1,
            "peak_cuda_bytes": None,
        },
    )
    camera = Camera(
        np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]),
        np.diag([1, -1, -1]),
        np.array([0, 0, 10]),
        ImageSize(640, 480),
    )
    valid = np.ones(n, dtype=bool)
    valid[260] = False
    if cut:
        valid[400:] = False
    arrays = {
        "t_s": times,
        "camera_valid": valid,
        "K": np.tile(camera.K, (n, 1, 1)),
        "R": np.tile(camera.R, (n, 1, 1)),
        "t": np.tile(camera.t, (n, 1)),
        "court_uv": np.tile(project(camera, court_landmarks())[0], (n, 1, 1)),
        "filtered_residual_px": np.zeros((n, 14)),
    }
    config = CalibrationConfig().model_dump(mode="json")
    write_json(run / "calibrate.config.json", config)
    write_npz(
        run / "cameras.npz",
        arrays,
        ArtifactProvenance(
            source_sha256="a" * 64,
            config_sha256=json_sha256(config),
            code_revision=code_identity(),
            upstream_sha256=hashes(run, ("frames.npz", "court.npz", "detect.json")),
        ),
    )
    write_json(
        run / "calibrate.json",
        {
            "config_sha256": json_sha256(config),
            "cameras_sha256": file_sha256(run / "cameras.npz"),
            "frame_count": n,
            "image_size": [640, 480],
            "mode": "estimated_static",
            "runtime": runtime_versions(CALIBRATION_RUNTIME),
            "assumptions": ["controlled fixture"],
        },
    )
    model_config = ModelConfig(width=16, depth=2)
    model = Lifter(model_config)
    for parameter in model.parameters():
        parameter.data.zero_()
    checkpoint = run / "test.pt"
    torch.save(
        {
            "schema_version": "1.0",
            "world_scale_m": 10.0,
            "step": 0,
            "config": {"model": model_config.model_dump()},
            "ema": {
                "module." + key: value for key, value in model.state_dict().items()
            },
        },
        checkpoint,
    )
    return LiftConfig(checkpoint=checkpoint, device="cpu"), frames


def test_source_alignment_gaps_and_cache_identity(tmp_path):
    run = tmp_path / "run"
    config, frames = prepared_run(run)
    lift(run, config)
    raw_hash = file_sha256(run / "predictions_raw.npz")
    summary = export(run, QualityConfig())
    result = read_result(run / "result.json")
    assert summary["status"] == "partial"
    assert len(result.frames) == 520
    assert result.source.width == 640 and result.source.height == 480
    np.testing.assert_array_equal([frame.t_s for frame in result.frames], frames["t_s"])
    assert result.frames[3].duplicate_of == 2
    assert result.frames[3].reconstruction == "inferred"
    assert result.frames[260].reason_codes == ["invalid_camera"]
    assert all("long_gap" in row.reason_codes for row in result.frames[300:500])
    assert result.frames[0].xyz_m == [0.0, 0.0, 0.0]
    assert result.frames[-1].xyz_m == [0.0, 0.0, 0.0]
    raw, _ = read_npz(run / "predictions_raw.npz")
    assert raw["covered"][
        300:500
    ].all()  # Quality does not erase raw model diagnostics.
    assert not raw["covered"][260]
    assert lift(run, config)["predictions_sha256"] == raw_hash
    assert export(run, QualityConfig()) == summary
    with pytest.raises(ValueError, match="cache identity"):
        export(run, QualityConfig(bracketed_gap_s=0.1))
    export(run, QualityConfig(bracketed_gap_s=0.1), overwrite=True)
    assert file_sha256(run / "predictions_raw.npz") == raw_hash
    with (run / "ball_track.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        lift(run, config)


def test_failed_shot_retains_timeline_and_cli_missing_assets(tmp_path, capsys):
    run = tmp_path / "run"
    config, frames = prepared_run(run, cut=True)
    assert (
        main(
            [
                "lift",
                "--run",
                str(run),
                "--checkpoint",
                str(config.checkpoint),
                "--device",
                "cpu",
            ]
        )
        == 2
    )
    assert main(["export", "--run", str(run)]) == 2
    result = read_result(run / "result.json")
    assert result.status == "failed" and result.failure_reasons == ["unsupported_shot"]
    assert len(result.frames) == len(frames["t_s"])
    assert all(frame.xyz_m is None for frame in result.frames)
    assert read_json(run / "metrics.json")["coverage"]["valid_frame_fraction"] == 0
    assert (
        main(
            [
                "reconstruct",
                str(tmp_path / "missing.mp4"),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
    config.checkpoint.unlink()
    assert (
        main(
            [
                "lift",
                "--run",
                str(run),
                "--checkpoint",
                str(config.checkpoint),
                "--device",
                "cpu",
            ]
        )
        == 3
    )


@pytest.mark.parametrize(
    ("stage", "dependency"),
    [("detect", "scikit-image"), ("calibrate", "scipy"), ("calibrate", "pykalman")],
)
@pytest.mark.parametrize("missing", [False, True])
def test_stage_runtime_identity_rejects_changed_or_missing_metadata(
    tmp_path, monkeypatch, stage, dependency, missing
):
    run = tmp_path / "run"
    prepared_run(run)
    detection, frames, _ = detection_inputs(run)
    if missing:
        path = run / f"{stage}.json"
        summary = read_json(path)
        del summary["runtime"][dependency]
        write_json(path, summary, overwrite=True)
    else:
        original = importlib.metadata.version
        monkeypatch.setattr(
            importlib.metadata,
            "version",
            lambda name: "999-review-probe" if name == dependency else original(name),
        )
    with pytest.raises(ValueError, match="runtime identity is missing|runtime changed"):
        if stage == "detect":
            detection_inputs(run)
        else:
            camera_inputs(run, detection, frames)
