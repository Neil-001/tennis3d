"""Small CPU checks; base installations can skip optional perception dependencies."""

from fractions import Fraction

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("pykalman")
pytest.importorskip("skimage")
pytest.importorskip("av")

from tennis3d.ball.associate import AMBIGUOUS, DETECTED, DUPLICATE, associate
from tennis3d.court.calibrate import Estimate, estimate_camera, intrinsic
from tennis3d.court.model import court_landmarks
from tennis3d.court.track import CameraFilter
from tennis3d.geometry.camera import Camera, ImageSize
from tennis3d.geometry.projection import project
from tennis3d.perception.config import CalibrationConfig
from tennis3d.video.decode import decode
from tennis3d.video.duplicates import DuplicateDetector


def camera(width=1920, center=(2.0, -30.0, 15.0)):
    center = np.array(center)
    forward = -center / np.linalg.norm(center)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    rotation = np.stack([right, np.cross(forward, right), forward])
    size = ImageSize(width, width * 9 // 16)
    return Camera(intrinsic(width * 1.2, size), rotation, -rotation @ center, size)


@pytest.mark.parametrize("width", [640, 1920])
def test_focal_pose_recovery_outliers_missing_and_scale(width):
    truth = camera(width)
    uv = project(truth, court_landmarks())[0]
    config = CalibrationConfig()
    clean = estimate_camera(uv, np.ones(14, bool), truth.image_size, config)
    assert clean.valid, clean.reason
    np.testing.assert_allclose(clean.camera.center, truth.center, atol=1e-4)
    np.testing.assert_allclose(clean.camera.K, truth.K, atol=1e-3)
    rng = np.random.default_rng(17)
    noisy = uv + rng.normal(0, width / 4000, uv.shape)
    noisy[8] += width * 0.2
    mask = np.ones(14, bool)
    mask[[4, 7]] = False
    estimate = estimate_camera(noisy, mask, truth.image_size, config)
    assert estimate.valid, estimate.reason
    assert not estimate.inliers[8]
    assert np.linalg.norm(estimate.camera.center - truth.center) < 1
    mask[:] = False
    mask[[0, 1, 4, 6, 8, 9]] = True
    assert not estimate_camera(uv, mask, truth.image_size, config).valid


def test_weak_planar_family_is_not_certified():
    truth = camera(center=(0.0, -240.0, 12.0))
    uv = project(truth, court_landmarks())[0]
    estimate = estimate_camera(
        uv, np.ones(14, bool), truth.image_size, CalibrationConfig()
    )
    assert not estimate.valid
    assert estimate.reason == "ambiguous_focal_depth"
    assert estimate.alternatives


def test_filter_uses_seconds_gaps_cuts_and_rotation_chart():
    truth = camera()
    uv = project(truth, court_landmarks())[0]
    measured = Estimate(truth, True, "accepted", np.ones(14, bool))
    tracker = CameraFilter(CalibrationConfig())
    result, _ = tracker.update(0.0, measured, uv, truth.image_size)
    assert result is not None
    for t in (0.02, 0.19, 0.49):
        result, reason = tracker.update(t, Estimate(), uv, truth.image_size)
        assert result is not None and reason == "predicted"
    result, reason = tracker.update(0.51, Estimate(), uv, truth.image_size)
    assert result is None and reason == "unsupported_gap"
    result, _ = tracker.update(0.6, measured, uv, truth.image_size)
    assert result is not None
    result, _ = tracker.update(0.7, measured, uv, truth.image_size, cut=True)
    assert result is None
    with pytest.raises(ValueError):
        tracker.update(0.7, measured, uv, truth.image_size)
    # Equivalent matrix orientations straddle the global Rodrigues pi boundary.
    from scipy.spatial.transform import Rotation

    tracker = CameraFilter(CalibrationConfig())
    for index, angle in enumerate([179.99, 180.01, 180.02]):
        r = Rotation.from_euler("x", angle, degrees=True).as_matrix()
        c = Camera(truth.K, r, -r @ truth.center, truth.image_size)
        # No valid landmarks on these intentionally non-broadcast orientations;
        # seed accepted measurement solely to test the local representation.
        e = Estimate(c, True, "accepted", np.ones(14, bool))
        tracker.update(index * 0.04, e, uv, truth.image_size)
    assert tracker.mean is not None
    assert np.linalg.norm(tracker.mean[4:7]) < 0.01


def test_duplicates_require_moving_region_or_exact_equality():
    tracker = DuplicateDetector()
    base = np.zeros((360, 640, 3), np.uint8)
    base[90:120, 100:125] = 200
    assert tracker.update(base, 0)[0] == -1
    moving = base.copy()
    moving[90:120, 100:125] = 0
    moving[90:120, 110:135] = 200
    assert tracker.update(moving, 1)[0] == -1
    assert tracker.update(moving.copy(), 2)[0] == 1
    assert tracker.update(moving.copy(), 3)[0] == 1
    tiny = moving.copy()
    tiny[200:204, 300:304] = 255
    assert tracker.update(tiny, 4)[0] == -1


def test_candidates_preserve_ambiguity_and_duplicates():
    one = np.array([[10.0, 10.0, 3.0, 0.8], [11.0, 12.0, 4.0, 0.7]])
    assert associate(one)[1] == DETECTED
    assert associate(np.r_[one, [[100.0, 90.0, 3.0, 0.9]]])[1] == AMBIGUOUS
    assert associate(one, duplicate_of=0)[1] == DUPLICATE


def test_pyav_variable_pts_preserved(tmp_path):
    import av

    path = tmp_path / "timestamps.mkv"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=25)
        assert isinstance(stream, av.video.stream.VideoStream)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "bgr0"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        for pts in [100, 140, 220, 260]:
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), pts % 255, np.uint8), format="bgr24"
            )
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    frames = list(decode(path))
    assert len(frames) == 4
    assert frames[0].source_time == Fraction(1, 10)
    np.testing.assert_allclose([f.t_s for f in frames], [0, 0.04, 0.12, 0.16])


def test_reencoded_duplicates_need_cadence():
    from tennis3d.video.duplicates import confirm_repetition

    proposed = np.full(22, -1)
    proposed[[3, 9, 15, 20]] = [2, 8, 14, 19]
    result = confirm_repetition(proposed, np.zeros(22, bool))
    assert result[20] == -1
    np.testing.assert_array_equal(result[[3, 9, 15]], [2, 8, 14])
    exact = np.zeros(22, bool)
    exact[20] = True
    assert confirm_repetition(proposed, exact)[20] == 19


def test_modest_motion_and_zoom_and_invalid_ambiguity():
    config = CalibrationConfig()
    tracker = CameraFilter(config)
    for index, time in enumerate([0.0, 0.03, 0.09, 0.12, 0.2, 0.26, 0.3]):
        truth = camera(center=(2 + 0.2 * time, -30.0, 15.0))
        k = truth.K.copy()
        k[0, 0] *= 1 + 0.02 * time
        k[1, 1] *= 1 + 0.02 * time
        truth = Camera(k, truth.R, truth.t, truth.image_size)
        uv = project(truth, court_landmarks())[0]
        estimate = estimate_camera(uv, np.ones(14, bool), truth.image_size, config)
        assert estimate.valid
        filtered, _ = tracker.update(time, estimate, uv, truth.image_size)
        assert filtered is not None
        assert np.linalg.norm(filtered.center - truth.center) < 0.2
    ambiguous = Estimate(reason="ambiguous_focal_depth")
    assert tracker.update(0.31, ambiguous, uv, truth.image_size)[0] is None


@pytest.mark.parametrize(("frames", "decode_error"), [(0, False), (0, True), (1, True)])
def test_failed_detection_publishes_every_decoded_time(
    tmp_path, monkeypatch, frames, decode_error
):
    pytest.importorskip("torch")
    from tennis3d.artifacts import read_npz
    from tennis3d.perception import detect as stage
    from tennis3d.perception.config import DetectConfig
    from tennis3d.video.decode import DecodedFrame

    def decode(path):
        for i in range(frames):
            image = np.zeros((72, 128, 3), np.uint8)
            yield DecodedFrame(i, i, Fraction(1, 30), Fraction(i, 30), i / 30, image)
        if decode_error:
            raise ValueError("Missing presentation timestamp")

    def network(tensor):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(stage, "decode", decode)
    monkeypatch.setattr(stage, "verify", lambda asset: None)
    monkeypatch.setattr(stage, "load_network", lambda *args: network)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    run = tmp_path / "run"
    summary = stage.detect(video, run, DetectConfig(device="cpu"))
    assert summary["status"] == "failed"
    assert summary["frame_count"] == frames
    assert summary["exit_code"] == (3 if frames else 2)
    assert (summary["failure"] is not None) == (decode_error or frames > 0)
    court = read_npz(run / "court.npz")[0]
    assert court["observed"].shape == (frames, 14)
    assert not court["observed"].any()
    track = read_npz(run / "ball_track.npz")[0]
    assert track["ball_uv"].shape == (frames, 2)
    assert (track["observation"] == 0).all()
    assert read_npz(run / "ball_candidates.npz")[0]["offsets"].tolist() == [0] * (
        frames + 1
    )
    assert read_npz(run / "frames.npz")[0]["t_s"].shape == (frames,)


def test_exact_duplicate_does_not_inherit_unconfirmed_near_match():
    from tennis3d.video.duplicates import confirm_repetition

    first = np.zeros((360, 640, 3), np.uint8)
    first[90:120, 100:125] = 200
    moved = np.zeros_like(first)
    moved[90:120, 110:135] = 200
    near = moved.copy()
    near[0, 0, 0] = 1
    tracker = DuplicateDetector()
    proposed, exact = [], []
    for index, frame in enumerate([first, moved, near, near.copy()]):
        proposed.append(tracker.update(frame, index)[0])
        exact.append(tracker.exact_match)
    # The isolated near match stays unique; its exact copy must reference it.
    np.testing.assert_array_equal(confirm_repetition(proposed, exact), [-1, -1, -1, 2])


def test_filter_reinitialization_clears_rejection_history():
    first, second = camera(), camera(center=(12.0, -30.0, 15.0))
    uv_a = project(first, court_landmarks())[0]
    uv_b = project(second, court_landmarks())[0]
    ea = Estimate(first, True, "accepted", np.ones(14, bool))
    eb = Estimate(second, True, "accepted", np.ones(14, bool))
    tracker = CameraFilter(CalibrationConfig())
    sequence = [(ea, uv_a)] + [(eb, uv_b)] * 5 + [(Estimate(reason="duplicate"), uv_b)]
    results = [
        tracker.update(i / 30, e, uv, first.image_size)
        for i, (e, uv) in enumerate(sequence)
    ]
    assert results[4][1] == "innovation_reset"
    assert results[5][1] == "measured"
    assert results[6][0] is not None
    assert results[6][1] == "predicted"
    assert tracker.rejections == 0


@pytest.mark.parametrize("detection_status", ["ok", "failed"])
def test_static_consensus_retains_good_frames_with_raw_pose_outlier(
    tmp_path, detection_status
):
    from tennis3d.artifacts import (
        ArtifactProvenance,
        file_sha256,
        read_npz,
        write_json,
        write_npz,
    )
    from tennis3d.perception.calibrate import calibrate

    truth, outlier = camera(), camera(center=(12.0, -30.0, 15.0))
    uv = np.repeat(project(truth, court_landmarks())[0][None], 31, axis=0)
    uv[15] = project(outlier, court_landmarks())[0]
    provenance = ArtifactProvenance(
        source_sha256="0" * 64,
        config_sha256="0" * 64,
        code_revision="exact-projection-regression",
    )
    write_npz(
        tmp_path / "frames.npz",
        {
            "t_s": np.arange(31) / 30,
            "image_size": np.array([1920, 1080]),
            "duplicate_of": np.full(31, -1),
            "cut": np.zeros(31, bool),
        },
        provenance,
    )
    write_npz(
        tmp_path / "court.npz",
        {"raw_uv": uv, "observed": np.ones((31, 14), bool)},
        provenance,
    )
    write_json(
        tmp_path / "detect.json",
        {
            "status": detection_status,
            "artifacts": {
                name: file_sha256(tmp_path / name)
                for name in ("frames.npz", "court.npz")
            },
        },
    )
    summary = calibrate(tmp_path, CalibrationConfig(mode="static"))
    arrays, _ = read_npz(tmp_path / "cameras.npz")
    assert summary["status"] == ("partial" if detection_status == "ok" else "failed")
    assert summary["exit_code"] == (0 if detection_status == "ok" else 2)
    assert arrays["raw_valid"].all()
    assert arrays["camera_valid"].sum() == 30
    assert not arrays["camera_valid"][15]
    np.testing.assert_allclose(arrays["R"][0], truth.R, atol=1e-6)


def test_rejected_filter_update_cannot_support_missing_predictions():
    first, moved = camera(), camera(center=(3.0, -30.0, 15.0))
    uv_a = project(first, court_landmarks())[0]
    uv_b = project(moved, court_landmarks())[0]
    ea = Estimate(first, True, "accepted", np.ones(14, bool))
    eb = Estimate(moved, True, "accepted", np.ones(14, bool))
    tracker = CameraFilter(CalibrationConfig())
    tracker.update(0.0, ea, uv_a, first.image_size)
    result, reason = tracker.update(1 / 30, eb, uv_b, moved.image_size)
    assert result is None and reason == "filtered_reprojection_error"
    assert tracker.rejections == 0  # The measurement passed innovation gating.
    assert tracker.mean is None and tracker.last_measurement is None
    assert (
        tracker.update(2 / 30, Estimate(reason="duplicate"), uv_b, moved.image_size)[0]
        is None
    )
    assert tracker.update(3 / 30, eb, uv_b, moved.image_size)[1] == "measured"
    result, reason = tracker.update(
        4 / 30, Estimate(reason="duplicate"), uv_b, moved.image_size
    )
    assert result is not None and reason == "predicted"


@pytest.mark.parametrize(
    ("indices", "confirmed"),
    [([4, 6, 8, 100], [4, 6, 8]), ([4, 6, 50, 52], [])],
)
def test_cadence_does_not_confirm_disconnected_holds(indices, confirmed):
    from tennis3d.video.duplicates import confirm_repetition

    proposed = np.full(101, -1)
    proposed[indices] = np.array(indices) - 1
    result = confirm_repetition(proposed, np.zeros(101, bool))
    assert np.flatnonzero(result >= 0).tolist() == confirmed


def test_cli_distinguishes_missing_inputs_and_assets(tmp_path, capsys):
    import json
    from pathlib import Path

    from tennis3d.cli import main
    from tennis3d.perception.assets import manifest

    video = tmp_path / "absent.mp4"
    run = tmp_path / "run"
    assert main(["detect", str(video), "--output", str(run)]) == 2
    assert main(["calibrate", "--run", str(run)]) == 2
    video.touch()
    assets = manifest("configs/assets.yaml").model_dump(mode="json")
    for asset in assets["assets"].values():
        asset["path"] = str(tmp_path / Path(asset["path"]).name)
    config = tmp_path / "assets.json"
    config.write_text(json.dumps(assets))
    assert (
        main(
            [
                "detect",
                str(video),
                "--output",
                str(run),
                "--assets-manifest",
                str(config),
            ]
        )
        == 3
    )
    assert not run.exists()
    capsys.readouterr()


@pytest.mark.parametrize("payload", [b"xy", b"xyz", b"xyzw"])
@pytest.mark.parametrize("download", [False, True])
def test_checkpoint_integrity_failures_return_unavailable(
    tmp_path, monkeypatch, capsys, payload, download
):
    import hashlib
    import io

    from tennis3d.cli import main
    from tennis3d.perception import assets

    checkpoint = tmp_path / "weights.pth"
    entry = assets.Asset(
        url="https://example.test/weights.pth",
        path=checkpoint,
        sha256=hashlib.sha256(b"abc").hexdigest(),
        bytes=3,
        source="test",
        revision="test",
        license="test",
        purpose="test",
    )
    manifest = tmp_path / "assets.json"
    manifest.write_text(assets.AssetManifest(assets={"ball": entry}).model_dump_json())
    if download:
        monkeypatch.setattr(
            assets.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload)
        )
        command = ["assets", "fetch", "--manifest", str(manifest)]
    else:
        checkpoint.write_bytes(payload)
        video = tmp_path / "video.mp4"
        video.touch()
        command = [
            "detect",
            str(video),
            "--output",
            str(tmp_path / "run"),
            "--assets-manifest",
            str(manifest),
        ]
    assert main(command) == 3
    assert not (tmp_path / "run").exists()
    if download:
        assert not checkpoint.exists()
    capsys.readouterr()
