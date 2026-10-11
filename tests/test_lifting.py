"""Tensor and geometry contracts, independent of synthetic/perception runtimes."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tennis3d.inference.windows import Window, WindowMerger, make_windows
from tennis3d.lifting.features import batch_features, build_features
from tennis3d.lifting.losses import position_loss
from tennis3d.lifting.model import Lifter, ModelConfig, RotaryEmbedding


def features(n=8, pad_to=None):
    observed = np.arange(n) % 3 == 0
    ball = np.ones((n, 2)) * [640, 180]
    ball[~observed] = np.nan
    return build_features(
        t_s=1000 + np.arange(n) * 0.023,
        ball_uv=ball,
        ball_observed=observed,
        court_uv=np.ones((n, 14, 2)) * [1280, 360],
        camera_valid=np.ones(n, dtype=bool),
        image_size=[1280, 720],
        pad_to=pad_to,
    )


def test_parameter_count_and_missing_padding_gradients():
    assert sum(p.numel() for p in Lifter().parameters()) == 1_633_731
    model = Lifter(ModelConfig(width=64, depth=4))
    assert sum(p.numel() for p in model.parameters()) == 212_963
    f = features(pad_to=12)
    f.ball_uv[~f.ball_observed] = torch.nan
    f.court_uv[f.padding] = torch.nan
    output = model(f)
    assert torch.isfinite(output).all()
    assert not output[f.padding].any()
    target = torch.ones_like(output)
    target[f.padding] = torch.nan
    loss = position_loss(output, target, ~f.padding, f.padding)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.missing_ball.grad is not None
    assert model.missing_ball.grad.abs().sum() > 0
    # Padding has no influence on real outputs.
    torch.testing.assert_close(output[:, :8], model(features()), atol=2e-5, rtol=2e-5)


def test_features_time_precision_and_unclipped_coordinates():
    f = features(4, pad_to=7)
    np.testing.assert_array_equal(f.t_s[0, :4].numpy(), 1000 + np.arange(4) * 0.023)
    np.testing.assert_array_equal(
        f.time_bins[0, :4], np.rint((f.t_s[0, :4].numpy() - 1000) / 0.002)
    )
    torch.testing.assert_close(f.ball_uv[0, 0], torch.tensor([0.5, 0.25]))
    torch.testing.assert_close(f.court_uv[0, 0, 0], torch.tensor([1.0, 0.5]))
    assert f.padding.sum() == 3
    assert batch_features([f, f]).t_s.shape == (2, 7)
    # Different real times may share one bin, with no removed output rows.
    g = build_features(
        t_s=[0.0, 0.0004, 0.0008],
        ball_uv=np.ones((3, 2)),
        ball_observed=np.ones(3, dtype=bool),
        court_uv=np.ones((3, 14, 2)),
        camera_valid=np.ones(3, dtype=bool),
        image_size=[1, 1],
    )
    assert g.time_bins.tolist() == [[0, 0, 0]]
    assert g.t_s.shape == (1, 3)


def test_evidence_and_camera_rejection():
    model = Lifter(ModelConfig(width=16, depth=2))
    f = features()
    f.ball_observed[:] = False
    with pytest.raises(ValueError, match="no ball evidence"):
        model(f)
    f = features()
    f.camera_valid[0, 3] = False
    with pytest.raises(ValueError, match="invalid-camera"):
        model(f)
    f = features()
    f.ball_uv[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        model(f)


def test_attention_prefix_masks_and_rope():
    model = Lifter(ModelConfig(width=16, depth=3, dropout=0.2)).eval()
    f = features(pad_to=10)
    captured = []
    hooks = [
        b.register_forward_pre_hook(lambda _, args: captured.append(args[2].clone()))
        for b in model.blocks
    ]
    first = model(f)
    second = model(f)
    torch.testing.assert_close(first, second)
    assert torch.equal(captured[0], f.ball_observed & ~f.padding)
    assert all(torch.equal(mask, ~f.padding) for mask in captured[1:3])
    for hook in hooks:
        hook.remove()
    x = torch.randn(2, 4, 8, 4)
    rope = RotaryEmbedding(4)
    bins = torch.arange(8).expand(2, 8)
    torch.testing.assert_close(rope(x, bins).square().sum(-1), x.square().sum(-1))


def test_loss_includes_missing_and_excludes_nan_padding():
    pred = torch.tensor([[[1.0, 2.0, 2.0], [0.0, 0.0, 0.0], [float("nan")] * 3]])
    truth = torch.zeros_like(pred)
    valid = torch.tensor([[True, True, False]])
    assert position_loss(pred, truth, valid, ~valid).item() == 4.5
    with pytest.raises(ValueError, match="no valid"):
        position_loss(pred, truth, ~valid & valid, ~valid)


def test_complete_window_coverage_invalid_boundaries_and_weighting():
    camera = np.ones(601, dtype=bool)
    camera[260:270] = False
    windows = make_windows(camera)
    assert windows == [
        Window(0, 250),
        Window(125, 260),
        Window(270, 520),
        Window(395, 601),
    ]
    merger = WindowMerger(len(camera))
    for w in windows:
        merger.add(w, np.column_stack([np.arange(w.start, w.stop)] * 3))
    xyz, covered, disagreement = merger.finish()
    np.testing.assert_array_equal(covered, camera)
    np.testing.assert_allclose(
        xyz[covered], np.column_stack([np.flatnonzero(camera)] * 3)
    )
    np.testing.assert_allclose(disagreement[covered], 0)
    assert np.isnan(xyz[~camera]).all()
    merger = WindowMerger(4)
    merger.add(Window(0, 3), np.zeros((3, 3)))
    merger.add(Window(1, 4), np.ones((3, 3)) * 3)
    xyz, covered, disagreement = merger.finish()
    np.testing.assert_allclose(xyz[:, 0], [0, 1, 2, 3])
    assert covered.all() and (disagreement[1:3] > 0).all()


def test_raw_predictor_preserves_invalid_camera_and_evidence_free_rows():
    from tennis3d.inference.predict import predict_view

    n = 270
    arrays = {
        "t_s": np.arange(n, dtype=np.float64) * 0.0004,
        "ball_uv": np.full((n, 2), np.nan),
        "ball_observed": np.zeros(n, dtype=bool),
        "court_uv": np.ones((n, 14, 2)),
        "camera_valid": np.ones(n, dtype=bool),
        "label_valid": np.ones(n, dtype=bool),
        "image_size": np.array([1280, 720]),
    }
    arrays["camera_valid"][10] = False
    arrays["ball_observed"][0] = True
    arrays["ball_uv"][0] = [300, 200]
    model = Lifter(ModelConfig(width=16, depth=2))
    xyz, covered, _ = predict_view(model, arrays)
    assert model.training  # Inference restores the caller's mode.
    np.testing.assert_array_equal(covered, np.arange(n) < 10)
    assert xyz.shape == (n, 3)
    assert np.isnan(xyz[10:]).all()
    with pytest.raises(ValueError, match="no ball evidence"):
        predict_view(model, arrays, require_complete=True)
    arrays["ball_observed"][:] = True
    arrays["ball_uv"][:] = [300, 200]
    xyz, covered, _ = predict_view(model, arrays, require_complete=True)
    np.testing.assert_array_equal(covered, arrays["camera_valid"])
    assert np.isfinite(xyz[covered]).all()
