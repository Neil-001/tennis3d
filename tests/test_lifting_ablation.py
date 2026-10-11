"""Architecture, physical time and fresh AdamW contracts for the frozen ablation."""

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tennis3d.lifting.features import batch_features, build_features
from tennis3d.lifting.losses import vertical_radial_position_loss
from tennis3d.lifting.model import ModelConfig, build_lifter
from tennis3d.lifting.settings import TrainConfig
from tennis3d.lifting.train import ScheduledAdamW, _cosine_policy, build_optimizer


def lstm_config(**kwargs):
    return ModelConfig(
        architecture_version="stage4-lstm-v1",
        width=16,
        depth=2,
        lstm_hidden_size=8,
        gated_head=True,
        **kwargs,
    )


def features(times, pad_to=None):
    n = len(times)
    return build_features(
        t_s=times,
        ball_uv=np.ones((n, 2)),
        ball_observed=np.arange(n) % 2 == 0,
        court_uv=np.ones((n, 14, 2)),
        camera_valid=np.ones(n, dtype=bool),
        image_size=[2, 2],
        pad_to=pad_to,
    )


def test_actual_production_matched_count_and_factory_defaults():
    cfg = ModelConfig(
        architecture_version="stage4-lstm-v1",
        width=176,
        depth=2,
        lstm_hidden_size=200,
        gated_head=True,
    )
    model = build_lifter(cfg)
    assert sum(p.numel() for p in model.parameters()) == 1769875
    assert abs(1769875 / 1759871 - 1) < 0.1
    assert model.temporal.num_layers == 2 and model.temporal.bidirectional
    assert not model.blocks
    assert build_lifter().config.architecture_version == "legacy-v1"
    with pytest.raises(ValueError, match="explicit AdamW"):
        TrainConfig(model=cfg)
    with pytest.raises(ValueError, match="depth 2"):
        ModelConfig(architecture_version="stage4-lstm-v1")


def test_mixed_length_batch_padding_has_no_context_or_gradient_effect():
    torch.manual_seed(7)
    model = build_lifter(lstm_config()).eval()
    short = features([1000.0, 1000.013, 1000.031])
    long = features(np.arange(7) * 0.02)
    padded = features([1000.0, 1000.013, 1000.031], pad_to=7)
    padded.ball_uv[~padded.ball_observed] = torch.nan
    padded.court_uv[padded.padding] = torch.nan
    padded.t_s[padded.padding] = torch.nan
    batched = batch_features([padded, long])
    prediction = model(batched)
    torch.testing.assert_close(prediction[:1, :3], model(short), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(prediction[1:], model(long), rtol=1e-5, atol=1e-6)
    assert not prediction[batched.padding].any()
    target = torch.zeros_like(prediction)
    target[batched.padding] = torch.nan
    vertical_radial_position_loss(
        prediction, target, ~batched.padding, batched.padding
    ).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert model.missing_ball.grad is not None
    assert model.missing_ball.grad.abs().sum() > 0


def test_raw_time_intervals_survive_shared_bins_and_origin_shift():
    torch.manual_seed(9)
    model = build_lifter(lstm_config()).eval()
    original = features([0.0, 0.0004, 0.0008])
    changed = features([0.0, 0.0002, 0.0008])
    assert torch.equal(original.time_bins, changed.time_bins)
    assert not torch.equal(model(original), model(changed))
    shifted = replace(original, t_s=original.t_s + 1000.0)
    torch.testing.assert_close(model(original), model(shifted), rtol=1e-6, atol=1e-7)
    # A future observation affects a past output in the noncausal encoder.
    future = replace(original, ball_uv=original.ball_uv.clone())
    future.ball_uv[0, -1] += 1
    assert not torch.equal(model(original)[:, 0], model(future)[:, 0])


def test_lstm_rejects_camera_boundaries_invalid_time_and_padding_holes():
    model = build_lifter(lstm_config())
    f = features([0.0, 0.02, 0.04], pad_to=5)
    f.camera_valid[0, 1] = False
    with pytest.raises(ValueError, match="invalid-camera"):
        model(f)
    f.camera_valid[0, 1] = True
    f.t_s[0, 1] = torch.nan
    with pytest.raises(ValueError, match="timestamps"):
        model(f)
    f.t_s[0, 1] = 0.02
    f.padding[0, 1] = True
    with pytest.raises(ValueError, match="right padding"):
        model(f)


def test_adamw_matches_pytorch_with_persistent_schedule_and_decay():
    model = build_lifter(lstm_config())
    reference = build_lifter(lstm_config())
    reference.load_state_dict(model.state_dict())
    optimizer = build_optimizer(model, 0.001, kind="adamw")
    assert isinstance(optimizer, ScheduledAdamW)
    optimizer.param_groups[0]["lr_schedule"] = _cosine_policy(0.001, 4)
    expected = torch.optim.AdamW(
        reference.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999)
    )
    for _ in range(3):
        for a, b in zip(model.parameters(), reference.parameters()):
            a.grad = torch.ones_like(a) * 0.1
            b.grad = a.grad.clone()
        optimizer.step()
        expected.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"]
        expected.step()
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert optimizer.param_groups[0]["lr_schedule"]["step"] == 3
    restored = build_optimizer(model, 0.001, optimizer.state_dict(), kind="adamw")
    restored.load_state_dict(optimizer.state_dict())
    assert restored.param_groups[0]["algorithm"] == "adamw"
    assert restored.param_groups[0]["lr_schedule"]["step"] == 3
