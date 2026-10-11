"""Selected recipe and recovery contracts, using only small CPU fixtures."""

import copy
import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tennis3d.cli import main
from tennis3d.config import load_config
from tennis3d.lifting.checkpoint import load_lifter, read_checkpoint, save_checkpoint
from tennis3d.lifting.features import build_features
from tennis3d.lifting.losses import vertical_radial_position_loss
from tennis3d.lifting.model import Block, Lifter, ModelConfig
from tennis3d.lifting.settings import TrainConfig
from tennis3d.lifting.train import MuonAdam, TrainControl, _cosine_policy, _cosine_rate

trainer = importlib.import_module("tennis3d.lifting.train")
FIXTURE = Path(__file__).parent / "fixtures" / "lifting"


def recipe_config():
    return load_config(TrainConfig, Path(__file__).parents[1] / "configs/lifter.yaml")


def fixture_config(**changes):
    return TrainConfig(
        train_manifest=str(FIXTURE / "train.manifest.json"),
        val_manifest=str(FIXTURE / "val.manifest.json"),
        model=ModelConfig(width=16, depth=2, dropout=0.1),
        max_steps=5,
        min_frames=20,
        max_frames=32,
        batch_size=1,
        device="cpu",
    ).model_copy(update=changes)


def no_evaluation(**changes):
    return TrainControl(
        validate_during_training=False, evaluate_at_end=False, **changes
    )


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_state(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_state(x, y)
    else:
        assert a == b


def test_selected_model_cpu_backward_partition_and_ema_load(tmp_path):
    cfg = recipe_config()
    assert (cfg.batch_size, cfg.ema_decay, cfg.learning_rate) == (48, 0.99, 0.0016)
    assert cfg.model.model_dump() == {
        "architecture_version": "stage3-v1",
        "lstm_hidden_size": 200,
        "width": 176,
        "depth": 4,
        "heads": 11,
        "mlp_ratio": 3,
        "dropout": 0.0,
        "gated_ffn": True,
        "gated_head": True,
        "temporal_bias": True,
        "schedule_steps": 2000,
        "outward_radial_brake": 0.5,
        "batched_muon": True,
    }
    model = Lifter(cfg.model)
    assert sum(p.numel() for p in model.parameters()) == 1_759_871
    f = build_features(
        t_s=np.arange(8) * 0.023,
        ball_uv=np.ones((8, 2)) * [640, 180],
        ball_observed=np.arange(8) % 3 == 0,
        court_uv=np.ones((8, 14, 2)) * [1280, 360],
        camera_valid=np.ones(8, dtype=bool),
        image_size=[1280, 720],
        pad_to=12,
    )
    optimizer = MuonAdam(model, cfg.learning_rate)
    muon, adam = optimizer.param_groups
    hidden = {id(p) for p in muon["params"]}
    auxiliary = {id(p) for p in adam["params"]}
    assert not hidden & auxiliary
    assert hidden | auxiliary == {id(p) for p in model.parameters()}
    for name, p in model.named_parameters():
        assert (id(p) in hidden) == (name.startswith("blocks.") and p.ndim == 2)
    assert len(muon["qkv_indices"]) == len(muon["gate_indices"]) == 4
    assert muon["batched_muon"] and muon["outward_radial_brake"] == 0.5
    assert [g["lr_schedule"]["peak_lr"] for g in (muon, adam)] == pytest.approx(
        [0.0024, 0.0016]
    )
    prediction = model(f)
    assert prediction.shape == (1, 12, 3)
    assert not prediction[f.padding].any()
    loss = vertical_radial_position_loss(
        prediction, torch.ones_like(prediction), ~f.padding, f.padding
    )
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg.ema_decay)
    )
    ema.update_parameters(model)
    with torch.no_grad():
        model.ball.weight.add_(1)
    ema.update_parameters(model)
    path = tmp_path / "selected.pt"
    save_checkpoint(
        path,
        {
            "schema_version": "1.0",
            "world_scale_m": 10.0,
            "config": cfg.model_dump(),
            "ema": ema.state_dict(),
        },
    )
    loaded, _ = load_lifter(path)
    assert not loaded.training
    equal_state(loaded.state_dict(), ema.module.state_dict())
    torch.testing.assert_close(loaded(f), ema.module.eval()(f), rtol=0, atol=0)
    assert not torch.equal(loaded.ball.weight, model.ball.weight)


def test_weighted_radial_loss_masks_and_units():
    prediction = torch.tensor(
        [[[0.3, 0.4, 0.2], [float("nan")] * 3]], requires_grad=True
    )
    valid = torch.tensor([[True, False]])
    target = torch.zeros_like(prediction)
    expected = 0.1 * (math.sqrt(0.3**2 + 0.4**2 + (1.5 * 0.2) ** 2 + 0.01**2) - 0.01)
    loss = vertical_radial_position_loss(prediction, target, valid, ~valid)
    assert float(loss.detach()) == pytest.approx(expected)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert not prediction.grad[:, 1].any()


def test_original_config_retains_relu_and_full_residuals():
    assert ModelConfig.model_fields["architecture_version"].default == "legacy-v1"
    assert ModelConfig().architecture_version == "legacy-v1"
    assert TrainConfig().model.architecture_version == "legacy-v1"
    cfg = ModelConfig.model_validate({"width": 16, "depth": 2})
    assert cfg.architecture_version == "legacy-v1"
    block = Block(cfg).eval()
    assert isinstance(block.ffn[1], torch.nn.ReLU)
    x = torch.randn(1, 5, 16)
    bins = torch.arange(5)[None]
    keys = torch.ones(1, 5, dtype=torch.bool)
    residual = x + block.attention(block.norm1(x), bins, keys)
    expected = residual + block.ffn(block.norm2(residual))
    torch.testing.assert_close(block(x, bins, keys), expected, rtol=0, atol=0)
    assert ModelConfig(gated_ffn=False).architecture_version == "stage3-v1"
    with pytest.raises(ValueError, match="legacy architecture"):
        ModelConfig(architecture_version="legacy-v1", gated_ffn=True)


def test_long_cosine_schedule_and_optimizer_resume():
    policy = _cosine_policy(0.0016, 10_000)
    assert policy["warmup_steps"] == 200
    for completed, expected in [
        (1, 0.0016 / 200),
        (200, 0.0016),
        (10_000, 0.00008),
        (12_000, 0.00008),
    ]:
        policy["step"] = completed - 1
        assert _cosine_rate(policy) == pytest.approx(expected)
    policy["step"] = 2100
    expected = 0.0016 * (
        0.05 + 0.95 * (1 + math.cos(math.pi * (2101 - 200) / 9800)) / 2
    )
    assert _cosine_rate(policy) == pytest.approx(expected)
    assert _cosine_rate(policy) > 0.0016 * 0.8
    model = Lifter(
        ModelConfig(
            width=8,
            depth=2,
            gated_ffn=True,
            batched_muon=True,
            outward_radial_brake=0.5,
        )
    )
    optimizer = MuonAdam(model, 0.0016)
    for group in optimizer.param_groups:
        group["lr_schedule"] = _cosine_policy(group["lr"], 10_000)
        group["lr_schedule"]["step"] = 2100
    clone = copy.deepcopy(model)
    resumed = MuonAdam(clone, 0.0016)
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for a, b in zip(model.parameters(), clone.parameters()):
        a.grad = torch.ones_like(a) * 0.01
        b.grad = a.grad.clone()
    optimizer.step()
    resumed.step()
    equal_state(model.state_dict(), clone.state_dict())
    equal_state(optimizer.state_dict(), resumed.state_dict())
    assert optimizer.param_groups[1]["lr"] == pytest.approx(expected)


def fixed_validation(model, manifest, **kwargs):
    return {
        "manifest_sha256": manifest.manifest_sha256,
        "source_identity": manifest.identity,
        "modes": {"corrupted": {"val_xyz_m": 1.0}},
    }


@pytest.mark.parametrize(
    "cadence,validate,expected",
    [
        (None, True, [0, 3, 5]),
        (2, True, [0, 2, 3, 4, 5]),
        (2, False, [0, 2, 4, 5]),
        (3, True, [0, 3, 5]),
    ],
)
def test_recovery_cadence_independent_of_validation(
    tmp_path, monkeypatch, cadence, validate, expected
):
    writes = []
    save = trainer.save_checkpoint

    def counted(path, state):
        if path.name == "last.pt":
            writes.append(state["step"])
        save(path, state)

    monkeypatch.setattr(trainer, "save_checkpoint", counted)
    monkeypatch.setattr(trainer, "evaluate_model", fixed_validation)
    cfg = fixture_config(checkpoint_every=cadence, validation_every=3)
    trainer.train(
        cfg,
        tmp_path / "run",
        control=TrainControl(validate_during_training=validate, evaluate_at_end=False),
    )
    assert writes == expected
    history = tmp_path / "run" / "validation-history.jsonl"
    if validate:
        assert [
            json.loads(line)["step"] for line in history.read_text().splitlines()
        ] == [3]
    else:
        assert not history.exists()


def test_selected_exact_recovery_after_crash_and_cadence_change(tmp_path):
    model = recipe_config().model.model_copy(
        update={"dropout": 0.1, "schedule_steps": 10_000}
    )
    cfg = fixture_config(model=model, checkpoint_every=2, log_every=2)
    full = trainer.train(cfg, tmp_path / "full", control=no_evaluation())

    def crash_after_uncheckpointed_step(row):
        if row.get("step") == 3 and "loss" in row:
            raise RuntimeError("fixture crash")

    with pytest.raises(RuntimeError, match="fixture crash"):
        trainer.train(
            cfg,
            tmp_path / "crashed",
            control=no_evaluation(on_update=crash_after_uncheckpointed_step),
        )
    path = tmp_path / "crashed" / "checkpoints" / "last.pt"
    assert read_checkpoint(path)["step"] == 2
    resumed = trainer.train(
        cfg.model_copy(update={"checkpoint_every": 1}),
        tmp_path / "crashed",
        resume=path,
        control=no_evaluation(),
    )
    a, b = (
        read_checkpoint(full["last_checkpoint"]),
        read_checkpoint(resumed["last_checkpoint"]),
    )
    for key in ("model", "ema", "optimizer", "sampler", "rng"):
        equal_state(a[key], b[key])
    assert a["optimizer"]["param_groups"][0]["lr_schedule"]["total_steps"] == 10_000
    rows = [
        json.loads(row)
        for row in (tmp_path / "full" / "loss.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in rows] == [2, 4]
    assert [r["windows"] for r in rows] == [2, 4]
    for row in rows:
        assert len(row["learning_rates"]) == len(row["optimizer_stability"]) == 2
        assert all(
            g["weight_rms"] > 0 and g["first_moment_rms"] > 0
            for g in row["optimizer_stability"]
        )


def test_legacy_adam_checkpoint_resumes_twice_with_original_loss_and_best(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(trainer, "evaluate_model", fixed_validation)
    original_builder = trainer.build_optimizer
    original_objective = trainer._training_objective
    monkeypatch.setattr(trainer, "_training_objective", lambda state: "xyz-squared-v1")
    monkeypatch.setattr(
        trainer,
        "build_optimizer",
        lambda model, lr, saved=None, **kwargs: torch.optim.Adam(
            model.parameters(), lr=lr
        ),
    )
    cfg = fixture_config(validation_every=2)
    full = trainer.train(cfg, tmp_path / "full")
    first = trainer.train(
        cfg, tmp_path / "legacy", control=TrainControl(stop_after_steps=2)
    )
    for path in Path(first["last_checkpoint"]).parent.glob("*.pt"):
        state = read_checkpoint(path)
        state.pop("training_objective")
        state["provenance"]["source_files"].update(
            {
                "tennis3d/lifting/train.py": "8f22eaad83de14813f43be55961eb345e1c777a2b9373b5957211db2087eff54",
                "tennis3d/lifting/losses.py": "82d4ab351ae28ad33021887d85c508ee086ce86cf7e2c3e833ed375ae0fae00d",
                "tennis3d/lifting/model.py": "572e55eebcee2822e328ed0cbc8e9dc5d2cda606010947fce87770ef0ae26cb6",
            }
        )
        state["config"].pop("checkpoint_every")
        state["config"]["model"] = {
            k: v
            for k, v in state["config"]["model"].items()
            if k in {"width", "depth", "heads", "mlp_ratio", "dropout"}
        }
        state.pop("dataset_identity")
        save_checkpoint(path, state)
    monkeypatch.setattr(trainer, "build_optimizer", original_builder)
    monkeypatch.setattr(trainer, "_training_objective", original_objective)
    for limit in (1, 2):
        first = trainer.train(
            cfg.model_copy(update={"checkpoint_every": 1}),
            tmp_path / "legacy",
            resume=first["last_checkpoint"],
            control=TrainControl(stop_after_steps=limit),
        )
    a, b = (
        read_checkpoint(full["last_checkpoint"]),
        read_checkpoint(first["last_checkpoint"]),
    )
    for key in ("model", "ema", "optimizer", "sampler", "rng"):
        equal_state(a[key], b[key])
    assert read_checkpoint(first["best_checkpoint"])["step"] == 2
    assert (
        read_checkpoint(first["last_checkpoint"])["training_objective"]
        == "xyz-squared-v1"
    )
    assert (
        "mean squared XYZ"
        in json.loads(
            (tmp_path / "legacy" / "loss.jsonl").read_text().splitlines()[-1]
        )["loss_units"]
    )


def test_unmarked_selected_objective_and_ambiguous_adam_rejection(tmp_path):
    cfg = fixture_config()
    result = trainer.train(
        cfg, tmp_path / "first", control=no_evaluation(stop_after_steps=2)
    )
    state = read_checkpoint(result["last_checkpoint"])
    state.pop("training_objective")
    state["provenance"]["source_files"].update(
        {
            "tennis3d/lifting/train.py": "6d8da83710ce1defe0c104c2b1fbe530d3feb95746230ce22e6c9a6d7b254342",
            "tennis3d/lifting/losses.py": "a5a4004548a25da81c45e924023a4604a0e35277bf12c220d06a53218a9a1a92",
            "tennis3d/lifting/model.py": "9c90a9a7e4bd06f56c9edec4f06058f2c9b4a111624181856915844733ccf8cc",
        }
    )
    save_checkpoint(result["last_checkpoint"], state)
    resumed = trainer.train(
        cfg,
        tmp_path / "selected-resume",
        resume=result["last_checkpoint"],
        control=no_evaluation(),
    )
    assert (
        read_checkpoint(resumed["last_checkpoint"])["training_objective"]
        == "xyz-weighted-radial-v1"
    )
    state["optimizer"] = torch.optim.Adam(Lifter(cfg.model).parameters()).state_dict()
    state["provenance"]["source_files"]["tennis3d/lifting/train.py"] = "0" * 64
    state["provenance"]["code_revision"] = "7f664cba7900aaebb0bab37294f516b73cb3bd43"
    save_checkpoint(result["last_checkpoint"], state)
    with pytest.raises(ValueError, match="ambiguous historical recipe"):
        trainer.train(
            cfg,
            tmp_path / "ambiguous",
            resume=result["last_checkpoint"],
            control=no_evaluation(),
        )
    assert not (tmp_path / "ambiguous" / "checkpoints" / "last.pt").exists()


def test_cli_checkpoint_override_on_legacy_resume(tmp_path, monkeypatch, capsys):
    cfg = fixture_config().model_dump()
    cfg.pop("checkpoint_every")
    path = tmp_path / "legacy.pt"
    save_checkpoint(
        path, {"schema_version": "1.0", "world_scale_m": 10.0, "config": cfg}
    )
    captured = []

    def fake_train(config, *args, **kwargs):
        captured.append(config)
        return {"step": 5}

    monkeypatch.setattr(trainer, "train", fake_train)
    assert main(["train", "--resume", str(path), "--checkpoint-every", "50"]) == 0
    assert captured[0].checkpoint_every == 50
    assert json.loads(capsys.readouterr().out)["step"] == 5
    assert main(["train", "--resume", str(path), "--checkpoint-every", "0"]) == 2
    assert "Invalid training input" in capsys.readouterr().out


def test_selected_tiny_train_fixture_overfit_with_full_horizon():
    """One fixed genuine train fragment, no validation selection or rate tuning."""
    from tennis3d.lifting.data import ViewDataset
    from tennis3d.lifting.features import WORLD_SCALE_M

    torch.manual_seed(42)
    cfg = recipe_config()
    cfg = cfg.model_copy(
        update={"model": cfg.model.model_copy(update={"schedule_steps": 83334})}
    )
    with ViewDataset(FIXTURE / "train.manifest.json") as dataset:
        arrays = dataset[0]
    f = build_features(
        t_s=arrays["t_s"],
        ball_uv=arrays["ball_uv"],
        ball_observed=arrays["ball_observed"],
        court_uv=arrays["court_uv"],
        camera_valid=arrays["camera_valid"],
        image_size=arrays["image_size"],
    )
    target = torch.as_tensor(arrays["xyz_m"] / WORLD_SCALE_M, dtype=torch.float32)[None]
    valid = torch.as_tensor(arrays["label_valid"])[None]
    model = Lifter(cfg.model)
    optimizer = trainer.build_optimizer(model, cfg.learning_rate)
    for group in optimizer.param_groups:
        group["lr_schedule"] = _cosine_policy(group["lr"], 83334)
    initial = float(
        vertical_radial_position_loss(model(f), target, valid, f.padding).detach()
    )
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = vertical_radial_position_loss(model(f), target, valid, f.padding)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.gradient_clip, error_if_nonfinite=True
        )
        optimizer.step()
    final = float(
        vertical_radial_position_loss(model(f), target, valid, f.padding).detach()
    )
    print(
        f"selected tiny fixture: initial={initial:.9g} final={final:.9g} updates=200 horizon=83334"
    )
    assert final < 0.1 * initial
