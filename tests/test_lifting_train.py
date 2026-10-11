"""Real reader/corruption integration, exact resume and fixed-time trial controls."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tennis3d.artifacts import read_json
from tennis3d.evaluation.synthetic import evaluate_checkpoint
from tennis3d.lifting.checkpoint import read_checkpoint
from tennis3d.lifting.model import ModelConfig
from tennis3d.lifting.settings import TrainConfig
from tennis3d.lifting.train import TrainControl, train

FIXTURE = Path(__file__).parent / "fixtures" / "lifting"


def config(**overrides):
    return TrainConfig(
        train_manifest=str(FIXTURE / "train.manifest.json"),
        val_manifest=str(FIXTURE / "val.manifest.json"),
        model=ModelConfig(width=16, depth=2, dropout=0.1),
        max_steps=4,
        min_frames=20,
        max_frames=32,
        batch_size=1,
        device="cpu",
        **overrides,
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


@pytest.mark.parametrize(
    "architecture,optimizer",
    [
        ("legacy-v1", "muon-adam"),
        ("stage3-v1", "muon-adam"),
        ("stage3-v1", "adamw"),
        ("stage4-lstm-v1", "adamw"),
    ],
)
def test_exact_optimizer_ema_sampler_corruption_resume(
    tmp_path, architecture, optimizer
):
    cfg = config().model_copy(
        update={
            "model": ModelConfig(
                architecture_version=architecture,
                width=16,
                depth=2,
                lstm_hidden_size=8,
                dropout=0.1,
            ),
            "optimizer": optimizer,
        }
    )
    uninterrupted = train(
        cfg,
        tmp_path / "full",
        control=TrainControl(validate_during_training=False, evaluate_at_end=False),
    )
    first = train(
        cfg,
        tmp_path / "resumed",
        control=TrainControl(
            stop_after_steps=2, validate_during_training=False, evaluate_at_end=False
        ),
    )
    resumed = train(
        cfg,
        tmp_path / "resumed",
        resume=first["last_checkpoint"],
        control=TrainControl(validate_during_training=False, evaluate_at_end=False),
    )
    a, b = (
        read_checkpoint(uninterrupted["last_checkpoint"]),
        read_checkpoint(resumed["last_checkpoint"]),
    )
    for field in ("model", "ema", "optimizer", "sampler", "rng"):
        equal_state(a[field], b[field])
    from tennis3d.lifting.checkpoint import load_lifter

    loaded, _ = load_lifter(resumed["last_checkpoint"])
    equal_state(
        loaded.state_dict(),
        {
            key.removeprefix("module."): value
            for key, value in b["ema"].items()
            if key.startswith("module.")
        },
    )
    assert resumed["step"] == 4
    assert resumed["train_seconds"] > first["train_seconds"]
    assert resumed["job_seconds"] > first["job_seconds"]


def test_trial_disables_step_limit_and_evaluates_final_ema_once(tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module("tennis3d.lifting.train")
    evaluate = module.evaluate_model
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(module, "evaluate_model", counted)
    cfg = config().model_copy(
        update={"max_steps": 1, "validation_every": 1, "patience": 1}
    )
    updates = []
    result = train(
        cfg,
        tmp_path / "trial",
        control=TrainControl(
            use_step_limit=False,
            use_early_stopping=False,
            validate_during_training=False,
            stop_after_steps=3,
            on_update=updates.append,
        ),
    )
    assert result["step"] == 3 and len(calls) == 1
    assert {u["phase"] for u in updates} == {
        "setup",
        "training",
        "evaluation",
        "checkpoint",
        "complete",
    }
    assert len([u for u in updates if "loss" in u]) == 3
    report = read_json(tmp_path / "trial" / "validation.json")
    again = evaluate_checkpoint(
        result["last_checkpoint"], cfg.val_manifest, device="cpu"
    )
    assert report["step"] == again["checkpoint_step"] == 3
    assert report["modes"] == again["modes"]
    assert result["execution_limits"]["max_steps"] is None


def test_time_stop_and_resume_contract_rejection(tmp_path):
    cfg = config(max_train_seconds=0.000001)
    result = train(cfg, tmp_path / "timed", control=TrainControl(evaluate_at_end=False))
    assert result["step"] == 1 and result["stop_reason"] == "train_budget"
    assert result["train_overrun_seconds"] > 0
    resumed = train(
        cfg,
        tmp_path / "timed",
        resume=result["last_checkpoint"],
        control=TrainControl(evaluate_at_end=False),
    )
    assert resumed["step"] == 1
    changed = cfg.model_copy(update={"seed": 43})
    with pytest.raises(ValueError, match="preserve"):
        train(changed, tmp_path / "changed", resume=result["last_checkpoint"])


def test_older_checkpoint_keeps_later_consumed_budget(tmp_path):
    import shutil

    cfg = config()
    first = train(
        cfg,
        tmp_path / "original",
        control=TrainControl(
            stop_after_steps=2, validate_during_training=False, evaluate_at_end=False
        ),
    )
    older = tmp_path / "original" / "checkpoints" / "older.pt"
    shutil.copyfile(first["last_checkpoint"], older)
    latest = train(
        cfg,
        tmp_path / "original",
        resume=first["last_checkpoint"],
        control=TrainControl(validate_during_training=False, evaluate_at_end=False),
    )
    recovered = train(
        cfg,
        tmp_path / "recovered",
        resume=older,
        control=TrainControl(
            stop_after_steps=0, validate_during_training=False, evaluate_at_end=False
        ),
    )
    assert recovered["step"] == 2
    assert recovered["train_seconds"] == latest["train_seconds"]
    assert recovered["job_seconds"] >= latest["job_seconds"]


def test_requested_stop_skips_boundary_validation(tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module("tennis3d.lifting.train")
    control = TrainControl()

    def stop_on_loss(update):
        if "loss" in update:
            control.stop_requested = True

    def forbidden(*args, **kwargs):
        pytest.fail("validation started after a graceful stop request")

    control.on_update = stop_on_loss
    monkeypatch.setattr(module, "evaluate_model", forbidden)
    cfg = config().model_copy(update={"validation_every": 1})
    result = train(cfg, tmp_path / "stopped", control=control)
    assert result["step"] == 1 and result["stop_reason"] == "requested"
    assert read_checkpoint(result["last_checkpoint"])["step"] == 1


def test_resume_keeps_earlier_best_and_does_not_repeat_patience(tmp_path, monkeypatch):
    import importlib

    from tennis3d.artifacts import file_sha256

    module = importlib.import_module("tennis3d.lifting.train")
    calls = []

    def fixed_score(model, manifest, **kwargs):
        calls.append(1)
        return {
            "manifest_sha256": manifest.manifest_sha256,
            "source_identity": manifest.identity,
            "modes": {"corrupted": {"val_xyz_m": 1.0}},
        }

    monkeypatch.setattr(module, "evaluate_model", fixed_score)
    cfg = config().model_copy(update={"validation_every": 1, "patience": 1})
    first = train(cfg, tmp_path / "first")
    assert first["step"] == 2
    original_best = file_sha256(first["best_checkpoint"])
    resumed = train(cfg, tmp_path / "resumed", resume=first["last_checkpoint"])
    assert resumed["step"] == 2 and resumed["stop_reason"] == "validation_plateau"
    assert len(calls) == 2
    assert resumed["best_checkpoint"] is not None
    assert file_sha256(resumed["best_checkpoint"]) == original_best
    assert read_checkpoint(resumed["best_checkpoint"])["step"] == 1
    assert read_checkpoint(resumed["last_checkpoint"])["stale_validations"] == 1
    continued = train(
        cfg,
        tmp_path / "resumed",
        resume=resumed["last_checkpoint"],
        control=TrainControl(use_early_stopping=False),
    )
    assert continued["step"] == 4
    assert file_sha256(continued["best_checkpoint"]) == original_best
    Path(first["best_checkpoint"]).unlink()
    with pytest.raises(ValueError, match="selected best"):
        train(cfg, tmp_path / "missing-best", resume=first["last_checkpoint"])


def test_cli_second_interrupt_aborts_graceful_stop(tmp_path, monkeypatch):
    import argparse
    import importlib
    import signal

    from tennis3d.lifting import cli

    module = importlib.import_module("tennis3d.lifting.train")
    previous = signal.getsignal(signal.SIGINT)
    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupted_train(*args, control, **kwargs):
        assert not control.stop_requested
        signal.raise_signal(signal.SIGINT)
        assert control.stop_requested
        signal.raise_signal(signal.SIGINT)
        pytest.fail("second SIGINT did not interrupt")

    monkeypatch.setattr(module, "train", interrupted_train)
    monkeypatch.setattr(cli, "load_config", lambda *args, **kwargs: config())
    args = argparse.Namespace(
        config=None,
        output=tmp_path / "run",
        resume=None,
        device=None,
        max_steps=None,
        max_train_seconds=None,
    )
    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        with pytest.raises(KeyboardInterrupt):
            cli.train_command(args)
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
        assert signal.getsignal(signal.SIGTERM) == previous_term
    finally:
        signal.signal(signal.SIGINT, previous)


def test_setup_charge_enforces_cap_before_fresh_and_resumed_updates(tmp_path):
    from tennis3d.artifacts import write_json

    cfg = config().model_copy(update={"max_job_seconds": 172800.0})
    fresh = train(
        cfg,
        tmp_path / "fresh",
        control=TrainControl(
            setup_job_seconds=172801.0,
            validate_during_training=False,
            evaluate_at_end=False,
        ),
    )
    assert fresh["step"] == 0 and fresh["stop_reason"] == "job_budget"
    assert read_checkpoint(fresh["last_checkpoint"])["budget"]["job_seconds"] >= 172801
    first = train(
        cfg,
        tmp_path / "continued",
        control=TrainControl(
            stop_after_steps=1, validate_during_training=False, evaluate_at_end=False
        ),
    )
    ledger = read_json(tmp_path / "continued/budget.json")
    ledger["job_seconds"] = 172799.0
    write_json(tmp_path / "continued/budget.json", ledger, overwrite=True)
    resumed = train(
        cfg,
        tmp_path / "continued",
        resume=first["last_checkpoint"],
        control=TrainControl(
            setup_job_seconds=2.0, validate_during_training=False, evaluate_at_end=False
        ),
    )
    assert resumed["step"] == first["step"] == 1
    assert resumed["stop_reason"] == "job_budget" and resumed["job_seconds"] >= 172801
    # Default zero charge carries the exhausted ledger instead of resetting it.
    again = train(
        cfg,
        tmp_path / "continued",
        resume=resumed["last_checkpoint"],
        control=TrainControl(validate_during_training=False, evaluate_at_end=False),
    )
    assert again["step"] == 1 and again["job_seconds"] >= resumed["job_seconds"]


@pytest.mark.parametrize("charge", [-1.0, float("nan"), float("inf")])
def test_invalid_setup_charge_rejected_before_output(tmp_path, charge):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        train(
            config(),
            tmp_path / "invalid",
            control=TrainControl(setup_job_seconds=charge),
        )
    assert not (tmp_path / "invalid").exists()
