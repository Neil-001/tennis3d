"""Validation journal integration using only the small train/val CPU fixtures."""

import importlib
import json
import shutil

import pytest
from test_lifting_recipe import equal_state, fixture_config, recipe_config

from tennis3d.artifacts import file_sha256, read_json
from tennis3d.lifting.checkpoint import read_checkpoint
from tennis3d.lifting.train import TrainControl

trainer = importlib.import_module("tennis3d.lifting.train")


def rows(directory):
    return [
        json.loads(line)
        for line in (directory / "validation-history.jsonl").read_text().splitlines()
    ]


def test_fresh_history_relocated_resume_and_selected_state(tmp_path):
    model = recipe_config().model
    cfg = fixture_config(
        model=model, max_steps=4, validation_every=2, checkpoint_every=1
    )
    full_dir, first_dir, resumed_dir = (
        tmp_path / p for p in ("full", "first", "resumed")
    )
    full = trainer.train(cfg, full_dir)
    first = trainer.train(cfg, first_dir, control=TrainControl(stop_after_steps=2))
    original = (first_dir / "validation-history.jsonl").read_bytes()
    resumed = trainer.train(cfg, resumed_dir, resume=first["last_checkpoint"])
    assert (first_dir / "validation-history.jsonl").read_bytes() == original
    assert (resumed_dir / "validation-history.jsonl").read_bytes().startswith(original)
    history = rows(resumed_dir)
    assert [r["step"] for r in history] == [2, 4]
    assert history[0]["run_id"] == history[1]["run_id"] == first["run_id"]
    assert history[0]["invocation_id"] != history[1]["invocation_id"]
    assert [r["resumed_from_step"] for r in history] == [None, 2]
    latest = read_json(resumed_dir / "validation.json")
    for row in history:
        assert row["windows"] == row["step"] * cfg.batch_size
        assert row["sampled_views"] >= row["windows"]
        assert row["training_objective"] == "xyz-weighted-radial-v1"
        assert "10m" in row["loss_units"]
        assert set(row["modes"]) == {"corrupted", "clean"}
        for metrics in row["modes"].values():
            assert metrics["val_xyz_m"] > 0
            assert metrics["vertical_mae_m"] >= 0
            assert metrics["p95_m"] >= metrics["median_m"]
            assert "occluded" in metrics["strata"]
            assert "physical rally_id" in metrics["definitions"]["val_xyz_m"]
            assert "views" not in metrics and "rallies" not in metrics
    for mode, metrics in history[-1]["modes"].items():
        assert metrics == {
            k: v
            for k, v in latest["modes"][mode].items()
            if k not in ("views", "rallies")
        }
    assert history[1]["train_seconds"] > history[0]["train_seconds"] > 0
    assert history[1]["job_seconds"] > history[0]["job_seconds"] > 0
    # Logging and relocation retain exact selected optimizer, schedule, sampler,
    # corruption RNG, weights and EMA. Best still selects the corrupted score.
    for checkpoint in ("last_checkpoint", "best_checkpoint"):
        a, b = read_checkpoint(full[checkpoint]), read_checkpoint(resumed[checkpoint])
        for key in ("model", "ema", "optimizer", "sampler", "rng"):
            equal_state(a[key], b[key])
    best_row = min(history, key=lambda r: r["modes"]["corrupted"]["val_xyz_m"])
    assert read_checkpoint(resumed["best_checkpoint"])["step"] == best_row["step"]
    assert resumed["best_score"] == best_row["modes"]["corrupted"]["val_xyz_m"]
    before = (resumed_dir / "validation-history.jsonl").read_bytes()
    best_hash = file_sha256(resumed["best_checkpoint"])
    trainer.train(cfg, resumed_dir, resume=resumed["last_checkpoint"])
    assert (resumed_dir / "validation-history.jsonl").read_bytes() == before
    assert file_sha256(resumed["best_checkpoint"]) == best_hash


def test_recovery_repeated_step_appends_and_does_not_reselect_best(
    tmp_path, monkeypatch
):
    cfg = fixture_config(max_steps=3, validation_every=1, checkpoint_every=2)

    def fixed_score(model, manifest, **kwargs):
        return {
            "manifest_sha256": manifest.manifest_sha256,
            "source_identity": manifest.identity,
            "modes": {"corrupted": {"val_xyz_m": 1.0}},
        }

    monkeypatch.setattr(trainer, "evaluate_model", fixed_score)
    directory = tmp_path / "run"
    first = trainer.train(cfg, directory, control=TrainControl(stop_after_steps=1))
    older = directory / "checkpoints" / "older.pt"
    shutil.copyfile(first["last_checkpoint"], older)
    best_hash = file_sha256(first["best_checkpoint"])
    save = trainer.save_checkpoint

    def crash_before_last(path, state):
        if path.name == "last.pt" and state["step"] == 2:
            raise RuntimeError("crash after validation journal append")
        save(path, state)

    monkeypatch.setattr(trainer, "save_checkpoint", crash_before_last)
    with pytest.raises(RuntimeError, match="crash after validation"):
        trainer.train(cfg, directory, resume=older)
    before = (directory / "validation-history.jsonl").read_bytes()
    monkeypatch.setattr(trainer, "save_checkpoint", save)
    recovered = trainer.train(cfg, directory, resume=older)
    assert (directory / "validation-history.jsonl").read_bytes().startswith(before)
    history = rows(directory)
    assert [r["step"] for r in history] == [1, 2, 2, 3]
    assert history[1]["invocation_id"] != history[2]["invocation_id"]
    assert history[1]["resumed_from_step"] == history[2]["resumed_from_step"] == 1
    assert file_sha256(recovered["best_checkpoint"]) == best_hash


@pytest.mark.parametrize("invalid", ['{"run_id":', '{"run_id":"other"}\n'])
def test_invalid_history_is_preserved_and_rejected(tmp_path, invalid):
    directory = tmp_path / "run"
    directory.mkdir()
    path = directory / "validation-history.jsonl"
    path.write_text(invalid)
    with pytest.raises(ValueError, match="preserve it and restore a complete copy"):
        trainer._prepare_validation_history(directory, None, "run")
    assert path.read_text() == invalid


def test_existing_relocated_history_restores_missing_suffix_and_rejects_divergence(
    tmp_path,
):
    source, target = tmp_path / "source", tmp_path / "target"
    (source / "checkpoints").mkdir(parents=True)
    target.mkdir()
    first = '{"run_id":"run","step":2}\n'
    second = '{"run_id":"run","step":4}\n'
    journal = source / "validation-history.jsonl"
    relocated = target / journal.name
    journal.write_text(first + second)
    relocated.write_text(first)
    resume = source / "checkpoints" / "last.pt"
    trainer._prepare_validation_history(target, resume, "run")
    assert relocated.read_text() == journal.read_text() == first + second
    # A target with later events retains its append-only suffix.
    journal.write_text(first)
    trainer._prepare_validation_history(target, resume, "run")
    assert relocated.read_text() == first + second
    journal.write_text(first + '{"run_id":"run","step":3}\n')
    before = journal.read_bytes(), relocated.read_bytes()
    with pytest.raises(ValueError, match="histories diverge"):
        trainer._prepare_validation_history(target, resume, "run")
    assert (journal.read_bytes(), relocated.read_bytes()) == before
    # A valid target must not conceal an incomplete source journal.
    journal.write_text(first + '{"run_id":')
    with pytest.raises(ValueError, match="invalid validation history"):
        trainer._prepare_validation_history(target, resume, "run")
    assert relocated.read_text() == first + second
