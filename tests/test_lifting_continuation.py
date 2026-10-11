"""Forking a finished run: exact inherited state, appended tail and own journals."""

import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tennis3d.artifacts import file_sha256, read_json
from tennis3d.cli import main
from tennis3d.lifting.checkpoint import read_checkpoint, save_checkpoint
from tennis3d.lifting.continuation import tail_policy, tail_rate
from tennis3d.lifting.model import ModelConfig
from tennis3d.lifting.settings import TrainConfig
from tennis3d.lifting.train import TrainControl, _cosine_rate

trainer = importlib.import_module("tennis3d.lifting.train")
FIXTURE = Path(__file__).parent / "fixtures" / "lifting"
SOURCE_STEPS, ADDED = 7, 5
CANDIDATE = "c" * 32


def source_config(**changes):
    return TrainConfig(
        train_manifest=str(FIXTURE / "train.manifest.json"),
        val_manifest=str(FIXTURE / "val.manifest.json"),
        model=ModelConfig(
            architecture_version="stage3-v1",
            width=16,
            depth=2,
            dropout=0.1,
            schedule_steps=SOURCE_STEPS,
        ),
        max_steps=SOURCE_STEPS,
        min_frames=20,
        max_frames=32,
        batch_size=2,
        validation_every=2,
        checkpoint_every=2,
        patience=8,
        device="cpu",
    ).model_copy(update=changes)


def candidate_config(**changes):
    return source_config(max_steps=SOURCE_STEPS + ADDED, patience=None).model_copy(
        update=changes
    )


def scored(scores):
    """Validation stub returning the given corrupted scores in call order."""
    remaining = list(scores)

    def evaluate(model, manifest, **kwargs):
        return {
            "manifest_sha256": manifest.manifest_sha256,
            "source_identity": manifest.identity,
            "modes": {"corrupted": {"val_xyz_m": remaining.pop(0)}},
        }

    return evaluate


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


def tree_hashes(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_plan(path, checkpoint, **changes):
    state = read_checkpoint(checkpoint)
    plan = {
        "schema_version": "1.0",
        "kind": "cosine-appended-tail-v1",
        "run_id": CANDIDATE,
        "source": {
            "checkpoint": str(checkpoint),
            "sha256": file_sha256(checkpoint),
            "run_id": state["run_id"],
            "step": state["step"],
            "best_score": state["best_score"],
        },
        "additional_steps": ADDED,
        "min_ratio": 0.05,
        "start_learning_rates": [
            group["lr"] for group in state["optimizer"]["param_groups"]
        ],
    }
    for key, value in changes.items():
        if key == "candidate_run_id":
            plan["run_id"] = value
        elif key in plan["source"]:
            plan["source"][key] = value
        else:
            plan[key] = value
    path.write_text(json.dumps(plan))
    return path


@pytest.fixture
def source(tmp_path, monkeypatch):
    """A finished source run whose selected score no candidate record can beat."""
    monkeypatch.setattr(trainer, "evaluate_model", scored([0.5, 0.4, 0.3, 0.2]))
    summary = trainer.train(source_config(), tmp_path / "source")
    assert summary["stop_reason"] == "max_steps" and summary["best_score"] == 0.2
    return tmp_path / "source" / "checkpoints" / "best.pt"


def test_fork_inherits_exact_state_and_keeps_its_own_records(
    tmp_path, monkeypatch, source
):
    before = tree_hashes(tmp_path / "source")
    inherited = read_checkpoint(source)
    plan = write_plan(tmp_path / "plan.json", source)
    saved = []
    save = trainer.save_checkpoint

    def capture(path, state):
        if path.name == "last.pt":
            saved.append((state["step"], state["last_validation_step"]))
        if not (tmp_path / "first.pt").exists():
            save(tmp_path / "first.pt", state)
        save(path, state)

    monkeypatch.setattr(trainer, "save_checkpoint", capture)
    # Every candidate score is worse than the source's 0.2; selection must not
    # fall back to the inherited record.
    monkeypatch.setattr(trainer, "evaluate_model", scored([0.9, 0.7, 0.6, 0.8]))
    output = tmp_path / "candidate"
    summary = trainer.train(candidate_config(), output, fork=plan)

    assert tree_hashes(tmp_path / "source") == before
    first = read_checkpoint(tmp_path / "first.pt")
    assert first["run_id"] == summary["run_id"] == CANDIDATE != inherited["run_id"]
    for key in ("model", "ema", "sampler", "rng"):
        equal_state(first[key], inherited[key])
    equal_state(first["optimizer"]["state"], inherited["optimizer"]["state"])
    assert int(first["ema"]["n_averaged"]) == SOURCE_STEPS + 1
    for new, old in zip(
        first["optimizer"]["param_groups"], inherited["optimizer"]["param_groups"]
    ):
        assert {k: v for k, v in new.items() if k != "lr_schedule"} == {
            k: v for k, v in old.items() if k != "lr_schedule"
        }
        assert new["lr_schedule"] == tail_policy(old["lr"], ADDED, SOURCE_STEPS)
    # A fork starts its own budget, selection and validation state.
    assert first["step"] == SOURCE_STEPS and first["best_score"] is None
    assert first["budget"]["train_seconds"] == 0.0
    assert first["last_validation_step"] == -1 and first["stale_validations"] == 0

    history = rows(output / "validation-history.jsonl")
    assert [row["step"] for row in history] == [7, 9, 11, 12]
    assert {row["run_id"] for row in history} == {CANDIDATE}
    assert all(row["resumed_from_step"] is None for row in history)
    assert [row["continuation"]["additional_steps"] for row in history] == [0, 2, 4, 5]
    assert [row["windows"] for row in history] == [14, 18, 22, 24]
    last = history[-1]["continuation"]
    assert last["source_run_id"] == inherited["run_id"]
    assert last["additional_windows"] == ADDED * 2
    assert (
        last["additional_sampled_views"]
        == history[-1]["sampled_views"] - inherited["sampler"]["draws"]
        >= ADDED * 2
    )
    # Recovery and validation cadence count from the fork point, not step zero.
    assert saved == [(7, -1), (7, 7), (9, 9), (11, 11), (12, 12)]

    final = read_checkpoint(summary["last_checkpoint"])
    best = read_checkpoint(summary["best_checkpoint"])
    assert summary["best_score"] == 0.6 and best["step"] == final["best_step"] == 11
    assert summary["added_exposure"] == last
    lineage = final["continuation"]
    assert lineage == read_json(output / "run.json")["continuation"]
    assert lineage["plan_sha256"] == file_sha256(plan)
    assert lineage["final_step"] == SOURCE_STEPS + ADDED
    assert lineage["source"]["best_score"] == 0.2
    assert lineage["source"]["checkpoint_sha256"] == file_sha256(source)
    assert lineage["source"]["budget"] == inherited["budget"]
    assert lineage["source"]["sampler"]["draws"] == inherited["sampler"]["draws"]
    assert lineage["source"]["learning_rate_policies"] == [
        group["lr_schedule"] for group in inherited["optimizer"]["param_groups"]
    ]

    starts = lineage["tail"]["start_learning_rates"]
    assert starts == [g["lr"] for g in inherited["optimizer"]["param_groups"]]
    logged = rows(output / "loss.jsonl")
    assert [row["step"] for row in logged] == [8, 9, 10, 11, 12]
    for index, row in enumerate(logged, start=1):
        factor = 0.05 + 0.95 * (1 + math.cos(math.pi * index / ADDED)) / 2
        assert row["learning_rates"] == pytest.approx(
            [start * factor for start in starts], rel=1e-12
        )
    rates = [row["learning_rates"][0] for row in logged]
    assert starts[0] > rates[0] and rates == sorted(rates, reverse=True)
    assert logged[-1]["learning_rates"] == pytest.approx(
        lineage["tail"]["final_learning_rates"], rel=1e-12
    )
    assert final["optimizer"]["param_groups"][0]["lr_schedule"]["step"] == 12


def test_interrupted_fork_resumes_exactly_as_an_ordinary_run(
    tmp_path, monkeypatch, source
):
    plan = write_plan(tmp_path / "plan.json", source)
    monkeypatch.setattr(trainer, "evaluate_model", scored([0.9, 0.7, 0.6, 0.8]))
    full = trainer.train(candidate_config(), tmp_path / "full", fork=plan)

    monkeypatch.setattr(trainer, "evaluate_model", scored([0.9, 0.7, 0.6, 0.8]))
    stopped = trainer.train(
        candidate_config(),
        tmp_path / "split",
        fork=plan,
        control=TrainControl(stop_after_steps=3, evaluate_at_end=False),
    )
    assert stopped["step"] == 10 and stopped["stop_reason"] == "external_step_limit"
    resumed = trainer.train(
        candidate_config(), tmp_path / "split", resume=stopped["last_checkpoint"]
    )
    a, b = (
        read_checkpoint(full["last_checkpoint"]),
        read_checkpoint(resumed["last_checkpoint"]),
    )
    for key in ("model", "ema", "optimizer", "sampler", "rng", "continuation"):
        equal_state(a[key], b[key])
    assert resumed["run_id"] == CANDIDATE and resumed["best_score"] == 0.6
    history = rows(tmp_path / "split" / "validation-history.jsonl")
    assert [row["step"] for row in history] == [7, 9, 11, 12]
    assert [row["resumed_from_step"] for row in history] == [None, None, 10, 10]
    # A resumed candidate cannot be extended past its planned final step.
    with pytest.raises(ValueError, match="planned final step"):
        trainer.train(
            candidate_config(max_steps=SOURCE_STEPS + ADDED + 1),
            tmp_path / "split",
            resume=resumed["last_checkpoint"],
        )
    # The fork's stop rule is frozen once it has started.
    with pytest.raises(ValueError, match="resume must preserve"):
        trainer.train(
            candidate_config(patience=8),
            tmp_path / "split",
            resume=resumed["last_checkpoint"],
        )


def test_fork_killed_before_its_initial_validation_still_records_it(
    tmp_path, monkeypatch, source
):
    plan = write_plan(tmp_path / "plan.json", source)

    def crash(model, manifest, **kwargs):
        raise RuntimeError("fixture crash")

    monkeypatch.setattr(trainer, "evaluate_model", crash)
    with pytest.raises(RuntimeError, match="fixture crash"):
        trainer.train(candidate_config(), tmp_path / "candidate", fork=plan)
    last = tmp_path / "candidate" / "checkpoints" / "last.pt"
    assert read_checkpoint(last)["last_validation_step"] == -1
    assert not (tmp_path / "candidate" / "validation-history.jsonl").exists()
    # Forking again into the started output is refused; the run resumes instead.
    with pytest.raises(ValueError, match="new empty output"):
        trainer.train(candidate_config(), tmp_path / "candidate", fork=plan)
    monkeypatch.setattr(trainer, "evaluate_model", scored([0.9, 0.7, 0.6, 0.8]))
    trainer.train(candidate_config(), tmp_path / "candidate", resume=last)
    history = rows(tmp_path / "candidate" / "validation-history.jsonl")
    assert [row["step"] for row in history] == [7, 9, 11, 12]


def tampered(tmp_path, source, change):
    state = read_checkpoint(source)
    change(state)
    path = tmp_path / "tampered" / "checkpoints" / "best.pt"
    path.parent.mkdir(parents=True)
    save_checkpoint(path, state)
    return path


def test_fork_fails_closed_on_conflicting_inputs(tmp_path, monkeypatch, source):
    monkeypatch.setattr(trainer, "evaluate_model", scored([0.9] * 20))
    good = write_plan(tmp_path / "plan.json", source)

    attempts = iter(range(100))

    def refused(match, *, config=None, plan=good, output=None, **kwargs):
        output = tmp_path / (output or f"refused-{next(attempts)}")
        with pytest.raises((ValueError, RuntimeError), match=match):
            trainer.train(config or candidate_config(), output, fork=plan, **kwargs)
        assert not (output / "checkpoints" / "last.pt").exists()

    refused("mutually exclusive", resume=source)
    for name, value in (
        ("sha256", "0" * 64),
        ("run_id", "d" * 32),
        ("step", SOURCE_STEPS - 1),
        ("best_score", 0.25),
        ("start_learning_rates", [0.01, 0.01]),
    ):
        refused(
            "differs? from",
            plan=write_plan(tmp_path / f"{name}.json", source, **{name: value}),
        )
    with pytest.raises(ValueError):
        trainer.train(
            candidate_config(),
            tmp_path / "refused",
            fork=write_plan(tmp_path / "ratio.json", source, min_ratio=0.2),
        )
    with pytest.raises(ValueError):
        trainer.train(
            candidate_config(),
            tmp_path / "refused",
            fork=write_plan(
                tmp_path / "same.json",
                source,
                candidate_run_id=read_checkpoint(source)["run_id"],
            ),
        )
    refused("planned final step", config=candidate_config(max_steps=SOURCE_STEPS + 1))
    for change in (
        {"batch_size": 1},
        {"ema_decay": 0.5},
        {"learning_rate": 0.01},
        {"gradient_clip": 1.0},
        {"seed": 7},
        {"validation_every": 3},
        {"model": source_config().model.model_copy(update={"width": 32})},
    ):
        refused("resume must preserve", config=candidate_config(**change))

    # A source stopped before its planned schedule ended cannot take a tail.
    monkeypatch.setattr(trainer, "evaluate_model", scored([0.3] * 4))
    early = trainer.train(
        source_config(),
        tmp_path / "early",
        control=TrainControl(stop_after_steps=3),
    )
    refused(
        "has not finished",
        plan=write_plan(tmp_path / "early.json", Path(early["last_checkpoint"])),
        config=candidate_config(max_steps=3 + ADDED),
    )

    # Strict EMA and optimizer loading: an incomplete source state is rejected.
    def drop_ema(state):
        state["ema"].pop("module.missing_ball")

    def drop_moment(state):
        state["optimizer"]["state"][0].pop("momentum_buffer")

    refused(
        "Missing key",
        plan=write_plan(
            tmp_path / "ema.json", tampered(tmp_path / "a", source, drop_ema)
        ),
    )
    refused(
        "complete optimizer moments",
        plan=write_plan(
            tmp_path / "moment.json", tampered(tmp_path / "b", source, drop_moment)
        ),
    )

    # Conflicting or torn candidate histories are preserved and refused.
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "validation-history.jsonl").write_bytes(
        (source.parent.parent / "validation-history.jsonl").read_bytes()
    )
    refused("new empty output", output="foreign")
    monkeypatch.setattr(trainer, "evaluate_model", scored([0.9, 0.7, 0.6, 0.8]))
    stopped = trainer.train(
        candidate_config(),
        tmp_path / "torn",
        fork=good,
        control=TrainControl(stop_after_steps=3, evaluate_at_end=False),
    )
    history = tmp_path / "torn" / "validation-history.jsonl"
    complete = history.read_bytes()
    for damaged in (
        complete[:-20],
        complete + (source.parent.parent / "validation-history.jsonl").read_bytes(),
    ):
        history.write_bytes(damaged)
        with pytest.raises(ValueError, match="invalid validation history"):
            trainer.train(
                candidate_config(), tmp_path / "torn", resume=stopped["last_checkpoint"]
            )
        assert history.read_bytes() == damaged
    # A finished candidate is resumed, never forked again.
    history.write_bytes(complete)
    done = trainer.train(
        candidate_config(), tmp_path / "torn", resume=stopped["last_checkpoint"]
    )
    refused(
        "resume it instead",
        plan=write_plan(
            tmp_path / "chain.json",
            Path(done["last_checkpoint"]),
            candidate_run_id="e" * 32,
        ),
        config=candidate_config(max_steps=SOURCE_STEPS + 2 * ADDED),
    )


def test_ordinary_runs_and_resume_carry_no_continuation(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "evaluate_model", scored([0.4, 0.3, 0.2, 0.1]))
    stopped = trainer.train(
        source_config(),
        tmp_path / "run",
        control=TrainControl(stop_after_steps=4, evaluate_at_end=False),
    )
    resumed = trainer.train(
        source_config(), tmp_path / "run", resume=stopped["last_checkpoint"]
    )
    state = read_checkpoint(resumed["last_checkpoint"])
    history = rows(tmp_path / "run" / "validation-history.jsonl")
    assert [row["step"] for row in history] == [2, 4, 6, 7]
    assert "continuation" not in state and "added_exposure" not in resumed
    assert "continuation" not in read_json(tmp_path / "run" / "run.json")
    assert all("continuation" not in row for row in history)
    assert state["optimizer"]["param_groups"][0]["lr_schedule"]["kind"] == (
        "cosine-planned-warmup-v1"
    )


def test_tail_rate_offsets_and_exhaustion():
    policy = tail_policy(2.0, 4, 100)
    observed = []
    for _ in range(6):
        observed.append(_cosine_rate(policy))
        policy["step"] += 1
    expected = [
        2.0 * (0.05 + 0.95 * (1 + math.cos(math.pi * min(k, 4) / 4)) / 2)
        for k in range(1, 7)
    ]
    assert observed == pytest.approx(expected, rel=1e-15)
    assert observed[3] == observed[5] == pytest.approx(0.1) and max(observed) < 2.0
    for broken in (
        {**policy, "step": 99},
        {**policy, "offset": 0},
        {**policy, "min_ratio": 0.2},
        {k: v for k, v in policy.items() if k != "offset"},
        {**policy, "warmup_steps": 1},
    ):
        with pytest.raises(ValueError):
            _cosine_rate(broken)
    with pytest.raises(ValueError):
        tail_rate({**policy, "offset": 1.0})


def test_cli_fork_requires_config_and_runs(tmp_path, monkeypatch, capsys, source):
    plan = write_plan(tmp_path / "plan.json", source)
    config = tmp_path / "candidate.json"
    config.write_text(candidate_config().model_dump_json())
    assert main(["train", "--fork", str(plan), "--output", str(tmp_path / "x")]) == 2
    assert "frozen --config" in capsys.readouterr().out
    monkeypatch.setattr(trainer, "evaluate_model", scored([0.9, 0.7, 0.6, 0.8]))
    arguments = ["train", "--fork", str(plan), "--config", str(config)]
    assert main([*arguments, "--output", str(tmp_path / "cli")]) == 0
    assert read_json(tmp_path / "cli" / "summary.json")["step"] == SOURCE_STEPS + ADDED
    assert read_json(tmp_path / "cli" / "run.json")["run_id"] == CANDIDATE


def test_wrapper_forks_once_then_resumes(tmp_path):
    commands = tmp_path / "bin"
    commands.mkdir()
    fake = commands / "python"
    fake.write_text(
        f"#!{sys.executable}\n"
        + """import json,sys,os
from pathlib import Path
if sys.argv[1] in ('-', '-c'):
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
a=sys.argv[1:]; out=Path(a[a.index('--output')+1]); out.mkdir(parents=True,exist_ok=True)
(out/'checkpoints').mkdir(exist_ok=True)
(out/'run.json').write_text('{"run_id":"fake"}')
(out/'checkpoints/last.pt').write_text('checkpoint')
(out/'summary.json').write_text(json.dumps({'stop_reason':'max_steps' if '--resume' in a else 'requested'}))
with open(os.environ['FAKE_ARGS'],'a') as f: f.write(json.dumps(a)+'\\n')
"""
    )
    fake.chmod(0o755)
    for name, content in (("nvidia-smi", "exit 0"), ("git", "echo fake-commit")):
        command = commands / name
        command.write_text("#!/bin/sh\n" + content + "\n")
        command.chmod(0o755)
    (tmp_path / "root").mkdir()
    env = {
        **os.environ,
        "PATH": str(commands) + ":" + os.environ["PATH"],
        "TENNIS3D_ROOT": str(tmp_path / "root"),
        "TENNIS3D_PYTHON": str(fake),
        "TENNIS3D_LOCAL": str(tmp_path / "local"),
        "TENNIS3D_OUTPUT": str(tmp_path / "durable"),
        "TENNIS3D_CONFIG": str(tmp_path / "candidate.yaml"),
        "TENNIS3D_FORK_PLAN": str(tmp_path / "plan.json"),
        "TENNIS3D_LOGS": str(tmp_path),
        "TENNIS3D_SYNC_SECONDS": "0.1",
        "FAKE_ARGS": str(tmp_path / "args.jsonl"),
    }
    for name in ("TENNIS3D_DATA_MASTER", "TENNIS3D_BACKUP_STORE"):
        env.pop(name, None)
    command = ["bash", str(Path("scripts/train.slurm").resolve())]
    for expected in (143, 0):
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=15, check=False
        )
        assert result.returncode == expected, result.stdout + result.stderr
    first, second = rows(tmp_path / "args.jsonl")
    assert first[first.index("--fork") + 1] == str(tmp_path / "plan.json")
    assert "--resume" not in first and "--config" in first
    assert "--fork" not in second and "--resume" in second
    assert second[second.index("--expect-plan") + 1] == str(tmp_path / "plan.json")


def test_crash_after_initial_validation_keeps_the_recorded_selection(
    tmp_path, monkeypatch, source
):
    plan = write_plan(tmp_path / "plan.json", source)
    save, writes = trainer.save_checkpoint, []

    def crash_before_second_last(path, state):
        writes.append(path.name)
        if writes == ["last.pt", "best.pt", "last.pt"]:
            raise RuntimeError("fixture crash")
        save(path, state)

    monkeypatch.setattr(trainer, "save_checkpoint", crash_before_second_last)
    monkeypatch.setattr(trainer, "evaluate_model", scored([0.1]))
    output = tmp_path / "candidate"
    with pytest.raises(RuntimeError, match="fixture crash"):
        trainer.train(candidate_config(), output, fork=plan)
    last = output / "checkpoints" / "last.pt"
    assert read_checkpoint(last)["best_score"] is None
    assert [row["step"] for row in rows(output / "validation-history.jsonl")] == [7]

    monkeypatch.setattr(trainer, "save_checkpoint", save)
    monkeypatch.setattr(trainer, "evaluate_model", scored([0.9, 0.7, 0.6, 0.8]))
    summary = trainer.train(candidate_config(), output, resume=last)
    history = rows(output / "validation-history.jsonl")
    assert [row["step"] for row in history] == [7, 7, 9, 11, 12]
    recorded = min(row["modes"]["corrupted"]["val_xyz_m"] for row in history)
    assert summary["best_score"] == recorded == 0.1
    best = read_checkpoint(summary["best_checkpoint"])
    assert best["step"] == 7 and best["best_score"] == 0.1
    equal_state(best["ema"], read_checkpoint(source)["ema"])


def test_resume_refuses_another_plan_or_rewritten_lineage(
    tmp_path, monkeypatch, source
):
    plan = write_plan(tmp_path / "plan.json", source)
    other = write_plan(tmp_path / "other.json", source, candidate_run_id="f" * 32)
    monkeypatch.setattr(trainer, "evaluate_model", scored([0.9] * 20))
    output = tmp_path / "candidate"
    stopped = trainer.train(
        candidate_config(),
        output,
        fork=plan,
        control=TrainControl(stop_after_steps=1, evaluate_at_end=False),
    )
    last = stopped["last_checkpoint"]
    before = tree_hashes(output)

    def refused(match, **kwargs):
        with pytest.raises(ValueError, match=match):
            trainer.train(candidate_config(), output, resume=last, **kwargs)
        assert tree_hashes(output) == before

    refused("expected plan", expected_plan=other)
    identity = output / "run.json"
    original = identity.read_text()
    changed = json.loads(original)
    changed["continuation"]["plan_sha256"] = "0" * 64
    identity.write_text(json.dumps(changed))
    before = tree_hashes(output)
    refused("lineage differs")
    refused("lineage differs", expected_plan=plan)
    identity.write_text(original)
    with pytest.raises(ValueError, match="resumed continuation only"):
        trainer.train(candidate_config(), tmp_path / "new", expected_plan=plan)
    done = trainer.train(candidate_config(), output, resume=last, expected_plan=plan)
    assert done["step"] == SOURCE_STEPS + ADDED

    # An ordinary run is not a continuation of any plan.
    ordinary = trainer.train(
        source_config(),
        tmp_path / "ordinary",
        control=TrainControl(stop_after_steps=2, evaluate_at_end=False),
    )
    with pytest.raises(ValueError, match="expected plan"):
        trainer.train(
            source_config(),
            tmp_path / "ordinary",
            resume=ordinary["last_checkpoint"],
            expected_plan=plan,
        )


def test_crash_between_best_and_journal_recovers_the_selected_record(
    tmp_path, monkeypatch, source
):
    plan = write_plan(tmp_path / "plan.json", source)
    write = trainer.write_json

    def crash_at_validation(path, value, **kwargs):
        if Path(path).name == "validation.json":
            raise RuntimeError("fixture crash")
        write(path, value, **kwargs)

    monkeypatch.setattr(trainer, "write_json", crash_at_validation)
    monkeypatch.setattr(trainer, "evaluate_model", scored([0.1]))
    output = tmp_path / "candidate"
    with pytest.raises(RuntimeError, match="fixture crash"):
        trainer.train(candidate_config(), output, fork=plan)
    assert (output / "checkpoints" / "best.pt").exists()
    assert not (output / "validation-history.jsonl").exists()

    monkeypatch.setattr(trainer, "write_json", write)
    monkeypatch.setattr(trainer, "evaluate_model", scored([0.9, 0.7, 0.6, 0.8]))
    summary = trainer.train(
        candidate_config(), output, resume=output / "checkpoints" / "last.pt"
    )
    history = rows(output / "validation-history.jsonl")
    scores = [row["modes"]["corrupted"]["val_xyz_m"] for row in history]
    assert scores == [0.1, 0.9, 0.7, 0.6, 0.8]
    assert [row["step"] for row in history] == [7, 7, 9, 11, 12]
    assert history[0]["run_id"] == CANDIDATE and history[0]["resumed_from_step"] is None
    assert summary["best_score"] == min(scores) == 0.1
    best = read_checkpoint(summary["best_checkpoint"])
    assert best["validation_record"] == history[0] and best["step"] == 7
    # A later resume finds the row already journaled and adds nothing.
    again = trainer.train(candidate_config(), output, resume=summary["last_checkpoint"])
    assert again["best_score"] == 0.1
    assert rows(output / "validation-history.jsonl") == history


def test_resume_refuses_a_journal_that_conflicts_with_the_selected_best(
    tmp_path, monkeypatch, source
):
    plan = write_plan(tmp_path / "plan.json", source)
    monkeypatch.setattr(trainer, "evaluate_model", scored([0.9, 0.7, 0.6, 0.8]))
    output = tmp_path / "candidate"
    stopped = trainer.train(
        candidate_config(),
        output,
        fork=plan,
        control=TrainControl(stop_after_steps=1, evaluate_at_end=False),
    )
    history = output / "validation-history.jsonl"
    row = rows(history)[0]
    best_path = output / "checkpoints" / "best.pt"
    best = read_checkpoint(best_path)
    assert best["validation_record"] == row

    def refused(match):
        before = tree_hashes(output)
        with pytest.raises(ValueError, match=match):
            trainer.train(candidate_config(), output, resume=stopped["last_checkpoint"])
        assert tree_hashes(output) == before

    # The journaled row of the selecting validation claims another score.
    row["modes"]["corrupted"]["val_xyz_m"] = 0.1
    history.write_text(json.dumps(row) + "\n")
    refused("conflicts with the selected best")
    history.write_text(json.dumps(best["validation_record"]) + "\n")
    # The embedded record disagrees with the checkpoint that carries it.
    for change in (
        {"step": 8},
        {"run_id": "d" * 32},
        {"modes": {"corrupted": {"val_xyz_m": 0.1}}},
        {"modes": {}},
    ):
        save_checkpoint(
            best_path,
            {**best, "validation_record": best["validation_record"] | change},
        )
        refused("validation record disagree")
    save_checkpoint(best_path, best)
    # The same event journaled twice with different content, after an exact row.
    history.write_text(
        json.dumps(best["validation_record"]) + "\n" + json.dumps(row) + "\n"
    )
    refused("one event twice")
    # An interrupted sync copied a newer journal without its checkpoints.
    newer = best["validation_record"] | {"step": 8, "invocation_id": "later"}
    newer = newer | {"modes": {"corrupted": {"val_xyz_m": 0.1}}}
    history.write_text(
        json.dumps(best["validation_record"]) + "\n" + json.dumps(newer) + "\n"
    )
    refused("better score than the available best")
    best_path.rename(tmp_path / "best.aside")
    history.write_text(json.dumps(best["validation_record"]) + "\n")
    refused("better score than the available best|selected best.pt")
    (tmp_path / "best.aside").rename(best_path)
    done = trainer.train(candidate_config(), output, resume=stopped["last_checkpoint"])
    assert done["step"] == SOURCE_STEPS + ADDED and done["best_score"] == 0.6
