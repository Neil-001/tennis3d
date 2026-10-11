"""Physical reserved membership and measured full-run planning contracts."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("pyarrow")
from test_dataset_parquet import sample, write_sample  # noqa: F401

from tennis3d.artifacts import file_sha256, write_json
from tennis3d.datasets.io import write_manifest
from tennis3d.datasets.schema import Artifact, Manifest, count_records
from tennis3d.datasets.source import SourceSpec, V2Source
from tennis3d.lifting.preparation import TARGET_WINDOWS, resolve
from tennis3d.lifting.settings import TrainConfig


def test_exclusion_removes_physical_rally_with_all_views_and_keeps_row_offsets(
    tmp_path,
    sample,  # noqa: F811
):
    record, provenance = sample
    second = deepcopy(record.views[0])
    second.metadata = second.metadata.model_copy(update={"view_id": 1})
    record.views.append(second)
    _, shard = write_sample(tmp_path, sample, count=3)
    manifest = Manifest(
        state="complete",
        provenance=provenance,
        shards=[shard],
        expected=count_records([shard]),
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    spec = SourceSpec(
        manifest=Artifact(
            path=path.name, size_bytes=path.stat().st_size, sha256=file_sha256(path)
        ),
        root=".",
        split="train",
        exclude_rally_ids=["rally-1"],
    )
    with_source = V2Source(tmp_path / "source.json", spec.model_dump())
    try:
        assert [(r["rally_id"], r["view_id"]) for r in with_source.records] == [
            ("rally-0", 0),
            ("rally-0", 1),
            ("rally-2", 0),
            ("rally-2", 1),
        ]
        assert with_source.locations == [(0, 0, 0), (0, 0, 1), (0, 2, 0), (0, 2, 1)]
        assert sorted(with_source.order(np.random.default_rng(42))) == list(range(4))
        assert with_source.read(2)["t_s"].shape == (3,)
        assert with_source.identity["membership_sha256"]
    finally:
        with_source.close()
    with pytest.raises(ValueError, match="absent from source split"):
        V2Source(
            tmp_path / "source.json",
            spec.model_copy(update={"exclude_rally_ids": ["unknown"]}).model_dump(),
        )


def test_measured_horizon_resolves_math_and_caps_exposure(tmp_path):
    recipe = Path("configs/lifter.yaml")
    original = yaml.safe_load(recipe.read_text())
    identities = {"train": {"split": "train"}, "reserved-val": {"split": "val"}}
    receipt = {"sources": {}}
    for name, identity in identities.items():
        path = tmp_path / f"{name}.source.json"
        write_json(path, identity)
        receipt["sources"][name] = {
            "identity": identity,
            "descriptor_sha256": file_sha256(path),
        }
    receipt_path = tmp_path / "sources.receipt.json"
    write_json(receipt_path, receipt)
    profile = {
        "recipe_sha256": file_sha256(recipe),
        "sources_receipt_sha256": file_sha256(receipt_path),
        "train_identity": identities["train"],
        "val_identity": identities["reserved-val"],
        "target_windows": TARGET_WINDOWS,
        "seed": 42,
        "final_step": 350,
        "warmup_updates": 50,
        "measured_updates": 300,
        "profile_schedule_steps": 83334,
        "exact_continuation": {"equal": True},
        "total_wall_seconds": 2500,
        "train_seconds": 45,
        "validation_wall_seconds": 1000,
        "timings": {
            "updates": [
                {"step_seconds": 2.0, "invocation": "uninterrupted", "step": step}
                for step in range(1, 351)
            ],
            "checkpoints": [{"seconds": 0.25}],
            "backup_publications": [{"seconds": 0.5}],
            "trainer_preflights": [{"seconds": 100}],
            "loader_batches": [{"seconds": 1.0}] * 100,
            "validation_preflight_seconds": 100,
            "validation_forward_seconds": 900,
        },
    }
    profile_path = tmp_path / "profile.json"
    write_json(profile_path, profile)
    config_path = tmp_path / "full.yaml"
    plan = resolve(recipe, tmp_path, profile_path, config_path)
    config = yaml.safe_load(config_path.read_text())
    assert (
        config["max_steps"]
        == config["model"]["schedule_steps"]
        == plan["planned_steps"]
    )
    assert config["max_steps"] < plan["target_steps"]
    assert config["max_train_seconds"] > 300
    assert config["max_job_seconds"] == 172800
    assert config["checkpoint_every"] < config["validation_every"]
    assert config["patience"] == 8
    assert plan["estimated_wall_seconds"] <= 172800
    assert plan["estimated_training_workload_seconds"] <= 0.8 * 172800
    assert config["max_train_seconds"] <= 0.8 * 172800
    for field in (
        "batch_size",
        "learning_rate",
        "ema_decay",
        "gradient_clip",
        "min_frames",
        "max_frames",
    ):
        assert config[field] == original[field]
    normalized = TrainConfig.model_validate(original).model_dump(mode="json")
    assert {k: v for k, v in config["model"].items() if k != "schedule_steps"} == {
        k: v for k, v in normalized["model"].items() if k != "schedule_steps"
    }
    assert yaml.safe_load(recipe.read_text()) == original
    assert resolve(recipe, tmp_path, profile_path, config_path) == plan
    (tmp_path / "train.source.json").write_text("changed")
    with pytest.raises(ValueError, match="descriptor or identity differs"):
        resolve(recipe, tmp_path, profile_path, config_path)
    write_json(receipt_path, {"different": "sources"}, overwrite=True)
    with pytest.raises(ValueError, match="different prepared sources"):
        resolve(recipe, tmp_path, profile_path, config_path)


def test_streaming_slurm_syncs_interruption_and_resumes_without_data_master(tmp_path):
    import os
    import subprocess
    import sys

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
(out/'checkpoints/last.pt').write_text('resumed checkpoint' if '--resume' in a else 'initial')
(out/'summary.json').write_text(json.dumps({'stop_reason':'max_steps' if '--resume' in a else 'requested'}))
with open(os.environ['FAKE_ARGS'],'a') as f: f.write(json.dumps(a)+'\\n')
"""
    )
    fake.chmod(0o755)
    for name, content in (("nvidia-smi", "exit 0"), ("git", "echo fake-commit")):
        command = commands / name
        command.write_text("#!/bin/sh\n" + content + "\n")
        command.chmod(0o755)
    root, local, durable = tmp_path / "root", tmp_path / "local", tmp_path / "durable"
    root.mkdir()
    env = {
        **os.environ,
        "PATH": str(commands) + ":" + os.environ["PATH"],
        "TENNIS3D_ROOT": str(root),
        "TENNIS3D_PYTHON": str(fake),
        "TENNIS3D_LOCAL": str(local),
        "TENNIS3D_OUTPUT": str(durable),
        "TENNIS3D_CONFIG": str(tmp_path / "full.yaml"),
        "TENNIS3D_LOGS": str(tmp_path),
        "TENNIS3D_SYNC_SECONDS": "0.1",
        "FAKE_ARGS": str(tmp_path / "args.jsonl"),
    }
    env.pop("TENNIS3D_DATA_MASTER", None)
    command = ["bash", str(Path("scripts/train.slurm").resolve())]
    first = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=15, check=False
    )
    assert first.returncode == 143, first.stdout + first.stderr
    assert (durable / "checkpoints/last.pt").read_text() == "initial"
    second = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=15, check=False
    )
    assert second.returncode == 0, second.stdout + second.stderr
    assert (durable / "checkpoints/last.pt").read_text() == "resumed checkpoint"
    assert not (root / "data").exists()
    calls = [
        __import__("json").loads(line)
        for line in (tmp_path / "args.jsonl").read_text().splitlines()
    ]
    assert len(calls) == 2 and "--resume" not in calls[0] and "--resume" in calls[1]
