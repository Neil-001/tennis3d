"""Private-Hub contract, immutable publications, and exact CPU fixture recovery."""

import shutil
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from tennis3d.artifacts import file_sha256, json_sha256, read_json, write_json
from tennis3d.lifting import backup
from tennis3d.lifting.checkpoint import read_checkpoint
from tennis3d.lifting.train import TrainControl, train
from tests.test_lifting_train import config, equal_state

REPO = "test/private-lifter"


class MemoryHub:
    def __init__(self, root):
        self.root = root
        self.head = "0" * 40
        self.history = []
        self.calls = 0
        self.lose_reply = False
        self.private = True
        self.corrupt = None

    def info(self, repo):
        if not self.private:
            raise ValueError("private model repository required")
        return self.head

    def exists_prefix(self, repo, commit, prefix):
        return (self.root / commit / prefix).exists()

    def commits(self, repo):
        return list(reversed(self.history))

    def commit(self, repo, parent, snapshot_id, path, manifest):
        assert parent == self.head
        self.calls += 1
        self.head = f"{self.calls:040x}"
        target = self.root / self.head / "snapshots" / snapshot_id
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(path, target)
        self.history.append(
            SimpleNamespace(title=f"tennis3d backup {snapshot_id}", commit_id=self.head)
        )
        if self.lose_reply:
            self.lose_reply = False
            raise TimeoutError("lost response with secret hf_DO_NOT_LOG")
        return self.head

    def download(self, repo, commit, name, cache):
        assert backup.HEX40.fullmatch(commit)
        path = self.root / commit / name
        if self.corrupt == name:
            path.write_bytes(b"corrupt")
            self.corrupt = None
        return path


def fixture_run(tmp_path, *, validations=False):
    cfg = config().model_copy(update={"validation_every": 1})
    source = tmp_path / "run"
    train(
        cfg,
        source,
        control=TrainControl(
            stop_after_steps=2,
            validate_during_training=validations,
            evaluate_at_end=validations,
        ),
    )
    return cfg, source


def test_downloaded_restored_fixture_continues_exact_state(tmp_path):
    cfg, source = fixture_run(tmp_path)
    source_budget = read_json(source / "budget.json")
    # Work charged after the selected weights must remain charged on recovery.
    source_budget["train_seconds"] += 10
    source_budget["job_seconds"] += 20
    write_json(source / "budget.json", source_budget, overwrite=True)
    (source / "secret.env").write_text("hf_DO_NOT_UPLOAD")
    store = tmp_path / "store"
    snapshot_id = backup.publish(source, store)
    snapshot = store / "snapshots" / snapshot_id
    manifest = backup.verify_snapshot(snapshot, snapshot_id)
    assert "secret.env" not in manifest["files"]
    assert manifest["source_files"]["budget.json"]["sha256"] == file_sha256(
        source / "budget.json"
    )
    charged = read_json(snapshot / "budget.json")
    assert (
        charged["job_seconds"]
        >= source_budget["job_seconds"] + manifest["capture_job_seconds"]
    )
    assert read_json(source / "budget.json") == source_budget
    assert backup.publish(source, store) == snapshot_id
    assert len(list((store / "snapshots").iterdir())) == 1
    hub = MemoryHub(tmp_path / "hub")
    receipt = backup.upload_pending(store, REPO, hub=hub)[snapshot_id]
    assert receipt["manifest"] == manifest and hub.calls == 1
    restored = tmp_path / "restored"
    backup.restore(
        REPO,
        receipt["commit"],
        snapshot_id,
        restored,
        run_id=manifest["run_id"],
        hub=hub,
    )
    assert file_sha256(restored / "checkpoints/last.pt") == file_sha256(
        source / "checkpoints/last.pt"
    )
    result = train(
        cfg,
        tmp_path / "continued",
        resume=restored / "checkpoints/last.pt",
        control=TrainControl(validate_during_training=False, evaluate_at_end=False),
    )
    full = train(
        cfg,
        tmp_path / "full",
        control=TrainControl(validate_during_training=False, evaluate_at_end=False),
    )
    actual, expected = (
        read_checkpoint(result["last_checkpoint"]),
        read_checkpoint(full["last_checkpoint"]),
    )
    for field in ("model", "ema", "optimizer", "sampler", "rng"):
        equal_state(actual[field], expected[field])
    assert result["train_seconds"] >= source_budget["train_seconds"]
    assert result["job_seconds"] >= charged["job_seconds"]


def test_preserves_original_best_and_history_bytes(tmp_path):
    _, source = fixture_run(tmp_path, validations=True)
    store = tmp_path / "store"
    snapshot_id = backup.publish(source, store)
    snapshot = store / "snapshots" / snapshot_id
    for name in ("checkpoints/best.pt", "validation-history.jsonl", "loss.jsonl"):
        assert (snapshot / name).read_bytes() == (source / name).read_bytes()
    assert "summary.json" in read_json(snapshot / "manifest.json")["files"]


@pytest.mark.parametrize(
    "damage", ["journal", "ledger", "config", "best", "dataset", "changed", "symlink"]
)
def test_rejects_torn_mismatched_and_changing_sources_without_rewriting(
    tmp_path, monkeypatch, damage
):
    _, source = fixture_run(tmp_path, validations=True)
    if damage == "journal":
        with (source / "validation-history.jsonl").open("ab") as stream:
            stream.write(b'{"run_id":')
    elif damage == "ledger":
        ledger = read_json(source / "budget.json")
        ledger["run_id"] = "different"
        write_json(source / "budget.json", ledger, overwrite=True)
    elif damage == "config":
        write_json(source / "config.resolved.json", {"seed": 100}, overwrite=True)
    elif damage == "best":
        shutil.copyfile(source / "checkpoints/last.pt", source / "checkpoints/best.pt")
        state = read_checkpoint(source / "checkpoints/best.pt")
        state["best_step"] = -1
        from tennis3d.lifting.checkpoint import save_checkpoint

        save_checkpoint(source / "checkpoints/best.pt", state)
    elif damage == "dataset":
        from tennis3d.lifting.checkpoint import save_checkpoint

        state = read_checkpoint(source / "checkpoints/best.pt")
        state["dataset_identity"] = {"different": {"format_version": 2}}
        save_checkpoint(source / "checkpoints/best.pt", state)
    elif damage == "symlink":
        (source / "loss.jsonl").unlink()
        (source / "loss.jsonl").symlink_to(source / "validation-history.jsonl")
    elif damage == "changed":
        original = backup._copy

        def change(origin, destination):
            original(origin, destination)
            if origin.name == "budget.json":
                with origin.open("ab") as stream:
                    stream.write(b"\n")

        monkeypatch.setattr(backup, "_copy", change)
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    with pytest.raises((ValueError, KeyError)):
        backup.publish(source, tmp_path / "store")
    if damage != "changed":
        assert {p: p.read_bytes() for p in before} == before
    assert not list((tmp_path / "store" / "snapshots").iterdir())


def test_stale_summary_is_omitted_without_modifying_source(tmp_path):
    _, source = fixture_run(tmp_path)
    summary = read_json(source / "summary.json")
    summary["last_checkpoint_sha256"] = "0" * 64
    write_json(source / "summary.json", summary, overwrite=True)
    snapshot_id = backup.publish(source, tmp_path / "store")
    manifest = backup.verify_snapshot(
        tmp_path / "store" / "snapshots" / snapshot_id, snapshot_id
    )
    assert "summary.json" not in manifest["files"]
    assert read_json(source / "summary.json") == summary


def test_lost_commit_response_is_reconciled_without_second_commit(tmp_path):
    _, source = fixture_run(tmp_path)
    store = tmp_path / "store"
    snapshot_id = backup.publish(source, store)
    hub = MemoryHub(tmp_path / "hub")
    hub.lose_reply = True
    with pytest.raises(TimeoutError):
        backup.upload_pending(store, REPO, hub=hub)
    assert not read_json(store / "state.json")["receipts"]
    receipts = backup.upload_pending(store, REPO, hub=hub)
    assert receipts[snapshot_id]["commit"] == hub.head and hub.calls == 1
    backup.upload_pending(store, REPO, hub=hub)
    assert hub.calls == 1
    assert not list(store.glob(".verify-*"))


def test_queue_and_completed_cache_are_bounded(tmp_path):
    _, source = fixture_run(tmp_path)
    store = tmp_path / "store"
    for value in range(backup.MAX_PENDING):
        ledger = read_json(source / "budget.json")
        ledger["job_seconds"] += 1
        write_json(source / "budget.json", ledger, overwrite=True)
        backup.publish(source, store)
    ledger["job_seconds"] += 1
    write_json(source / "budget.json", ledger, overwrite=True)
    with pytest.raises(ValueError, match="queue full"):
        backup.publish(source, store)
    assert len(list((store / "snapshots").iterdir())) == backup.MAX_PENDING
    hub = MemoryHub(tmp_path / "hub")
    backup.upload_pending(store, REPO, hub=hub)
    assert hub.calls == backup.MAX_PENDING
    assert len(list((store / "snapshots").iterdir())) == backup.KEEP_UPLOADED
    backup.publish(source, store)
    backup.upload_pending(store, REPO, hub=hub)
    assert hub.calls == backup.MAX_PENDING + 1


@pytest.mark.parametrize(
    "damage", ["hash", "traversal", "run_id", "existing", "unpinned"]
)
def test_restore_fails_without_conflicting_or_partial_output(tmp_path, damage):
    _, source = fixture_run(tmp_path)
    store = tmp_path / "store"
    snapshot_id = backup.publish(source, store)
    hub = MemoryHub(tmp_path / "hub")
    receipt = backup.upload_pending(store, REPO, hub=hub)[snapshot_id]
    output = tmp_path / "restored"
    kwargs = {}
    commit = receipt["commit"]
    if damage == "hash":
        hub.corrupt = f"snapshots/{snapshot_id}/checkpoints/last.pt"
    elif damage == "traversal":
        path = hub.root / commit / "snapshots" / snapshot_id / "manifest.json"
        manifest = read_json(path)
        manifest["files"]["../credentials"] = manifest["files"].pop("budget.json")
        snapshot_id = json_sha256(backup._manifest_body(manifest))
        manifest["snapshot_id"] = snapshot_id
        target = hub.root / commit / "snapshots" / snapshot_id
        target.mkdir()
        write_json(target / "manifest.json", manifest)
    elif damage == "run_id":
        kwargs["run_id"] = "other-run"
    elif damage == "existing":
        output.mkdir()
        (output / "preserve").write_text("original")
    elif damage == "unpinned":
        commit = "main"
    with pytest.raises((ValueError, FileExistsError)):
        backup.restore(REPO, commit, snapshot_id, output, hub=hub, **kwargs)
    if damage == "existing":
        assert (output / "preserve").read_text() == "original"
    else:
        assert not output.exists()
    assert not list(tmp_path.glob(".restore-*"))


def test_sdk_batches_operations_and_requires_private_model_repo(tmp_path, monkeypatch):
    import huggingface_hub

    _, source = fixture_run(tmp_path)
    snapshot_id = backup.publish(source, tmp_path / "store")
    path = tmp_path / "store" / "snapshots" / snapshot_id
    manifest = backup.verify_snapshot(path, snapshot_id)
    calls = []
    api = SimpleNamespace(
        repo_info=lambda repo, **kwargs: SimpleNamespace(private=True, sha="0" * 40),
        create_commit=lambda **kwargs: (
            calls.append(kwargs) or SimpleNamespace(oid="1" * 40)
        ),
    )
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    hub = backup.Hub()
    assert hub.info(REPO) == "0" * 40
    hub.commit(REPO, "0" * 40, snapshot_id, path, manifest)
    assert len(calls) == 1 and calls[0]["repo_type"] == "model"
    assert len(calls[0]["operations"]) == len(manifest["files"]) + 1
    assert all(
        operation.path_in_repo.startswith(f"snapshots/{snapshot_id}/")
        for operation in calls[0]["operations"]
    )
    api.repo_info = lambda *args, **kwargs: SimpleNamespace(private=False)
    with pytest.raises(ValueError, match="private"):
        hub.info(REPO)
    with pytest.raises(ValueError, match="dataset"):
        hub.info("N-il/tennis-3d-synthetic")


def test_failure_is_visible_and_does_not_log_sdk_secret(tmp_path, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise TimeoutError("hf_DO_NOT_LOG")

    monkeypatch.setattr(backup, "upload_pending", fail)
    assert backup.main(["upload", "--repo", REPO, "--store", str(tmp_path)]) == 1
    error = capsys.readouterr().err
    assert "TimeoutError" in error and "retry required" in error
    assert "hf_DO_NOT_LOG" not in error


@pytest.mark.parametrize("requested", [False, True])
def test_slurm_backup_failure_preserves_success_and_requeue(tmp_path, requested):
    import os
    import subprocess
    import sys
    from pathlib import Path

    commands = tmp_path / "bin"
    commands.mkdir()
    fake = commands / "python"
    fake.write_text(
        f"#!{sys.executable}\n"
        + """import json,sys,os,signal,time
from pathlib import Path
if sys.argv[1] in ('-', '-c'):
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
if sys.argv[2] == 'tennis3d.lifting.backup':
    with open(os.environ['BACKUP_CALLS'],'a') as f: f.write(json.dumps(sys.argv[3:])+'\\n')
    sys.exit(7)
a=sys.argv[1:]; out=Path(a[a.index('--output')+1]); out.mkdir(parents=True,exist_ok=True)
(out/'checkpoints').mkdir(exist_ok=True)
(out/'run.json').write_text('{"run_id":"fake"}')
(out/'checkpoints/last.pt').write_text('durable weights')
reason='max_steps'
if os.environ['REQUESTED'] == '1':
    signal.signal(signal.SIGTERM, lambda *args: None)
    os.kill(os.getppid(), signal.SIGUSR1)
    time.sleep(0.05)
    reason='requested'
(out/'summary.json').write_text(json.dumps({'stop_reason':reason}))
"""
    )
    fake.chmod(0o755)
    for name, content in (
        ("nvidia-smi", "exit 0"),
        ("git", "echo fake-commit"),
        ("scontrol", 'echo "$*" >> "$REQUEUE_CALLS"'),
    ):
        command = commands / name
        command.write_text("#!/bin/sh\n" + content + "\n")
        command.chmod(0o755)
    root = tmp_path / "root"
    root.mkdir()
    durable = tmp_path / "durable"
    env = {
        **os.environ,
        "PATH": str(commands) + ":" + os.environ["PATH"],
        "TENNIS3D_ROOT": str(root),
        "TENNIS3D_PYTHON": str(fake),
        "TENNIS3D_LOCAL": str(tmp_path / "local"),
        "TENNIS3D_OUTPUT": str(durable),
        "TENNIS3D_CONFIG": str(tmp_path / "full.yaml"),
        "TENNIS3D_LOGS": str(tmp_path),
        "TENNIS3D_SYNC_SECONDS": "100",
        "TENNIS3D_BACKUP_STORE": str(tmp_path / "store"),
        "BACKUP_CALLS": str(tmp_path / "backup-calls.jsonl"),
        "REQUEUE_CALLS": str(tmp_path / "requeue.txt"),
        "REQUESTED": "1" if requested else "0",
        "SLURM_JOB_ID": "123",
    }
    env.pop("TENNIS3D_DATA_MASTER", None)
    result = subprocess.run(
        ["bash", str(Path("scripts/train.slurm").resolve())],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "backup publication failed" in result.stderr
    assert (durable / "checkpoints/last.pt").read_text() == "durable weights"
    calls = [
        __import__("json").loads(line)
        for line in (tmp_path / "backup-calls.jsonl").read_text().splitlines()
    ]
    assert calls[-2][:7] == [
        "publish",
        "--source",
        str(durable),
        "--store",
        str(tmp_path / "store"),
        "--interval",
        "0",
    ]
    if requested:
        assert "requeue 123" in (tmp_path / "requeue.txt").read_text()
    else:
        assert not (tmp_path / "requeue.txt").exists()


def test_final_charge_is_idempotent_and_recovers_lost_ack(tmp_path):
    import time

    _, source = fixture_run(tmp_path)
    store = tmp_path / "store"
    original = read_json(source / "budget.json")
    charge = backup.account_final(
        source, store, event="allocation-1", started=time.monotonic() - 0.5
    )
    charged = read_json(source / "budget.json")
    assert charge["wrapper_job_seconds"] >= 0.5
    assert charged["job_seconds"] >= original["job_seconds"] + 0.5
    assert charged["train_seconds"] == original["train_seconds"]
    backup.account_final(
        source, store, event="allocation-1", started=time.monotonic() - 100
    )
    assert read_json(source / "budget.json") == charged
    # Simulate death before the ledger write and after its write, respectively.
    state = read_json(store / "state.json")
    state["final_charges"]["allocation-1"]["applied"] = False
    write_json(store / "state.json", state, overwrite=True)
    write_json(source / "budget.json", original, overwrite=True)
    backup.account_final(source, store)
    assert read_json(source / "budget.json") == charged
    state["final_charges"]["allocation-1"]["applied"] = False
    write_json(store / "state.json", state, overwrite=True)
    backup.account_final(source, store)
    assert read_json(source / "budget.json") == charged
    # A divergent ledger must never be overwritten by a stale write-ahead record.
    state["final_charges"]["allocation-1"]["applied"] = False
    write_json(store / "state.json", state, overwrite=True)
    divergent = {**charged, "job_seconds": charged["job_seconds"] + 1}
    write_json(source / "budget.json", divergent, overwrite=True)
    with pytest.raises(ValueError, match="conflicts"):
        backup.account_final(source, store)
    assert read_json(source / "budget.json") == divergent


def test_ordinary_slurm_requeue_resume_carries_final_publication_charge(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    _, source = fixture_run(tmp_path)
    commands = tmp_path / "bin"
    commands.mkdir()
    fake = commands / "python"
    fake.write_text(
        f"#!{sys.executable}\n"
        + """import json,sys,os,signal,time
from pathlib import Path
if sys.argv[1] in ('-', '-c'):
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
if sys.argv[2] == 'tennis3d.lifting.backup':
    if sys.argv[3] == 'publish': time.sleep(0.08)
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
a=sys.argv[1:]; out=Path(a[a.index('--output')+1])
print('--job-setup-seconds', a[a.index('--job-setup-seconds')+1], flush=True)
with open(os.environ['STAGED_BUDGETS'],'a') as f: f.write((out/'budget.json').read_text())
summary=json.loads((out/'summary.json').read_text())
if os.environ['REQUESTED'] == '1':
    signal.signal(signal.SIGTERM, lambda *args: None)
    os.kill(os.getppid(), signal.SIGUSR1)
    time.sleep(0.05)
    summary['stop_reason']='requested'
else:
    summary['stop_reason']='max_steps'
(out/'summary.json').write_text(json.dumps(summary))
"""
    )
    fake.chmod(0o755)
    for name, content in (
        ("nvidia-smi", "exit 0"),
        ("git", "echo fake-commit"),
        ("scontrol", "exit 0"),
    ):
        path = commands / name
        path.write_text("#!/bin/sh\n" + content + "\n")
        path.chmod(0o755)
    durable = tmp_path / "durable"
    shutil.copytree(source, durable)
    summary = read_json(durable / "summary.json")
    summary["stop_reason"] = "requested"
    write_json(durable / "summary.json", summary, overwrite=True)
    original = read_json(durable / "budget.json")
    store = tmp_path / "store"
    env = {
        **os.environ,
        "PATH": str(commands) + ":" + os.environ["PATH"],
        "TENNIS3D_ROOT": str(Path.cwd()),
        "TENNIS3D_PYTHON": str(fake),
        "TENNIS3D_LOCAL": str(tmp_path / "local"),
        "TENNIS3D_OUTPUT": str(durable),
        "TENNIS3D_CONFIG": str(tmp_path / "config.yaml"),
        "TENNIS3D_LOGS": str(tmp_path),
        "TENNIS3D_SYNC_SECONDS": "100",
        "TENNIS3D_BACKUP_STORE": str(store),
        "STAGED_BUDGETS": str(tmp_path / "staged.jsonl"),
        "SLURM_JOB_ID": "123",
        "REQUESTED": "1",
    }
    env.pop("TENNIS3D_DATA_MASTER", None)
    command = ["bash", str(Path("scripts/train.slurm").resolve())]
    first = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=30, check=False
    )
    assert first.returncode == 0, first.stdout + first.stderr
    assert "requeueing" in first.stdout
    charged = read_json(durable / "budget.json")
    state = read_json(store / "state.json")
    charge = next(iter(state["final_charges"].values()))
    staging = float(
        first.stdout.split("charged wrapper staging seconds: ")[1].splitlines()[0]
    )
    assert staging > 0
    assert charge["applied"] and charge["wrapper_job_seconds"] >= 0.08
    assert "--job-setup-seconds" in first.stdout
    final_snapshot = store / "snapshots" / state["latest_snapshot"]
    manifest = read_json(final_snapshot / "manifest.json")
    snapshot_ledger = read_json(final_snapshot / "budget.json")
    assert manifest["wrapper_job_seconds"] >= 0.08
    assert manifest["wrapper_job_seconds"] >= manifest["capture_job_seconds"]
    assert (
        snapshot_ledger["job_seconds"]
        >= original["job_seconds"] + manifest["wrapper_job_seconds"]
    )
    assert (
        charged["job_seconds"]
        >= original["job_seconds"] + charge["wrapper_job_seconds"]
    )
    assert charged["train_seconds"] == original["train_seconds"]
    env["REQUESTED"] = "0"
    env["SLURM_RESTART_COUNT"] = "1"
    second = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=30, check=False
    )
    assert second.returncode == 0, second.stdout + second.stderr
    staged = [
        __import__("json").loads(line)
        for line in (tmp_path / "staged.jsonl").read_text().splitlines()
    ]
    assert staged[1] == charged
    assert read_json(durable / "budget.json")["job_seconds"] > charged["job_seconds"]
    assert len(read_json(store / "state.json")["final_charges"]) == 2


def test_final_publication_after_periodic_same_source_charges_once(tmp_path):
    import time

    _, source = fixture_run(tmp_path)
    store = tmp_path / "store"
    periodic = backup.publish(source, store)
    started = time.monotonic() - 120
    final = backup.publish(source, store, started=started)
    assert final != periodic
    manifest = backup.verify_snapshot(store / "snapshots" / final, final)
    assert manifest["wrapper_job_seconds"] >= 120
    assert backup.publish(source, store, started=started) == final
    backup.account_final(source, store, event="allocation-1", started=started)
    assert backup.publish(source, store, started=started) == final
    assert len(list((store / "snapshots").iterdir())) == 2


def test_remote_existing_path_with_different_title_is_never_overwritten(tmp_path):
    _, source = fixture_run(tmp_path)
    store = tmp_path / "store"
    snapshot_id = backup.publish(source, store)
    hub = MemoryHub(tmp_path / "hub")
    hub.lose_reply = True
    with pytest.raises(TimeoutError):
        backup.upload_pending(store, REPO, hub=hub)
    hub.history[0].title = "another uploader"
    receipts = backup.upload_pending(store, REPO, hub=hub)
    assert receipts[snapshot_id]["commit"] == hub.head
    assert hub.calls == 1
    # Corrupt existing content must fail, not be repaired by an overwrite.
    state = read_json(store / "state.json")
    state["receipts"] = {}
    write_json(store / "state.json", state, overwrite=True)
    hub.corrupt = f"snapshots/{snapshot_id}/checkpoints/last.pt"
    with pytest.raises(ValueError, match="hash"):
        backup.upload_pending(store, REPO, hub=hub)
    assert hub.calls == 1


def test_original_legacy_best_config_defaults_remain_resumable(tmp_path):
    from tennis3d.lifting.checkpoint import carry_best_checkpoint, save_checkpoint

    _, source = fixture_run(tmp_path, validations=True)
    best = read_checkpoint(source / "checkpoints/best.pt")
    for field in ("schedule_steps", "outward_radial_brake", "batched_muon"):
        del best["config"]["model"][field]
    save_checkpoint(source / "checkpoints/best.pt", best)
    original_hash = file_sha256(source / "checkpoints/best.pt")
    store = tmp_path / "store"
    snapshot_id = backup.publish(source, store)
    snapshot = store / "snapshots" / snapshot_id
    last = read_checkpoint(snapshot / "checkpoints/last.pt")
    carry_best_checkpoint(
        snapshot / "checkpoints/last.pt", last, tmp_path / "carried-best.pt"
    )
    assert file_sha256(tmp_path / "carried-best.pt") == original_hash


def test_remote_partial_snapshot_without_manifest_is_never_overwritten(tmp_path):
    _, source = fixture_run(tmp_path)
    store = tmp_path / "store"
    snapshot_id = backup.publish(source, store)
    hub = MemoryHub(tmp_path / "hub")
    hub.lose_reply = True
    with pytest.raises(TimeoutError):
        backup.upload_pending(store, REPO, hub=hub)
    hub.history[0].title = "another uploader"
    remote = hub.root / hub.head / "snapshots" / snapshot_id
    (remote / "manifest.json").unlink()
    evidence = remote / "checkpoints/last.pt"
    evidence.write_bytes(b"conflicting evidence")
    with pytest.raises(FileNotFoundError):
        backup.upload_pending(store, REPO, hub=hub)
    assert hub.calls == 1 and evidence.read_bytes() == b"conflicting evidence"


def test_slurm_refuses_undercharged_resume_after_pending_charge_conflict(tmp_path):
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    _, source = fixture_run(tmp_path)
    summary = read_json(source / "summary.json")
    summary["stop_reason"] = "requested"
    write_json(source / "summary.json", summary, overwrite=True)
    store = tmp_path / "store"
    backup.account_final(
        source, store, event="allocation-1", started=time.monotonic() - 120
    )
    state = read_json(store / "state.json")
    state["final_charges"]["allocation-1"]["applied"] = False
    write_json(store / "state.json", state, overwrite=True)
    ledger = read_json(source / "budget.json")
    ledger["job_seconds"] += 1
    write_json(source / "budget.json", ledger, overwrite=True)
    commands = tmp_path / "bin"
    commands.mkdir()
    fake = commands / "python"
    fake.write_text(
        f"#!{sys.executable}\n"
        + """import sys,os
if sys.argv[1] in ('-', '-c') or sys.argv[2] == 'tennis3d.lifting.backup':
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
raise AssertionError('undercharged trainer must not launch')
"""
    )
    fake.chmod(0o755)
    env = {
        **os.environ,
        "TENNIS3D_ROOT": str(Path.cwd()),
        "TENNIS3D_PYTHON": str(fake),
        "TENNIS3D_LOCAL": str(tmp_path / "local"),
        "TENNIS3D_OUTPUT": str(source),
        "TENNIS3D_BACKUP_STORE": str(store),
    }
    env.pop("TENNIS3D_DATA_MASTER", None)
    result = subprocess.run(
        ["bash", str(Path("scripts/train.slurm").resolve())],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert "refusing undercharged resume" in result.stderr
    assert not (tmp_path / "local" / "runs" / source.name).exists()
    assert read_json(source / "budget.json") == ledger
    assert not read_json(store / "state.json")["final_charges"]["allocation-1"][
        "applied"
    ]


def test_symlinked_scratch_ancestor_supports_publish_upload_and_restore(tmp_path):
    real = tmp_path / "scratch"
    real.mkdir()
    _, source = fixture_run(real)
    linked = tmp_path / "scratch-link"
    linked.symlink_to(real, target_is_directory=True)
    alias_source = linked / source.name
    store = linked / "store"
    snapshot_id = backup.publish(alias_source, store)
    hub = MemoryHub(tmp_path / "hub")
    receipt = backup.upload_pending(store, REPO, hub=hub)[snapshot_id]
    output = linked / "restored"
    backup.restore(REPO, receipt["commit"], snapshot_id, output, hub=hub)
    assert file_sha256(output / "checkpoints/last.pt") == file_sha256(
        source / "checkpoints/last.pt"
    )
    with pytest.raises(ValueError, match="separate"):
        backup.publish(source, linked / source.name / "nested-store")
    # Symlinks beneath the source remain forbidden.
    (source / "checkpoints").rename(real / "outside-checkpoints")
    (source / "checkpoints").symlink_to(
        real / "outside-checkpoints", target_is_directory=True
    )
    with pytest.raises(ValueError, match="symlink"):
        backup.publish(alias_source, store)


def test_full_queue_rejects_before_copy_and_allows_recorded_final_retry(
    tmp_path, monkeypatch
):
    import time

    _, source = fixture_run(tmp_path)
    store = tmp_path / "store"
    started = time.monotonic() - 1
    final = backup.publish(source, store, started=started)
    for _ in range(backup.MAX_PENDING - 1):
        ledger = read_json(source / "budget.json")
        ledger["job_seconds"] += 1
        write_json(source / "budget.json", ledger, overwrite=True)
        backup.publish(source, store)
    assert backup.publish(source, store, started=started) == final

    def forbidden(*args, **kwargs):
        pytest.fail("a full queue must reject before copying or validating files")

    monkeypatch.setattr(backup, "_copy", forbidden)
    monkeypatch.setattr(backup, "_validate_run", forbidden)
    with pytest.raises(ValueError, match="queue full"):
        backup.publish(source, store)
    assert not list(store.glob(".capture-*"))
