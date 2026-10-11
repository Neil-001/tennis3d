"""Sealed inference plus CPU analysis reproduces the single-process evaluation."""

import os
import shutil
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tennis3d.artifacts import file_sha256, json_sha256, read_json, write_json
from tennis3d.evaluation import synthetic_phases as phases
from tennis3d.evaluation.prediction_journal import PredictionJournal
from tennis3d.evaluation.synthetic_phases import (
    RECEIPT,
    SEAL,
    Budget,
    Invocation,
    _analyze_released,
    _infer_released,
    read_prediction,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures/lifting"
TIMING_KEYS = ("full_view_prediction_wall_s", "prediction_wall_s_total")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """Two views of two rallies over the genuine fixture view, scored in one process."""
    import torch

    import tennis3d.lifting.checkpoint as checkpoint_module
    from tennis3d.evaluation.delivery import _synthetic_released
    from tennis3d.lifting.data import ViewDataset
    from tennis3d.lifting.model import Lifter, ModelConfig

    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = Lifter(ModelConfig(width=16, depth=2))
    root = tmp_path_factory.mktemp("phases")
    (root / "source").mkdir()
    for name in ("config.resolved.json", "val.npz"):
        shutil.copyfile(FIXTURES / name, root / "source" / name)
    manifest = read_json(FIXTURES / "val.manifest.json")
    manifest["records"] = [
        {**manifest["records"][0], "rally_id": f"val-rally-{i:05d}"} for i in range(2)
    ]
    source = root / "source/val.manifest.json"
    write_json(source, manifest)
    with ViewDataset(source) as dataset:
        counts = {
            "rallies": len({r["rally_id"] for r in dataset.records}),
            "views": len(dataset),
            "frames": sum(r["frames"] for r in dataset.records),
        }
    digest = file_sha256(source)
    values = SimpleNamespace(
        model=model,
        source=source,
        release=SimpleNamespace(
            checkpoint=SimpleNamespace(path=source, sha256=digest), step=1
        ),
        provenance={
            "release": {"checkpoint": {"sha256": digest}},
            "release_sha256": "0" * 64,
        },
        counts=counts,
        single_root=root / "single",
    )
    with pytest.MonkeyPatch.context() as context:
        context.setattr(checkpoint_module, "load_lifter", lambda *args: (model, {}))
        with PredictionJournal(values.single_root, {"fixture": "single"}) as journal:
            values.single = _synthetic_released(
                values.release,
                values.provenance,
                source,
                journal,
                "cpu",
                expected_counts=counts,
            )
    return values


@pytest.fixture
def released(prepared, monkeypatch):
    import tennis3d.lifting.checkpoint as checkpoint_module

    monkeypatch.setattr(
        checkpoint_module, "load_lifter", lambda *args: (prepared.model, {})
    )
    for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS_ON_NODE"):
        monkeypatch.delenv(key, raising=False)
    return prepared


def infer(released, output, **options):
    return _infer_released(
        released.release,
        released.provenance,
        released.source,
        output,
        "cpu",
        expected_counts=released.counts,
        **options,
    )


def analyze(released, inference, output, **options):
    return _analyze_released(
        released.release,
        released.provenance,
        released.source,
        inference,
        output,
        expected_counts=released.counts,
        **options,
    )


def read_invocations(root):
    identity = read_json(root / "identity.json")
    return phases.read_invocations(root, identity["phase"], identity)


def forbid_inference(monkeypatch):
    import tennis3d.evaluation.synthetic as synthetic_module
    import tennis3d.inference.predict as predict_module
    import tennis3d.lifting.checkpoint as checkpoint_module

    def forbidden(*args, **kwargs):
        raise AssertionError("analysis loaded a model or recomputed a prediction")

    monkeypatch.setattr(checkpoint_module, "load_lifter", forbidden)
    monkeypatch.setattr(predict_module, "predict_view", forbidden)
    monkeypatch.setattr(synthetic_module, "predict_view", forbidden)


def without_timing(exported):
    return {
        **exported,
        "modes": {
            mode: {k: v for k, v in values.items() if k not in TIMING_KEYS}
            for mode, values in exported["modes"].items()
        },
    }


def test_split_phases_match_the_single_process_evaluation(
    released, tmp_path, monkeypatch
):
    import torch

    views, single = released.counts["views"], released.single
    seal = infer(released, tmp_path / "inference")
    inference = tmp_path / "inference"
    assert (seal["kind"], seal["status"]) == (
        "synthetic_inference_seal",
        "predictions_sealed",
    )
    assert len(seal["predictions"]) == 2 * views
    assert seal["source"]["counts"] == released.counts
    assert sorted(path.name for path in inference.iterdir()) == [
        "identity.json",
        SEAL,
        "invocations",
        "predictions",
        "writer.lock",
    ]
    inference_files = {
        path.relative_to(inference): file_sha256(path)
        for path in inference.rglob("*")
        if path.is_file()
    }

    forbid_inference(monkeypatch)
    # A CPU-only scheduler job may report a zero GPU count.
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "0")
    result = analyze(released, inference, tmp_path / "analysis")
    analysis = tmp_path / "analysis"
    assert (result["schema_version"], result["kind"], result["status"]) == (
        "1.0",
        "model_only_synthetic",
        "measured",
    )
    assert result["metrics"] == single["metrics"]
    assert result["physical"] == single["physical"]
    assert without_timing(result["exported_result_metrics"]) == without_timing(
        single["exported_result_metrics"]
    )
    assert sorted(p.name for p in (analysis / "physical").glob("*.json")) == [
        f"{index:05d}.json" for index in range(views)
    ]
    for index in range(views):
        name = f"physical/{index:05d}.json"
        assert (analysis / name).read_bytes() == (
            released.single_root / name
        ).read_bytes()
    for row in seal["predictions"]:
        _, _, split = read_prediction(inference / row["file"])
        _, _, whole = read_prediction(released.single_root / row["file"])
        assert all(
            np.array_equal(split[key], whole[key], equal_nan=True) for key in split
        )

    # The analysis reports the durations saved at inference time, not replay time.
    for mode in ("corrupted", "clean"):
        saved = [
            row["prediction_wall_s"]
            for row in seal["predictions"]
            if row["mode"] == mode
        ]
        exported = result["exported_result_metrics"]["modes"][mode]
        assert exported["prediction_wall_s_total"] == sum(saved)
        assert seal["prediction_wall_s_total"][mode] == sum(saved)
        assert min(saved) > 0
    assert result["runtime_s"]["inference_prediction_wall_s_total"] == sum(
        seal["prediction_wall_s_total"].values()
    )

    phases = result["phases"]
    assert phases["inference"]["seal_file_sha256"] == file_sha256(inference / SEAL)
    assert phases["inference"]["identity"] == read_json(inference / "identity.json")
    assert result["execution_identity"]["inference_seal_sha256"] == file_sha256(
        inference / SEAL
    )
    assert [row["status"] for row in phases["inference"]["invocations"]] == ["complete"]
    assert phases["analysis"]["cuda_initialized"] is False
    assert not torch.cuda.is_initialized()
    analysis_run = read_invocations(analysis)[0]
    assert analysis_run["status"] == "complete"
    assert analysis_run["counts"] == {
        "predictions_created": 0,
        "predictions_reused": 2 * views,
        "physical_created": views,
        "physical_reused": 0,
    }
    assert set(analysis_run["timings_s"]) == {
        "seal_verify",
        "preflight_including_reader",
        "sealed_prediction_read_and_verify",
        "physics_fit",
        "scoring_pass",
        "reader",
        "scoring_metrics_export_and_journal",
        "physical_summary",
    }
    inference_run = read_invocations(inference)[0]
    assert inference_run["timings_s"]["prediction"] == pytest.approx(
        sum(seal["prediction_wall_s_total"].values())
    )
    assert inference_run["hardware"]["cpu_count"] and inference_run["code"]
    receipt = read_json(analysis / RECEIPT)
    assert receipt["metrics_sha256"] == file_sha256(analysis / "metrics.json")
    assert receipt["physical_reports"] == views
    assert not (analysis / "predictions").exists()
    # Analysis only reads the sealed root.
    assert inference_files == {
        path.relative_to(inference): file_sha256(path)
        for path in inference.rglob("*")
        if path.is_file()
    }


def test_interrupted_phases_resume_without_repeating_finished_work(
    released, tmp_path, monkeypatch
):
    import tennis3d.evaluation.physical as physical_module
    import tennis3d.inference.predict as predict_module

    views = released.counts["views"]
    inference = tmp_path / "inference"
    original = predict_module.predict_view
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 4:
            raise RuntimeError("controlled inference interruption")
        return original(*args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(predict_module, "predict_view", interrupted)
        with pytest.raises(RuntimeError, match="controlled inference"):
            infer(released, inference)
    assert not (inference / SEAL).exists()
    with pytest.raises(ValueError, match="inference is incomplete"):
        analyze(released, inference, tmp_path / "early")
    assert not (tmp_path / "early").exists()
    saved = {
        path.name: file_sha256(path) for path in (inference / "predictions").iterdir()
    }
    assert len(saved) == 3
    with pytest.raises(OSError):
        infer(released, inference)
    seal = infer(released, inference, resume=True)
    runs = read_invocations(inference)
    assert [row["status"] for row in runs] == ["failed", "complete"]
    assert runs[0]["counts"]["predictions_created"] == 3
    assert runs[1]["counts"]["predictions_created"] == 2 * views - 3
    assert runs[1]["counts"]["predictions_reused"] == 3
    # Finished predictions keep their bytes and therefore their original timing.
    assert (
        saved.items()
        <= {
            Path(row["file"]).name: row["file_sha256"] for row in seal["predictions"]
        }.items()
    )
    assert sum(row["timings_s"]["prediction"] for row in runs) == pytest.approx(
        sum(seal["prediction_wall_s_total"].values())
    )
    # A sealed root is only verified again, and never on a GPU.
    assert infer(released, inference, resume=True) == seal
    assert len(read_invocations(inference)) == 2

    forbid_inference(monkeypatch)
    evaluate = physical_module.evaluate_physical
    fits = []

    def interrupted_fit(*args, **kwargs):
        fits.append(1)
        if len(fits) == 2:
            raise RuntimeError("controlled physics interruption")
        return evaluate(*args, **kwargs)

    analysis = tmp_path / "analysis"
    with monkeypatch.context() as context:
        context.setattr(physical_module, "evaluate_physical", interrupted_fit)
        with pytest.raises(RuntimeError, match="controlled physics"):
            analyze(released, inference, analysis)
    assert not (analysis / "metrics.json").exists()
    assert not (analysis / RECEIPT).exists()
    resumed = analyze(released, inference, analysis, resume=True)
    runs = read_invocations(analysis)
    assert [row["status"] for row in runs] == ["failed", "complete"]
    assert runs[1]["counts"]["physical_reused"] == 1
    assert runs[1]["counts"]["physical_created"] == views - 1
    assert len(resumed["phases"]["analysis"]["invocations"]) == 2
    assert resumed["metrics"] == released.single["metrics"]
    assert resumed["physical"] == released.single["physical"]
    assert without_timing(resumed["exported_result_metrics"]) == without_timing(
        released.single["exported_result_metrics"]
    )
    assert read_json(analysis / RECEIPT)["invocations"] == [
        {"file": f"invocations/{i:04d}.json", "sha256": row["record_sha256"]}
        for i, row in enumerate(runs)
    ]

    # A finished root must still match its receipt before it is resumed.
    def refused(change, match):
        root = tmp_path / f"finished-{match[:8]}-{len(list(tmp_path.iterdir()))}"
        shutil.copytree(analysis, root)
        change(root)
        with pytest.raises(ValueError, match=match):
            analyze(released, inference, root, resume=True)
        assert len(list((root / "invocations").glob("*.json"))) <= 2

    def inflate(root):
        path = root / "invocations/0001.json"
        write_json(path, {**read_json(path), "wall_s": 987654.0}, overwrite=True)

    refused(inflate, "differs from its completion receipt")
    refused(
        lambda root: (root / RECEIPT).write_bytes(b"torn"),
        "differs from its completion receipt",
    )
    refused(
        lambda root: shutil.rmtree(root / "invocations"),
        "differs from its completion receipt",
    )
    refused(
        lambda root: (root / "metrics.json").write_text("{}"),
        "differs from its completion receipt",
    )

    def miscount(root):
        receipt = read_json(root / RECEIPT)
        counts = dict.fromkeys(receipt["counts"], 9999)
        write_json(root / RECEIPT, {**receipt, "counts": counts}, overwrite=True)

    refused(miscount, "differs from its completion receipt")

    def unfinished(root):
        (root / RECEIPT).unlink()
        shutil.rmtree(root / "invocations")

    refused(unfinished, "results without invocation records")

    def chained(root):
        (root / RECEIPT).unlink()
        path = root / "invocations/0000.json"
        write_json(path, {**read_json(path), "wall_s": 987654.0}, overwrite=True)

    refused(chained, "damaged or foreign")
    replayed = tmp_path / "replayed"
    shutil.copytree(analysis, replayed)
    again = analyze(released, inference, replayed, resume=True)
    assert again["metrics"] == resumed["metrics"]
    assert len(read_json(replayed / RECEIPT)["invocations"]) == 3

    # A root that gained files this phase never writes is refused.
    shutil.copytree(inference / "predictions", analysis / "predictions")
    with pytest.raises(ValueError, match="another phase or command"):
        analyze(released, inference, analysis, resume=True)
    shutil.rmtree(analysis / "predictions")
    shutil.copyfile(analysis / "physical/00000.json", analysis / "physical/99999.json")
    with pytest.raises(ValueError, match="undeclared or missing physical"):
        analyze(released, inference, analysis, resume=True)


def test_analysis_rejects_incomplete_changed_or_foreign_predictions(
    released, tmp_path, monkeypatch
):
    from tennis3d.artifacts import canonical_json
    from tennis3d.evaluation.delivery import _synthetic_released

    sealed = tmp_path / "sealed"
    seal = infer(released, sealed)
    forbid_inference(monkeypatch)
    attempt = iter(range(100))

    def rejected(inference, match, **options):
        output = tmp_path / f"rejected-{next(attempt)}"
        with pytest.raises(ValueError, match=match):
            analyze(released, inference, output, **options)
        assert not (output / "metrics.json").exists()
        assert not (output / "physical").exists()

    def copied():
        target = tmp_path / f"copy-{next(attempt)}"
        shutil.copytree(sealed, target)
        return target

    first = seal["predictions"][0]["file"]
    root = copied()
    (root / first).unlink()
    rejected(root, "incomplete or holds undeclared")

    root = copied()
    shutil.copyfile(root / first, root / "predictions/99999-clean.npz")
    rejected(root, "incomplete or holds undeclared")

    # A prediction rewritten with consistent internal checksums still differs
    # from the sealed file hash.
    root = copied()
    _, metadata, values = read_prediction(root / first)
    np.savez(
        root / first,
        **values,
        metadata=np.frombuffer(canonical_json(metadata), dtype=np.uint8),
    )
    rejected(root, "differs from the seal")

    root = copied()
    (root / first).write_bytes((root / first).read_bytes()[:100])
    rejected(root, "missing, torn or damaged")

    root = copied()
    changed = read_json(root / SEAL)
    changed["predictions"][0]["prediction_wall_s"] = 0.0
    write_json(root / SEAL, changed, overwrite=True)
    rejected(root, "seal checksum or identity")

    # A seal over fewer views is internally consistent but not the full source.
    root = copied()
    changed = read_json(root / SEAL)
    for row in changed["predictions"][-2:]:
        (root / row["file"]).unlink()
    del changed["predictions"][-2:]
    del changed["seal_sha256"]
    changed["seal_sha256"] = json_sha256(changed)
    write_json(root / SEAL, changed, overwrite=True)
    rejected(root, "incomplete or holds undeclared")

    # A seal rehashed around invented timings contradicts the predictions.
    root = copied()
    changed = read_json(root / SEAL)
    changed["predictions"][0]["prediction_wall_s"] = 1e6
    changed["prediction_wall_s_total"] = phases.wall_totals(changed["predictions"])
    del changed["seal_sha256"]
    changed["seal_sha256"] = json_sha256(changed)
    write_json(root / SEAL, changed, overwrite=True)
    rejected(root, "differs from the seal")

    # Invocation records are part of the sealed provenance.
    root = copied()
    (root / "invocations/0000.json").unlink()
    rejected(root, "timings or invocation records differ")
    root = copied()
    record = read_json(root / "invocations/0000.json")
    write_json(
        root / "invocations/0000.json", {**record, "wall_s": 987654.0}, overwrite=True
    )
    rejected(root, "timings or invocation records differ")
    root = copied()
    write_json(
        root / "invocations/0000.json",
        {**record, "phase": "synthetic_analysis"},
        overwrite=True,
    )
    rejected(root, "damaged or foreign")
    root = copied()
    (root / "metrics.json").write_text("{}")
    rejected(root, "incomplete or holds undeclared")

    # Rerunning inference on a sealed root reads every prediction again.
    root = copied()
    (root / first).write_bytes(b"torn")
    with pytest.raises(ValueError, match="missing, torn or damaged"):
        infer(released, root, resume=True)

    other = SimpleNamespace(
        **{**vars(released), "provenance": {"release_sha256": "1" * 64}}
    )
    with pytest.raises(ValueError, match="different phase, release"):
        analyze(other, sealed, tmp_path / "other-release")
    with pytest.raises(ValueError, match="own root outside"):
        analyze(released, sealed, sealed / "nested")
    assert not (sealed / "nested").exists()
    for key, value in (("SLURM_JOB_GPUS", "0"), ("SLURM_GPUS_ON_NODE", "1")):
        monkeypatch.setenv(key, value)
        rejected(sealed, "release the GPU allocation")
        monkeypatch.delenv(key)

    # Roots of the single-process command and of either phase never mix.
    single = tmp_path / "single"
    with PredictionJournal(single, {"fixture": "single"}) as journal:
        journal.prediction(
            0,
            {"view_id": "fixture"},
            {"xyz": np.zeros((2, 3))},
            "clean",
            lambda: (np.zeros((2, 3)), np.ones(2, dtype=bool), np.zeros(2), 0.1),
        )
    rejected(single, "inference is incomplete")
    with pytest.raises(ValueError, match="resume identity differs"):
        infer(released, single, resume=True)
    with (
        pytest.raises(ValueError, match="resume identity differs"),
        PredictionJournal(sealed, {"fixture": "single"}, resume=True) as journal,
    ):
        _synthetic_released(
            released.release, released.provenance, released.source, journal, "cpu"
        )


def reservation(path, root, **changes):
    write_json(
        path,
        {
            "ledger_owner": "fixture ledger",
            "recorded_utc": "2026-10-08T00:00:00+00:00",
            "cap_gpu_seconds": 100.0,
            "charged_gpu_seconds": 40.0,
            "reserved_gpu_seconds": 60.0,
            "inference_root": str(root),
            **changes,
        },
    )
    return path


def test_reservation_is_bound_recorded_and_spent_once(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    for key in ("SLURM_JOB_ID", "SLURM_JOB_START_TIME", "SLURM_JOB_END_TIME"):
        monkeypatch.delenv(key, raising=False)

    def invocation(target=root):
        return Invocation(
            target,
            "synthetic_inference",
            {"fixture": 1},
            "cpu",
            time.perf_counter(),
            {},
        )

    def runs():
        return phases.read_invocations(root, "synthetic_inference", {"fixture": 1})

    with pytest.raises(ValueError, match="exceeds the remaining recorded budget"):
        Budget(
            reservation(tmp_path / "over.json", root, reserved_gpu_seconds=60.5),
            invocation(),
        )
    granted = reservation(tmp_path / "granted.json", root)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="another inference root"):
        Budget(granted, invocation(other))

    # Under a scheduler the allocation is charged from the job's start, and
    # the job's own limit must fit in the reservation.
    now = [1045.0]
    monkeypatch.setattr("time.time", lambda: now[0])
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_JOB_START_TIME", "1000")
    with pytest.raises(ValueError, match="did not export the job start and end"):
        Budget(granted, invocation())
    monkeypatch.setenv("SLURM_JOB_END_TIME", "1061")
    with pytest.raises(ValueError, match="time limit exceeds the remaining"):
        Budget(granted, invocation())
    monkeypatch.setenv("SLURM_JOB_END_TIME", "1060")
    run = invocation()
    budget = Budget(granted, run)
    with run:
        # The durable start record already names the grant.
        started = read_json(run.path)
        assert started["status"] == "running"
        assert started["gpu_reservation"]["sha256"] == file_sha256(granted)
        budget.check()
    first = runs()[-1]["gpu_reservation"]
    assert first["allocation_basis"] == "SLURM_JOB_START_TIME"
    assert first["allocation_elapsed_s"] == 45.0
    assert first["scheduler_time_limit_s"] == 60.0

    # The requeued job under the same reservation has 15 seconds left.
    monkeypatch.setenv("SLURM_JOB_START_TIME", "2000")
    monkeypatch.setenv("SLURM_JOB_END_TIME", "2016")
    with pytest.raises(ValueError, match="time limit exceeds the remaining"):
        Budget(granted, invocation())
    monkeypatch.setenv("SLURM_JOB_END_TIME", "2015")
    now[0] = 2014.0
    run = invocation()
    budget = Budget(granted, run)
    assert budget.record["used_before_s"] == 45.0
    with pytest.raises(InterruptedError, match="reserved GPU-seconds are spent"), run:
        budget.check()
        now[0] = 2015.0
        budget.check()
    last = runs()[-1]
    assert last["status"] == "interrupted"
    assert last["gpu_reservation"]["allocation_elapsed_s"] == 15.0

    # A hard kill leaves only the start record. That voids its reservation.
    killed = reservation(
        tmp_path / "killed.json", root, recorded_utc="2026-10-08T01:00:00+00:00"
    )
    now[0] = 2001.0
    run = invocation()
    Budget(killed, run)
    run.__enter__()
    with pytest.raises(ValueError, match="has no end record"):
        Budget(killed, invocation())
    renewed = reservation(
        tmp_path / "renewed.json", root, recorded_utc="2026-10-08T02:00:00+00:00"
    )
    assert Budget(renewed, invocation()).record["used_before_s"] == 0.0


def test_cuda_inference_stops_replay_and_sealing_when_the_reservation_is_spent(
    released, tmp_path, monkeypatch
):
    import tennis3d.inference.predict as predict_module
    from tennis3d.lifting import runtime

    root = tmp_path / "inference"
    with pytest.raises(ValueError, match="CPU inference takes none"):
        infer(
            released,
            tmp_path / "cpu",
            reservation_path=reservation(tmp_path / "cpu.json", tmp_path / "cpu"),
        )
    # Present the CPU fixture as a CUDA device without touching a GPU.
    original = predict_module.predict_view
    monkeypatch.setattr(runtime, "resolve_device", lambda device: "cuda")
    monkeypatch.setattr(runtime, "environment", lambda device: {"device": device})
    monkeypatch.setattr(phases, "hardware", lambda device: {})
    monkeypatch.setattr(runtime, "synchronize", lambda device: None)
    now = [1000.0]

    def predict(model, inputs, *, device, **options):
        now[0] += 10.0
        return original(model, inputs, device="cpu", **options)

    monkeypatch.setattr(predict_module, "predict_view", predict)
    monkeypatch.setattr("time.time", lambda: now[0])
    monkeypatch.setenv("SLURM_JOB_ID", "1")

    def cuda(name, start, limit, **options):
        monkeypatch.setenv("SLURM_JOB_START_TIME", str(start))
        monkeypatch.setenv("SLURM_JOB_END_TIME", str(start + limit))
        now[0] = float(start)
        return _infer_released(
            released.release,
            released.provenance,
            released.source,
            root,
            "cuda",
            expected_counts=released.counts,
            reservation_path=reservation(
                tmp_path / name,
                root,
                reserved_gpu_seconds=float(limit),
                recorded_utc=name,
            ),
            **options,
        )

    with pytest.raises(ValueError, match="requires the ledger owner"):
        _infer_released(
            released.release, released.provenance, released.source, root, "cuda"
        )
    assert not root.exists()
    # Four predictions take 40 seconds; the last one lands on the limit.
    with pytest.raises(InterruptedError, match="reserved GPU-seconds are spent"):
        cuda("first.json", 1000, 40)
    assert len(list((root / "predictions").iterdir())) == 4
    assert not (root / SEAL).exists()

    # Replaying saved predictions does not seal past a spent reservation either.
    verify = phases.sealed_prediction

    def slow_verify(*args):
        now[0] += 10.0
        return verify(*args)

    with monkeypatch.context() as context:
        context.setattr(phases, "sealed_prediction", slow_verify)
        with pytest.raises(InterruptedError, match="reserved GPU-seconds are spent"):
            cuda("second.json", 2000, 40, resume=True)
    assert not (root / SEAL).exists()
    seal = cuda("third.json", 3000, 40, resume=True)
    assert seal["status"] == "predictions_sealed"
    with pytest.raises(ValueError, match="already sealed"):
        cuda("fourth.json", 4000, 40, resume=True)
    assert len(read_invocations(root)) == 3
    assert [row["status"] for row in read_invocations(root)] == [
        "interrupted",
        "interrupted",
        "complete",
    ]
    assert len(seal["invocations"]) == 3


SECOND_VIEW = "val-rally-00001"


def failing_fit(numeric, metadata):
    if metadata["artifact_id"].startswith(SECOND_VIEW):
        stop = time.process_time() + 0.2
        while time.process_time() < stop:
            pass
        raise RuntimeError("controlled fit failure")
    return phases.fit_report(numeric, metadata)


def dying_fit(numeric, metadata):
    if metadata["artifact_id"].startswith(SECOND_VIEW):
        os._exit(3)
    return phases.fit_report(numeric, metadata)


def warning_fit(numeric, metadata):
    os.kill(os.getppid(), signal.SIGUSR1)
    return phases.fit_report(numeric, metadata)


def test_worker_fits_reproduce_serial_analysis_and_fail_closed(
    released, tmp_path, monkeypatch
):
    views, single = released.counts["views"], released.single
    inference = tmp_path / "inference"
    infer(released, inference)
    forbid_inference(monkeypatch)

    def same_as_serial(result, root):
        assert result["metrics"] == single["metrics"]
        assert result["physical"] == single["physical"]
        assert without_timing(result["exported_result_metrics"]) == without_timing(
            single["exported_result_metrics"]
        )
        for index in range(views):
            name = f"physical/{index:05d}.json"
            assert (root / name).read_bytes() == (
                released.single_root / name
            ).read_bytes()

    for workers in (0, 9, 2.0):
        with pytest.raises(ValueError, match="physics workers must be"):
            analyze(released, inference, tmp_path / "bad", physics_workers=workers)
    # The scheduler's CPU grant bounds workers even under a wider affinity.
    with monkeypatch.context() as context:
        context.setenv("SLURM_CPUS_PER_TASK", "1")
        with pytest.raises(ValueError, match="physics workers must be"):
            analyze(released, inference, tmp_path / "bad", physics_workers=2)
    assert not (tmp_path / "bad").exists()

    parallel = tmp_path / "parallel"
    same_as_serial(analyze(released, inference, parallel, physics_workers=2), parallel)
    run = read_invocations(parallel)[0]
    assert run["physics"] == {
        "workers": 2,
        "outstanding_views_limit": 4,
        "worker_start_method": "spawn",
        "cpu_affinity": len(os.sched_getaffinity(0)),
        "scheduler_cpus_per_task": None,
        "failed_fits_timed": 0,
        "fits_without_timing": 0,
    }
    assert run["counts"]["physical_created"] == views
    assert run["counts"]["predictions_reused"] == 2 * views
    assert "physics_fit" not in run["timings_s"]
    assert 0 < run["timings_s"]["physics_fit_worker_cpu"]
    assert 0 < run["timings_s"]["physics_fit_worker_wall"]
    assert run["timings_s"]["physics_worker_pass"] < run["wall_s"]
    assert run["process_cpu_s"] > 0 and run["cuda_initialized"] is False

    # A failed task stops the run; finished fits stay committed and a resume
    # computes only the rest, with any worker count. A dead worker may also
    # take the fits still running in the pool with it.
    for fit, resumed_workers in ((failing_fit, 1), (dying_fit, 2)):
        root = tmp_path / fit.__name__
        with pytest.raises(ValueError, match="failed in a worker"):
            analyze(released, inference, root, physics_workers=2, fit=fit)
        assert not (root / "metrics.json").exists()
        assert not (root / "physical-journal/00001.json").exists()
        committed = len(list((root / "physical-journal").glob("*.json")))
        result = analyze(
            released, inference, root, resume=True, physics_workers=resumed_workers
        )
        same_as_serial(result, root)
        runs = read_invocations(root)
        assert [row["status"] for row in runs] == ["failed", "complete"]
        assert runs[0]["counts"]["physical_created"] == committed
        # A raising fit keeps its cost; a dead worker's fits are counted unmeasured.
        first = runs[0]
        if fit is failing_fit:
            assert first["physics"]["failed_fits_timed"] == 1
            assert first["physics"]["fits_without_timing"] == 0
            assert first["timings_s"]["physics_fit_worker_cpu"] > 0.2
        else:
            assert first["physics"]["fits_without_timing"] >= 1
        assert runs[1]["counts"]["physical_created"] == views - committed

    # The scheduler warning keeps finished fits and stops before scoring. The
    # start record on disk already states the worker configuration.
    root = tmp_path / "warned"
    started = {}

    def record_start(*args):
        started.update(read_json(root / "invocations/0000.json"))
        return fit_ahead(*args)

    fit_ahead = phases.fit_ahead
    with monkeypatch.context() as context:
        context.setattr(phases, "fit_ahead", record_start)
        with pytest.raises(InterruptedError, match="resume the analysis"):
            analyze(released, inference, root, physics_workers=2, fit=warning_fit)
    assert started["status"] == "running" and started["physics"]["workers"] == 2
    assert not (root / "metrics.json").exists()
    assert not (root / "physical").exists()
    same_as_serial(
        analyze(released, inference, root, resume=True, physics_workers=2), root
    )
    runs = read_invocations(root)
    assert [row["status"] for row in runs] == ["interrupted", "complete"]
    assert sum(row["counts"]["physical_created"] for row in runs) == views


def test_phase_commands_report_unavailable_inputs(tmp_path, monkeypatch, capsys):
    from tennis3d.evaluation.delivery import main

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    source = FIXTURES / "val.manifest.json"
    common = ["--release", str(tmp_path / "none.json"), "--source", str(source)]
    assert main(["synthetic-inference", *common, "--output", str(tmp_path / "i")]) == 2
    assert (
        main(
            [
                "synthetic-analysis",
                *common,
                "--inference",
                str(tmp_path / "i"),
                "--output",
                str(tmp_path / "a"),
            ]
        )
        == 2
    )
    assert "Evaluation unavailable" in capsys.readouterr().out
    assert not (tmp_path / "i").exists() and not (tmp_path / "a").exists()
