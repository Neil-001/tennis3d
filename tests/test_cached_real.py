"""Cached-input real evaluation on a controlled gallery, with tiny real EMA models."""

import importlib.util
import shutil
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from tennis3d.artifacts import (
    ArtifactProvenance,
    file_sha256,
    json_sha256,
    read_json,
    read_npz,
    write_json,
    write_npz,
)
from tennis3d.config import load_config
from tennis3d.court.model import court_landmarks
from tennis3d.evaluation.contact_delivery import read_gallery_clip
from tennis3d.evaluation.inventory import Clip, Inventory
from tennis3d.geometry.camera import Camera, ImageSize
from tennis3d.geometry.projection import project
from tennis3d.inference.export import export
from tennis3d.inference.inputs import detection_inputs, hashes
from tennis3d.inference.lift import lift
from tennis3d.inference.settings import ReconstructionConfig
from tennis3d.lifting.model import ModelConfig, build_lifter
from tennis3d.perception.provenance import (
    CALIBRATION_RUNTIME,
    DETECTION_RUNTIME,
    code_identity,
    runtime_versions,
)
from tennis3d.provenance import code_identity as package_identity

SCRIPT = Path(__file__).parents[1] / "scripts/cached_real.py"
SPEC = importlib.util.spec_from_file_location("cached_real", SCRIPT)
assert SPEC and SPEC.loader
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)

OLD_PRODUCER = "python-source-sha256:" + "0" * 64
N = 60


def checkpoint(path, model_config, seed):
    torch.manual_seed(seed)
    model = build_lifter(model_config)
    # Seed-dependent rows near one metre high, plausible for the quality rules.
    readout = model.head[-1]
    assert isinstance(readout, torch.nn.Linear)
    with torch.no_grad():
        readout.weight.mul_(0.01)
        readout.bias.copy_(torch.tensor([0.0, 0.0, 0.1]))
    torch.save(
        {
            "schema_version": "1.0",
            "world_scale_m": 10.0,
            "step": 0,
            "config": {"model": model_config.model_dump()},
            "ema": {
                "module." + key: value for key, value in model.state_dict().items()
            },
        },
        path,
    )
    return SimpleNamespace(
        checkpoint=SimpleNamespace(path=str(path), sha256=file_sha256(path))
    )


def cached_gallery(
    tmp_path,
    *,
    producer=OLD_PRODUCER,
    camera_producer=None,
    cut=False,
    recorded_source=None,
    numpy_version=None,
    gallery_threads=None,
):
    """One clip of detector/calibration output as an earlier package left it."""
    source = tmp_path / "controlled.mp4"
    source.write_bytes(b"controlled fixture, not real media")
    source_hash = recorded_source or file_sha256(source)
    pts = 1000 + 40 * np.arange(N)
    times = (pts - pts[0]) / 1000.0
    rows = [
        {"frame_index": i, "pts": int(p), "time_base": [1, 1000]}
        for i, p in enumerate(pts)
    ]
    write_json(
        tmp_path / "timeline.json",
        {
            "source_sha256": file_sha256(source),
            "image_size": [640, 480],
            "frames": rows,
        },
    )
    clip = Clip(
        clip_id="controlled",
        recording_group="controlled-recording",
        recording_evidence="controlled fixture only",
        distinct_recording_verified=True,
        source_url="fixture",
        source_sha256=file_sha256(source),
        source_bytes=source.stat().st_size,
        source_path=str(source),
        frame_count=N,
        width=640,
        height=480,
        time_basis="source_pts",
        timeline_path=str(tmp_path / "timeline.json"),
        timeline_sha256=file_sha256(tmp_path / "timeline.json"),
        timing_note="Controlled integer PTS",
        local_use="permitted",
        redistribution="restricted",
        rights_evidence="Fixture",
        label_status="unavailable",
        detector_training_independence="unknown",
        categories=["static"],
        category_evidence={"static": "Controlled fixture"},
        eligible=True,
        barriers=[],
    )
    unavailable = clip.model_copy(
        update={"clip_id": "barred", "eligible": False, "barriers": ["no access"]}
    )
    write_json(
        tmp_path / "inventory.json",
        Inventory(
            frozen_utc="2026-10-09",
            intended_role="resolution_stress",
            selection_rule="Controlled fixture",
            excluded_development_recording_groups=["development"],
            excluded_development_source_sha256=["b" * 64],
            exclusion_evidence="controlled",
            clips=[clip, unavailable],
            limitations=["Not real evaluation data"],
        ).model_dump(),
    )
    config = ReconstructionConfig()
    gallery = tmp_path / "gallery"
    run = gallery / "clips" / clip.clip_id
    run.mkdir(parents=True)
    states = np.ones(N, dtype=np.int8)
    states[:2] = 0
    states[3] = 3
    states[30:50] = 0
    duplicate = np.full(N, -1, dtype=np.int64)
    duplicate[3] = 2
    observed = states == 1
    ball = np.column_stack([200.0 + 3 * np.arange(N), 150.0 + np.arange(N)])
    ball[~observed] = np.nan
    frames = {
        "frame_index": np.arange(N),
        "t_s": times,
        "pts": pts,
        "time_base": np.tile([1, 1000], (N, 1)),
        "time_origin_s": np.array(1.0),
        "image_size": np.array([640, 480]),
        "duplicate_of": duplicate,
        "cut": np.zeros(N, dtype=bool),
    }
    frames["cut"][40] = cut
    detect_config = config.detection.model_dump(mode="json")
    provenance = ArtifactProvenance(
        source_sha256=source_hash,
        config_sha256=json_sha256(detect_config),
        upstream_sha256={"ball": "b" * 64, "court": "c" * 64},
        code_revision=producer,
    )
    write_json(run / "detect.config.json", detect_config)
    write_npz(run / "frames.npz", frames, provenance)
    write_npz(run / "court.npz", {"raw_uv": np.zeros((N, 14, 2))}, provenance)
    write_npz(run / "ball_candidates.npz", {"candidates": np.empty((0, 4))}, provenance)
    write_npz(
        run / "ball_track.npz",
        {"ball_uv": ball, "observation": states, "ball_observed": observed},
        provenance,
    )
    runtime = runtime_versions(DETECTION_RUNTIME)
    if numpy_version:
        runtime["numpy"] = numpy_version
    write_json(
        run / "detect.json",
        {
            "source": "controlled.mp4",
            "source_sha256": source_hash,
            "frame_count": N,
            "config_sha256": provenance.config_sha256,
            "assets": {
                "assets": {"ball": {"sha256": "b" * 64}, "court": {"sha256": "c" * 64}}
            },
            "artifacts": hashes(
                run,
                ("frames.npz", "court.npz", "ball_candidates.npz", "ball_track.npz"),
            ),
            "status": "failed" if cut else "partial",
            "exit_code": 2 if cut else 0,
            "failure": None,
            "shot_cut": cut,
            "device": "cpu",
            "runtime": runtime,
            "elapsed_s": 0.1,
            "peak_cuda_bytes": None,
        },
    )
    camera = Camera(
        np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]),
        np.diag([1, -1, -1]),
        np.array([0, 0, 10]),
        ImageSize(640, 480),
    )
    valid = np.ones(N, dtype=bool)
    valid[25] = False
    if cut:
        valid[40:] = False
    calibrate_config = config.calibration.model_dump(mode="json")
    write_json(run / "calibrate.config.json", calibrate_config)
    write_npz(
        run / "cameras.npz",
        {
            "t_s": times,
            "camera_valid": valid,
            "K": np.tile(camera.K, (N, 1, 1)),
            "R": np.tile(camera.R, (N, 1, 1)),
            "t": np.tile(camera.t, (N, 1)),
            "court_uv": np.tile(project(camera, court_landmarks())[0], (N, 1, 1)),
            "filtered_residual_px": np.zeros((N, 14)),
        },
        ArtifactProvenance(
            source_sha256=source_hash,
            config_sha256=json_sha256(calibrate_config),
            code_revision=camera_producer or producer,
            upstream_sha256=hashes(run, ("frames.npz", "court.npz", "detect.json")),
        ),
    )
    write_json(
        run / "calibrate.json",
        {
            "config_sha256": json_sha256(calibrate_config),
            "cameras_sha256": file_sha256(run / "cameras.npz"),
            "frame_count": N,
            "image_size": [640, 480],
            "mode": "estimated_static",
            "runtime": runtime_versions(CALIBRATION_RUNTIME),
            "assumptions": ["controlled fixture"],
            "elapsed_s": 0.2,
        },
    )
    # Outputs of the earlier model sit beside the caches and must not be reused.
    write_json(run / "lift.json", {"fixture": "earlier model"})
    recorded = config.model_dump(mode="json")
    if gallery_threads:
        recorded["detection"]["threads"] = gallery_threads
    write_json(gallery / "config.json", recorded)
    return SimpleNamespace(
        root=tmp_path,
        gallery=gallery,
        run=run,
        clip=clip,
        rows=rows,
        inventory=tmp_path / "inventory.json",
        manifest=tmp_path / "artifact-manifest.json",
    )


def release_manifest(fixture):
    files = {
        str(path.relative_to(fixture.gallery)): {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(fixture.gallery.rglob("*"))
        if path.is_file()
    }
    prefix = f"clips/{fixture.clip.clip_id}/"
    write_json(
        fixture.manifest,
        {
            "schema": "real-artifact-interface/v1",
            "original_inventory_sha256": file_sha256(fixture.inventory),
            "files": files,
            "clips": [
                {
                    "clip_id": fixture.clip.clip_id,
                    "artifacts": {
                        name.removeprefix(prefix): entry
                        for name, entry in files.items()
                        if name.startswith(prefix)
                    },
                }
            ],
        },
        overwrite=True,
    )
    return file_sha256(fixture.manifest)


def run_adapter(fixture, release, output, monkeypatch, **options):
    provenance = {"release": {"checkpoint": {"sha256": release.checkpoint.sha256}}}
    monkeypatch.setattr(
        adapter, "checked_release", lambda _: (release, None, provenance)
    )
    if "manifest_sha256" not in options:
        options["manifest_sha256"] = release_manifest(fixture)
    return adapter.cached_real(
        fixture.root / "unused-release.json",
        fixture.inventory,
        fixture.gallery,
        fixture.manifest,
        options.pop("manifest_sha256"),
        output,
        **options,
    )


def frozen_source(fixture):
    times = [Fraction(row["pts"], 1000) for row in fixture.rows]
    return {
        "clip": {
            "source_sha256": fixture.clip.source_sha256,
            "width": 640,
            "height": 480,
        },
        "t_s": np.array([float(value) for value in times]),
        "frames": fixture.rows,
    }


def tree(root):
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(Path(root).rglob("*"))
        if path.is_file()
    }


def forbid_perception(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("cached-input evaluation must not rerun perception")

    for module, name in (
        ("tennis3d.perception.detect", "detect"),
        ("tennis3d.perception.calibrate", "calibrate"),
        ("tennis3d.inference.pipeline", "reconstruct"),
    ):
        monkeypatch.setattr(f"{module}.{name}", forbidden)


def test_adapter_equals_stock_stages_on_the_same_cached_bytes(tmp_path, monkeypatch):
    # A current-package producer lets the stock stages run as the reference.
    fixture = cached_gallery(tmp_path, producer=code_identity())
    release = checkpoint(tmp_path / "transformer.pt", ModelConfig(width=16, depth=2), 1)
    stock = tmp_path / "stock"
    shutil.copytree(fixture.run, stock)
    (stock / "lift.json").unlink()
    config = load_config(
        ReconstructionConfig,
        fixture.gallery / "config.json",
        overrides={"lifting": {"checkpoint": release.checkpoint.path, "device": "cpu"}},
    )
    lift(stock, config.lifting)
    export(stock, config.quality)
    before = tree(fixture.gallery)
    forbid_perception(monkeypatch)
    result = run_adapter(fixture, release, tmp_path / "new", monkeypatch)
    assert tree(fixture.gallery) == before
    run = tmp_path / "new/clips/controlled"
    expected, _ = read_npz(stock / "predictions_raw.npz")
    actual, identity = read_npz(run / "predictions_raw.npz")
    assert expected["covered"].any() and not expected["covered"][25]
    for name, values in expected.items():
        assert values.tobytes() == actual[name].tobytes()
    assert identity.upstream_sha256 == hashes(run, adapter.LIFT_INPUTS) | {
        "checkpoint": release.checkpoint.sha256
    }
    reference, produced = (
        read_json(root / "result.json")["frames"] for root in (stock, run)
    )
    assert reference == produced
    assert any(row["xyz_m"] for row in produced)
    assert (
        read_json(stock / "export.json")["coverage"]
        == (read_json(run / "export.json")["coverage"])
    )
    assert result["schema_version"] == "1.0" and result["kind"] == "end_to_end_real"
    records = {row["clip_id"]: row for row in result["records"]}
    assert records["barred"]["status"] == "ineligible"
    assert records["barred"]["failure_reasons"] == ["no access"]
    record = records["controlled"]
    assert record["status"] == read_json(stock / "export.json")["status"] == "partial"
    assert [record[key] for key in ("tracking", "landing", "labelled_lifting")] == [
        None
    ] * 3
    assert read_json(run / "physical.json")["metadata"]["mode"] == "end_to_end"
    assert len(result["physical"]["cohorts"]) == 1


def test_old_producer_is_bound_verbatim_and_lstm_output_feeds_contact_loader(
    tmp_path, monkeypatch
):
    fixture = cached_gallery(tmp_path)
    with pytest.raises(ValueError, match="provenance differs"):
        detection_inputs(fixture.run)
    release = checkpoint(
        tmp_path / "lstm.pt",
        ModelConfig(
            architecture_version="stage4-lstm-v1",
            width=16,
            depth=2,
            lstm_hidden_size=8,
            gated_head=True,
        ),
        2,
    )
    forbid_perception(monkeypatch)
    package = package_identity()["source_sha256"]
    output = tmp_path / "new"
    result = run_adapter(
        fixture, release, output, monkeypatch, package_source_sha256=package
    )
    run = output / "clips/controlled"
    upstream = {
        name: file_sha256(fixture.run / name) for name in adapter.UPSTREAM_FILES
    }
    assert hashes(run, adapter.UPSTREAM_FILES) == upstream
    assert read_json(run / "lift.json")["checkpoint_config"]["model"][
        "architecture_version"
    ] == ("stage4-lstm-v1")
    cached = result["provenance"]["cached_inputs"]
    assert cached["detection_rerun"] is False and cached["calibration_rerun"] is False
    assert cached["upstream_producer_code_revisions"] == [OLD_PRODUCER]
    assert cached["numerical_package"]["source_sha256"] == package
    assert cached["numerical_package"]["stage_code_identity"] == code_identity()
    assert cached["adapter"]["script_sha256"] == file_sha256(SCRIPT)
    assert set(cached["measured_timings_s"]) == {
        "load_s",
        "predict_s",
        "lift_wall_s",
        "export_wall_s",
        "physical_fit_s",
        "clip_wall_s",
        "physical_summary_s",
    }
    for name in ("lift.json", "export.json", "metrics.json", "manifest.json"):
        bound = read_json(run / name)["cached_inputs"]
        assert bound["upstream_sha256"] == upstream
        assert bound["upstream_producer"]["code_revision"] == OLD_PRODUCER
        assert bound["artifact_manifest_sha256"] == file_sha256(fixture.manifest)
        assert bound["adapter"] == cached["adapter"]
        assert bound["numerical_package"]["source_sha256"] == package
    execution = read_json(run / "metrics.json")
    assert set(execution["runtime_s"]) == {"lift"}
    assert execution["historical_preprocessing"]["runtime_s"] == {
        "detect": 0.1,
        "calibrate": 0.2,
    }
    # New arrays and results name the importing package, never the old producer.
    _, raw_identity = read_npz(run / "predictions_raw.npz")
    assert raw_identity.code_revision == code_identity()
    assert read_json(run / "result.json")["provenance"]["code_revision"] == (
        code_identity()
    )
    # The unchanged contact loader accepts the new root and only its own model.
    gallery_config = read_json(output / "config.json")
    assert json_sha256(gallery_config) == result["config_sha256"]
    numeric, artifacts = read_gallery_clip(
        run,
        frozen_source(fixture),
        release.checkpoint.sha256,
        expected_config=gallery_config,
    )
    assert not artifacts["absent_artifacts"]
    assert numeric["raw"]["available"].sum() > numeric["exported"]["available"].sum()
    other = checkpoint(tmp_path / "other.pt", ModelConfig(width=16, depth=2), 3)
    with pytest.raises(ValueError, match="checkpoint"):
        read_gallery_clip(run, frozen_source(fixture), other.checkpoint.sha256)
    mixed = run_adapter(fixture, other, tmp_path / "other", monkeypatch)
    assert mixed["records"][0]["status"] == "partial"
    shutil.copyfile(
        tmp_path / "other/clips/controlled/predictions_raw.npz",
        run / "predictions_raw.npz",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        read_gallery_clip(run, frozen_source(fixture), release.checkpoint.sha256)


@pytest.mark.parametrize(
    "case,message",
    [
        ("manifest_hash", "artifact manifest hash"),
        ("inventory", "inventory differs"),
        ("mutated", "differs from manifest"),
        ("absent_file", "unreadable"),
        ("missing_clip", "does not list"),
        ("source", "different source"),
        ("config", "configuration differs"),
        ("runtime", "runtime changed"),
        ("producer", "different producers"),
        ("package", "tennis3d source differs"),
    ],
)
def test_untrusted_cached_inputs_stop_the_whole_run(
    tmp_path, monkeypatch, case, message
):
    fixture = cached_gallery(
        tmp_path,
        recorded_source="e" * 64 if case == "source" else None,
        gallery_threads=7 if case == "config" else None,
        numpy_version="0.0" if case == "runtime" else None,
        camera_producer="another producer" if case == "producer" else None,
    )
    release = checkpoint(tmp_path / "model.pt", ModelConfig(width=16, depth=2), 1)
    options = {"manifest_sha256": release_manifest(fixture)}
    if case == "manifest_hash":
        options["manifest_sha256"] = "f" * 64
    elif case == "inventory":
        with fixture.inventory.open("ab") as stream:
            stream.write(b"\n")
    elif case == "mutated":
        with (fixture.run / "ball_track.npz").open("ab") as stream:
            stream.write(b"changed")
    elif case == "absent_file":
        (fixture.run / "cameras.npz").unlink()
    elif case == "missing_clip":
        manifest = read_json(fixture.manifest)
        manifest["files"] = {"config.json": manifest["files"]["config.json"]}
        manifest["clips"] = []
        write_json(fixture.manifest, manifest, overwrite=True)
        options["manifest_sha256"] = file_sha256(fixture.manifest)
    elif case == "package":
        options["package_source_sha256"] = "d" * 64
    with pytest.raises(adapter.CachedInputError, match=message):
        run_adapter(fixture, release, tmp_path / "new", monkeypatch, **options)
    assert not (tmp_path / "new/metrics.json").exists()
    assert (
        adapter.main(
            [
                *("--release", str(tmp_path / "absent.json")),
                *("--inventory", str(fixture.inventory)),
                *("--gallery", str(fixture.gallery)),
                *("--artifact-manifest", str(fixture.manifest)),
                *("--artifact-manifest-sha256", "f" * 64),
                *("--output", str(tmp_path / "cli")),
            ]
        )
        == 2
        and not (tmp_path / "cli").exists()
    )


def test_failed_and_unsupported_records_are_preserved(tmp_path, monkeypatch):
    fixture = cached_gallery(tmp_path, cut=True)
    release = checkpoint(tmp_path / "model.pt", ModelConfig(width=16, depth=2), 1)
    result = run_adapter(fixture, release, tmp_path / "new", monkeypatch)
    record = result["records"][0]
    run = tmp_path / "new/clips/controlled"
    assert record["status"] == "failed"
    assert "unsupported_shot" in record["failure_reasons"]
    assert "invalid_camera" in record["failure_reasons"]
    assert record["pipeline"]["exit_code"] == 2
    frames = read_json(run / "result.json")["frames"]
    assert len(frames) == N and all(row["xyz_m"] is None for row in frames)
    # Raw rows before the cut stay available for diagnosis, as in the stock lift.
    raw, _ = read_npz(run / "predictions_raw.npz")
    assert raw["covered"][:25].all() and not raw["covered"][40:].any()
    assert result["physical"]["cohorts"][0]["counts"]["source_frames"] == N
    # A model that cannot run fails its clip; the record and its reason remain.
    broken = tmp_path / "broken.pt"
    broken.write_bytes(b"not a checkpoint")
    release = SimpleNamespace(
        checkpoint=SimpleNamespace(path=str(broken), sha256=file_sha256(broken))
    )
    result = run_adapter(fixture, release, tmp_path / "broken", monkeypatch)
    assert [row["status"] for row in result["records"]] == ["failed", "ineligible"]
    assert "lifter checkpoint" in result["records"][0]["failure_reasons"][0]
    assert result["physical"] is None and result["status"] == "incomplete_inventory"


@pytest.mark.parametrize(
    "case,message",
    [
        ("predictions_after_lift", "Raw prediction hash mismatch"),
        ("predictions_during_export", "verified artifacts changed"),
        ("upstream_during_prediction", "Lifting inputs changed"),
        ("upstream_with_model_error", "verified artifacts changed"),
        ("checkpoint_with_model_error", "checkpoint changed"),
        ("upstream_before_publication", "verified artifacts changed"),
        ("root_config_before_publication", "['config.json']"),
        ("root_inventory_before_publication", "['inventory.json']"),
    ],
)
def test_integrity_failures_abort_without_published_metrics(
    tmp_path, monkeypatch, capsys, case, message
):
    from tennis3d.inference import predict

    fixture = cached_gallery(tmp_path)
    release = checkpoint(tmp_path / "model.pt", ModelConfig(width=16, depth=2), 1)
    provenance = {"release": {"checkpoint": {"sha256": release.checkpoint.sha256}}}
    monkeypatch.setattr(
        adapter, "checked_release", lambda _: (release, None, provenance)
    )
    output = tmp_path / "new"
    run = output / "clips/controlled"

    def change(path):
        if Path(path).name == "config.json":
            # Still a valid config, so only its recorded hash can expose it.
            recorded = read_json(path)
            recorded["quality"]["bracketed_gap_s"] += 1.0
            write_json(path, recorded, overwrite=True)
            return
        with Path(path).open("ab") as stream:
            stream.write(b"changed")

    def after(module, name, path):
        original = getattr(module, name)

        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            change(path)
            return result

        monkeypatch.setattr(module, name, wrapped)

    def during_prediction(path, *, fail):
        original = predict.predict_view

        def wrapped(*args, **kwargs):
            change(path)
            if fail:
                raise RuntimeError("controlled numerical failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(predict, "predict_view", wrapped)

    if case == "predictions_after_lift":
        after(adapter, "lift_cached", run / "predictions_raw.npz")
    elif case == "predictions_during_export":
        after(adapter, "export_cached", run / "predictions_raw.npz")
    elif case == "upstream_during_prediction":
        during_prediction(run / "frames.npz", fail=False)
    elif case == "upstream_with_model_error":
        during_prediction(run / "frames.npz", fail=True)
    elif case == "checkpoint_with_model_error":
        during_prediction(release.checkpoint.path, fail=True)
    elif case == "upstream_before_publication":
        after(adapter, "write_report", run / "cameras.npz")
    else:
        name = "config.json" if "config" in case else "inventory.json"
        after(adapter, "write_report", output / name)
    status = adapter.main(
        [
            *("--release", str(tmp_path / "unused-release.json")),
            *("--inventory", str(fixture.inventory)),
            *("--gallery", str(fixture.gallery)),
            *("--artifact-manifest", str(fixture.manifest)),
            *("--artifact-manifest-sha256", release_manifest(fixture)),
            *("--output", str(output)),
        ]
    )
    assert status == 2 and message in capsys.readouterr().out
    assert not (output / "metrics.json").exists()
    aborted = read_json(output / "aborted.json")
    assert aborted["status"] == "aborted" and message in aborted["error"]


def test_model_failure_on_intact_inputs_remains_a_clip_record(tmp_path, monkeypatch):
    from tennis3d.inference import predict

    def failing(*args, **kwargs):
        raise RuntimeError("controlled numerical failure")

    monkeypatch.setattr(predict, "predict_view", failing)
    fixture = cached_gallery(tmp_path)
    release = checkpoint(tmp_path / "model.pt", ModelConfig(width=16, depth=2), 1)
    result = run_adapter(fixture, release, tmp_path / "new", monkeypatch)
    assert result["records"][0]["status"] == "failed"
    assert result["records"][0]["failure_reasons"] == [
        "RuntimeError: controlled numerical failure"
    ]
    assert read_json(tmp_path / "new/metrics.json")["records"] == result["records"]
    assert not (tmp_path / "new/aborted.json").exists()
