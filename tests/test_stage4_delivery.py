"""Delivery preparation never reads held-out arrays or invents real eligibility."""

import json
from pathlib import Path

import numpy as np
import pytest

from tennis3d.artifacts import file_sha256, write_json
from tennis3d.evaluation.delivery import checked_release, plan
from tennis3d.evaluation.inventory import Clip, Inventory, verify_clip

ROOT = Path(__file__).resolve().parents[1]


def clip(tmp_path):
    source = tmp_path / "controlled.mp4"
    source.write_bytes(b"controlled fixture, not real media")
    timeline = tmp_path / "timeline.json"
    write_json(
        timeline,
        {
            "source_sha256": file_sha256(source),
            "image_size": [640, 480],
            "frames": [
                {"frame_index": i, "pts": 100 + i * 7, "time_base": [1, 100]}
                for i in range(3)
            ],
        },
    )
    return Clip(
        clip_id="controlled",
        recording_group="controlled-recording",
        recording_evidence="controlled fixture only",
        distinct_recording_verified=True,
        source_url="fixture",
        source_sha256=file_sha256(source),
        source_bytes=source.stat().st_size,
        source_path=str(source),
        frame_count=3,
        width=640,
        height=480,
        time_basis="source_pts",
        timeline_path=str(timeline),
        timeline_sha256=file_sha256(timeline),
        timing_note="Controlled integer PTS",
        local_use="permitted",
        redistribution="restricted",
        rights_evidence="Fixture; redistribution independent of evaluation eligibility",
        label_status="unavailable",
        detector_training_independence="unknown",
        categories=["static"],
        category_evidence={"static": "Controlled fixture"},
        eligible=True,
        barriers=[],
    )


def inventory(selected):
    return Inventory(
        frozen_utc="2026-10-07",
        intended_role="resolution_stress",
        selection_rule="Controlled fixture",
        excluded_development_recording_groups=["development"],
        excluded_development_source_sha256=["b" * 64],
        exclusion_evidence="controlled",
        clips=selected,
        limitations=["Not real evaluation data"],
    )


def test_metadata_only_plan_and_development_separation(tmp_path, monkeypatch):
    from tennis3d.lifting.data import ViewDataset

    def forbidden(*args, **kwargs):
        raise AssertionError("preparation opened test arrays")

    monkeypatch.setattr(ViewDataset, "__init__", forbidden)
    selected = clip(tmp_path)
    path = tmp_path / "inventory.json"
    write_json(path, inventory([selected]).model_dump())
    output = tmp_path / "plan"
    result = plan(ROOT / "configs/release-pins-main-v1.json", path, output)
    assert result["test_arrays_opened"] is False
    assert result["expected_test_counts"] == {
        "rallies": 5000,
        "views": 10000,
        "frames": 1829263,
    }
    spec = json.loads((output / "test.source.json").read_text())
    assert spec["split"] == "test" and spec["cache_bytes"] == 1 << 30
    assert not result["real_inventory"]["required_gallery_ready"]
    with pytest.raises(ValueError, match="already exists"):
        plan(ROOT / "configs/release-pins-main-v1.json", path, output)
    with pytest.raises(ValueError, match="development"):
        inventory([selected.model_copy(update={"recording_group": "development"})])
    with pytest.raises(ValueError, match="eligible clip"):
        Clip.model_validate({**selected.model_dump(), "time_basis": "unavailable"})


def test_source_timeline_hash_and_identity_are_checked(tmp_path):
    selected = clip(tmp_path)
    assert len(verify_clip(selected)["frames"]) == 3
    timeline = Path(selected.timeline_path)
    rows = json.loads(timeline.read_text())
    rows["frames"][1]["pts"] = rows["frames"][0]["pts"]
    timeline.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="timeline changed"):
        verify_clip(selected)
    selected = selected.model_copy(update={"timeline_sha256": file_sha256(timeline)})
    with pytest.raises(ValueError, match="strictly increase"):
        verify_clip(selected)


def test_validation_selection_and_final_compute_receipt(tmp_path, monkeypatch):
    from tennis3d.lifting import checkpoint as module

    source_identity = {"split": "val", "membership_sha256": "c" * 64}
    state = {
        "run_id": "controlled-fixture",
        "step": 1,
        "best_step": 1,
        "best_score": 2.0,
        "dataset_identity": {"val": source_identity},
        "manifest_sha256": {"val": "d" * 64},
        "budget": {"train_seconds": 1.0, "job_seconds": 2.0},
        "sampler": {"draws": 2, "rejected": 0, "epoch": 0, "cursor": 2},
        "provenance": {"kind": "fixture"},
        "config": {"artifact_label": "fixture"},
    }
    monkeypatch.setattr(module, "read_checkpoint", lambda path: state)
    files = {}
    for name, value in {
        "checkpoint": {"fixture": "not model weights"},
        "final_budget": {
            "step": 2,
            "run_id": state["run_id"],
            "train_seconds": 3.0,
            "job_seconds": 4.0,
        },
        "exposure_receipt": {
            "run_id": state["run_id"],
            "final_step": 2,
            "sampled_views": 4,
        },
        "validation_history": {
            "sampled_views": 2,
            "train_seconds": 1.0,
            "job_seconds": 2.0,
            "run_id": state["run_id"],
            "step": 1,
            "source_identity": source_identity,
            "manifest_sha256": "d" * 64,
            "modes": {"corrupted": {"val_xyz_m": 2.0}},
        },
    }.items():
        path = tmp_path / f"{name}.json"
        write_json(path, value)
        files[name] = {"path": str(path), "sha256": file_sha256(path)}
    receipt: dict = {
        "status": "selected_best_released",
        "released_by": "controlled-fixture",
        "released_utc": "2026-10-07",
        "run_id": state["run_id"],
        "step": 1,
        "selection": "minimum_reserved_validation_corrupted_rally_mean",
        **files,
    }
    path = tmp_path / "release.json"
    write_json(path, receipt)
    _, _, provenance = checked_release(path)
    assert provenance["actual_final_budget"]["job_seconds"] == 4.0
    history = Path(files["validation_history"]["path"])
    first = json.loads(history.read_text())

    def replace_history(rows):
        history.write_text("".join(json.dumps(row) + "\n" for row in rows))
        receipt["validation_history"]["sha256"] = file_sha256(history)
        write_json(path, receipt, overwrite=True)

    replace_history(
        [
            first,
            {
                **first,
                "step": 3,
                "sampled_views": 6,
                "train_seconds": 5.0,
                "job_seconds": 8.0,
                "modes": {"corrupted": {"val_xyz_m": 3.0}},
            },
        ]
    )
    with pytest.raises(ValueError, match="final exposure"):
        checked_release(path)
    replace_history(
        [
            first,
            {
                **first,
                "source_identity": {"foreign": "validation"},
                "modes": {"corrupted": {"val_xyz_m": 3.0}},
            },
        ]
    )
    with pytest.raises(ValueError, match="every history row"):
        checked_release(path)
    replace_history([first, {**first, "train_seconds": 2.0, "job_seconds": 3.0}])
    checked_release(path)  # A repeated/replayed validation step remains valid.
    replace_history(
        [
            first,
            {
                **first,
                "train_seconds": 5.0,
                "job_seconds": 8.0,
                "modes": {"corrupted": {"val_xyz_m": 3.0}},
            },
        ]
    )
    with pytest.raises(ValueError, match="final charged compute"):
        checked_release(path)
    replace_history([first])
    with history.open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "run_id": state["run_id"],
                    "step": 2,
                    "sampled_views": 4,
                    "train_seconds": 2.0,
                    "job_seconds": 3.0,
                    "source_identity": source_identity,
                    "manifest_sha256": "d" * 64,
                    "modes": {"corrupted": {"val_xyz_m": 1.0}},
                }
            )
            + "\n"
        )
    receipt["validation_history"]["sha256"] = file_sha256(history)
    write_json(path, receipt, overwrite=True)
    with pytest.raises(ValueError, match="validation-selected best"):
        checked_release(path)


def test_synthetic_callback_receives_each_complete_fixture_mode():
    import torch

    from tennis3d.evaluation.synthetic import evaluate_model
    from tennis3d.lifting.model import Lifter, ModelConfig

    torch.set_num_threads(2)
    model = Lifter(ModelConfig(width=16, depth=2))
    seen = []

    def on_view(record, arrays, inputs, xyz, covered, mode, diagnostics):
        assert diagnostics["prediction_wall_s"] >= 0
        assert diagnostics["disagreement_m"].shape == covered.shape
        assert xyz.shape == arrays["xyz_m"].shape
        assert covered[arrays["label_valid"] & arrays["camera_valid"]].all()
        if mode == "clean":
            np.testing.assert_array_equal(inputs["ball_uv"], arrays["ball_clean_uv"])
        seen.append((record["rally_id"], record["view_id"], mode))

    result = evaluate_model(
        model, ROOT / "tests/fixtures/lifting/val.manifest.json", on_view=on_view
    )
    assert [row[2] for row in seen] == ["corrupted", "clean"]
    assert result["modes"]["corrupted"]["valid_coverage"] == 1.0


def test_exported_accuracy_keeps_denominator_when_quality_rejects():
    from tennis3d.evaluation.delivery import ExportedMetrics

    n = 3
    arrays = {
        "t_s": np.arange(n) / 25.0,
        "camera_valid": np.ones(n, dtype=bool),
        "label_valid": np.ones(n, dtype=bool),
        "xyz_m": np.zeros((n, 3)),
        "camera_R": np.tile(np.eye(3), (n, 1, 1)),
        "camera_t": np.tile([0, 0, 10.0], (n, 1)),
    }
    prediction = np.tile([0.0, 0.0, -1.0], (n, 1))
    inputs = {"ball_observed": np.ones(n, dtype=bool)}
    covered = np.ones(n, dtype=bool)
    report = ExportedMetrics()
    for mode in ("clean", "corrupted"):
        report.add(
            arrays,
            inputs,
            prediction,
            covered,
            mode,
            {"disagreement_m": np.zeros(n), "prediction_wall_s": 0.01},
        )
    result = report.finish()
    for mode in result["modes"].values():
        assert mode["supported_frames"] == n
        assert mode["count"] == 0 and mode["mean_m"] is None
        assert mode["supported_coverage"] == 0
        assert mode["reason_counts_all_source_rows"] == {"implausible_prediction": n}


def test_real_label_adapter_preserves_single_reference_authority(tmp_path, monkeypatch):
    physical = pytest.importorskip("tennis3d.evaluation.physical")
    from test_reconstruction import prepared_run

    from tennis3d.evaluation import delivery
    from tennis3d.evaluation.annotations import Annotation, AnnotationManifest
    from tennis3d.inference import pipeline
    from tennis3d.inference.export import export
    from tennis3d.inference.lift import lift
    from tennis3d.inference.quality import QualityConfig
    from tennis3d.inference.settings import ReconstructionConfig

    run = tmp_path / "controlled-run"
    lifting, frames = prepared_run(run)
    lift(run, lifting)
    summary = export(run, QualityConfig())
    labels = AnnotationManifest(
        source_sha256="a" * 64,
        labels_sha256="f" * 64,
        clip_id="controlled",
        recording_group="controlled-recording",
        split="test",
        width=640,
        height=480,
        source_frame_count=520,
        time_basis="source_pts",
        timing_note="Controlled fixture only",
        pixel_convention="ball center",
        rights_and_split_note="Controlled fixture only",
        annotations=[
            Annotation(
                frame_index=10,
                image_name="10.jpg",
                t_s=float(frames["t_s"][10]),
                visibility=1,
                uv=[320.0, 240.0],
                event="bounce",
            )
        ],
    )
    labels_path = tmp_path / "labels.json"
    write_json(labels_path, labels.model_dump())
    selected = clip(tmp_path).model_copy(
        update={
            "source_sha256": "a" * 64,
            "frame_count": 520,
            "labels_path": str(labels_path),
            "labels_sha256": file_sha256(labels_path),
            "label_status": "normalized_source_pts",
        }
    )
    frozen = {
        "frames": [
            {"frame_index": i, "pts": int(pts), "time_base": basis.tolist()}
            for i, (pts, basis) in enumerate(zip(frames["pts"], frames["time_base"]))
        ]
    }
    monkeypatch.setattr(delivery, "verify_clip", lambda clip: frozen)
    monkeypatch.setattr(pipeline, "reconstruct", lambda *args: summary)
    original = physical.evaluate_physical
    seen = []

    def checked(arrays, **kwargs):
        assert "reference_uv_px" not in arrays and "reference_valid" not in arrays
        assert not any(key.startswith("event_") for key in arrays)
        assert kwargs["metadata"]["bounce_convention"] == "ball_center"
        seen.append(kwargs["metadata"]["mode"])
        # Exercise actual identity/contact/reference validation without costly fits.
        return original(arrays, config=physical.FitConfig(min_frames=1000), **kwargs)

    monkeypatch.setattr(physical, "evaluate_physical", checked)
    config = ReconstructionConfig(lifting=lifting)
    record, report = delivery.evaluate_real_clip(selected, config, run)
    assert report["reference"]["image"]["kind"] == "annotations"
    assert record["tracking"]["all"]["labelled_frames"] == 1
    assert record["landing"]["annotated_bounce_rows"] == 1
    assert record["labelled_lifting"]["kind"] == "manual_2D_input_lifting_diagnostic"
    assert "long_gap" in record["failure_reasons"]
    delivery.write_gallery(tmp_path, {"records": [{"clip_id": "controlled", **record}]})
    assert "long_gap" in (tmp_path / "gallery.html").read_text()
    assert seen == ["end_to_end", "manual_2D_input"]


def test_private_package_keeps_side_by_side_controls_and_file_hashes(tmp_path):
    from tennis3d.evaluation.delivery import package

    synthetic, real = tmp_path / "synthetic", tmp_path / "real"
    synthetic.mkdir()
    real.mkdir()
    release = {"checkpoint": {"sha256": "a" * 64}}
    write_json(
        synthetic / "metrics.json",
        {"kind": "model_only_synthetic", "provenance": {"release": release}},
    )
    write_json(
        real / "metrics.json",
        {
            "kind": "end_to_end_real",
            "provenance": {"release": release},
            "records": [],
            "inventory": {"required_gallery_ready": False},
        },
    )
    # Explicit controlled fixture report; no measured model/test result invented.
    write_json(
        synthetic / "physical" / "00000.json",
        {
            "schema_version": "1.0",
            "physics": {},
            "config": {},
            "definitions": {},
            "metadata": {
                "artifact_id": "controlled-only",
                "domain": "controlled_fixture",
                "split": "val",
                "time_basis": "simulation",
                "segmentation_source": "simulator",
                "segmentation_method": "fixture",
            },
            "counts": {"source_frames": 4, "fitted_segments": 0},
            "reason_counts": {},
            "segments": [],
        },
    )
    selected = clip(tmp_path)
    controlled_report = json.loads((synthetic / "physical" / "00000.json").read_text())
    for filename, mode in (
        ("physical.json", "end_to_end"),
        ("labelled-physical.json", "manual_2D_input"),
    ):
        report = {
            **controlled_report,
            "metadata": {
                **controlled_report["metadata"],
                "artifact_id": selected.clip_id,
                "mode": mode,
                "source_identity": {"source_sha256": selected.source_sha256},
                "prediction_identity": {"checkpoint_sha256": "a" * 64},
            },
        }
        write_json(real / "clips" / selected.clip_id / filename, report)
    value = json.loads((real / "metrics.json").read_text())
    value["records"] = [
        {
            "clip_id": selected.clip_id,
            "source": selected.model_dump(),
            "physical_path": f"clips/{selected.clip_id}/physical.json",
            "labelled_lifting": {
                "physical_path": f"clips/{selected.clip_id}/labelled-physical.json"
            },
        }
    ]
    write_json(real / "metrics.json", value, overwrite=True)
    viewer = tmp_path / "viewer.html"
    viewer.write_text("<!doctype html><title>Controlled viewer fixture</title>")
    output = tmp_path / "package"
    result = package(synthetic, real, viewer, output)
    controls = json.loads((output / "physical-controls.json").read_text())
    assert controls["cohorts"][0]["counts"]["source_frames"] == 4
    assert {cohort["mode"] for cohort in controls["cohorts"]} == {
        None,
        "end_to_end",
        "manual_2D_input",
    }
    assert (output / "real" / "clips" / selected.clip_id / "physical.json").is_file()
    assert (
        output / "real" / "clips" / selected.clip_id / "labelled-physical.json"
    ).is_file()
    assert result["files"]["report.md"] == file_sha256(output / "report.md")
    assert result["files"]["physical-controls.json"] == file_sha256(
        output / "physical-controls.json"
    )
    assert (output / "synthetic" / "physical" / "00000.json").read_bytes() == (
        synthetic / "physical" / "00000.json"
    ).read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        package(synthetic, real, viewer, output)


def review_bundle_inputs(tmp_path):
    """Controlled fixtures exercise packaging, never measured model outputs."""
    from tennis3d.artifacts import read_json

    synthetic, real = tmp_path / "synthetic", tmp_path / "real"
    selected = clip(tmp_path).model_copy(update={"frame_count": 9, "height": 360})
    release = {"checkpoint": {"sha256": "a" * 64}}
    write_json(
        synthetic / "metrics.json",
        {"kind": "model_only_synthetic", "provenance": {"release": release}},
    )
    data = read_json(ROOT / "viewer/example.json")
    data["source"]["sha256"] = selected.source_sha256
    data["source"]["name"] = "</script><script>window.injected=true</script>"
    data["provenance"]["lifter_checkpoint_sha256"] = "a" * 64
    write_json(real / "clips/controlled/result.json", data)
    failed = selected.model_copy(update={"clip_id": "failed_fixture"})
    rows = [
        {
            "clip_id": selected.clip_id,
            "source": selected.model_dump(),
            "status": "partial",
            "result_path": "clips/controlled/result.json",
            "failure_reasons": ["Fixture null gaps"],
        },
        {
            "clip_id": failed.clip_id,
            "source": failed.model_dump(),
            "status": "failed",
            "failure_reasons": ["Controlled calibration failure"],
        },
    ]
    metrics = {
        "kind": "end_to_end_real",
        "provenance": {"release": release},
        "inventory_sha256": "b" * 64,
        "inventory": {
            "required_gallery_ready": False,
            "intended_role": "representative_end_to_end",
        },
        "records": rows,
    }
    write_json(real / "metrics.json", metrics)
    viewer = tmp_path / "viewer.html"
    viewer.write_text("<!doctype html><title>Controlled viewer fixture</title>")
    contacts = tmp_path / "contacts"
    contact_rows = []
    for row in rows:
        values = {}
        for cohort in ("raw", "exported"):
            relative = f"clips/{row['clip_id']}/contacts-{cohort}.json"
            physical_paths = {}
            values[cohort] = {
                "contact_path": relative,
                "physical_paths": physical_paths,
            }
            write_json(contacts / relative, {"kind": "controlled_contact_fixture"})
            for segmentation in ("automatic", "manual_events"):
                relative = (
                    f"clips/{row['clip_id']}/physical-{cohort}-{segmentation}.json"
                )
                physical_paths[segmentation] = relative
                write_json(
                    contacts / relative,
                    {"kind": "controlled_ODE_fixture", "segmentation": segmentation},
                )
        contact_rows.append(
            {
                "clip_id": row["clip_id"],
                "source_identity": {
                    "source_sha256": selected.source_sha256,
                    "timeline_sha256": selected.timeline_sha256,
                },
                "artifact_identity": {
                    "files_sha256": {
                        "result.json": file_sha256(
                            real / "clips" / row["clip_id"] / "result.json"
                        )
                    }
                    if row.get("result_path")
                    else {},
                    "absent_artifacts": ["predictions_raw.npz"]
                    + ([] if row.get("result_path") else ["result.json"]),
                },
                "cohorts": values,
            }
        )
    write_json(
        contacts / "metrics.json",
        {
            "checkpoint_sha256": "a" * 64,
            "gallery_provenance": metrics["provenance"],
            "gallery_metrics_sha256": file_sha256(real / "metrics.json"),
            "snapshot": {"inventory_sha256": "b" * 64},
            "records": contact_rows,
        },
    )
    (contacts / "report.md").write_text("Controlled fixture; no measured contacts.")
    write_json(
        contacts / "files.json",
        {
            str(p.relative_to(contacts)): file_sha256(p)
            for p in contacts.rglob("*")
            if p.is_file()
        },
    )
    stress = tmp_path / "stress"
    stress_inventory = inventory([failed])
    write_json(stress / "inventory.json", stress_inventory.model_dump())
    write_json(
        stress / "metrics.json",
        {
            **metrics,
            "inventory_sha256": file_sha256(stress / "inventory.json"),
            "inventory": stress_inventory.summary(),
            "records": [rows[1]],
        },
    )
    (stress / "report.md").write_text("Separate controlled stress fixture.")
    return synthetic, real, viewer, contacts, stress


def test_portable_gallery_keeps_gaps_failures_and_separate_supplements(tmp_path):
    import re

    from tennis3d.artifacts import read_json
    from tennis3d.evaluation.delivery import package

    synthetic, real, viewer, contacts, stress = review_bundle_inputs(tmp_path)
    output = tmp_path / "package"
    result = package(
        synthetic,
        real,
        viewer,
        output,
        contacts_path=contacts,
        stress_path=stress,
    )
    gallery = (output / "real/gallery.html").read_text()
    assert gallery.count("<article>") == 2
    assert "Controlled calibration failure" in gallery
    assert "clips/controlled/viewer.html" in gallery
    assert "width=device-width" in gallery
    assert "default-src 'none'" in gallery
    page = (output / "real/clips/controlled/viewer.html").read_text()
    embedded_match = re.search(
        r'<script id="packaged-result" type="application/json">(.*?)</script>', page
    )
    assert embedded_match is not None
    embedded = embedded_match.group(1)
    original = read_json(real / "clips/controlled/result.json")
    assert json.loads(embedded) == original
    assert any(row["xyz_m"] is None for row in json.loads(embedded)["frames"])
    assert "<script>window.injected" not in page
    assert "new File(" in page and "fetch(" not in page
    assert (output / "real/clips/controlled/result.json").read_bytes() == (
        real / "clips/controlled/result.json"
    ).read_bytes()
    for relative, digest in read_json(contacts / "files.json").items():
        assert file_sha256(output / "contacts" / relative) == digest
    assert (
        "Separate resolution-stress gallery"
        in (output / "resolution-stress/gallery.html").read_text()
    )
    assert "not a paired resolution comparison" in (output / "report.md").read_text()
    manifest = read_json(output / "package.json")
    assert manifest["contact_supplement"] is True
    assert manifest["separate_resolution_stress_supplement"] is True
    assert all(
        file_sha256(output / path) == sha for path, sha in result["files"].items()
    )


@pytest.mark.parametrize("change", ["checkpoint", "gallery", "source", "file", "path"])
def test_package_rejects_changed_contact_join_before_creating_output(tmp_path, change):
    from tennis3d.artifacts import read_json
    from tennis3d.evaluation.delivery import package

    synthetic, real, viewer, contacts, _ = review_bundle_inputs(tmp_path)
    metrics = read_json(contacts / "metrics.json")
    if change == "checkpoint":
        metrics["checkpoint_sha256"] = "d" * 64
    elif change == "gallery":
        metrics["gallery_metrics_sha256"] = "d" * 64
    elif change == "source":
        metrics["records"][0]["source_identity"]["source_sha256"] = "d" * 64
    elif change == "file":
        (contacts / "report.md").write_text("Changed after frozen contact report.")
    else:
        metrics["records"][0]["cohorts"]["raw"]["contact_path"] = "../outside.json"
    write_json(contacts / "metrics.json", metrics, overwrite=True)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="contact"):
        package(synthetic, real, viewer, output, contacts_path=contacts)
    assert not output.exists()


def test_stress_supplement_cannot_relabel_the_primary_gallery(tmp_path):
    from tennis3d.artifacts import read_json
    from tennis3d.evaluation.delivery import package

    synthetic, real, viewer, _, stress = review_bundle_inputs(tmp_path)
    metrics = read_json(stress / "metrics.json")
    metrics["inventory_sha256"] = "b" * 64
    write_json(stress / "metrics.json", metrics, overwrite=True)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="own inventory"):
        package(synthetic, real, viewer, output, stress_path=stress)
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["valid_xyz", "previously_absent"])
def test_contacts_reject_gallery_artifact_changes_after_scoring(tmp_path, mutation):
    from tennis3d.artifacts import read_json
    from tennis3d.evaluation.delivery import package

    synthetic, real, viewer, contacts, _ = review_bundle_inputs(tmp_path)
    if mutation == "valid_xyz":
        path = real / "clips/controlled/result.json"
        data = read_json(path)
        data["frames"][0]["xyz_m"][0] += 1
        write_json(path, data, overwrite=True)
    else:
        (real / "clips/controlled/predictions_raw.npz").write_bytes(b"appeared")
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="contact.*(changed|appeared)"):
        package(synthetic, real, viewer, output, contacts_path=contacts)
    assert not output.exists()


def test_contacts_reject_copied_trajectory_change(tmp_path, monkeypatch):
    from tennis3d.artifacts import read_json
    from tennis3d.evaluation import review_bundle
    from tennis3d.evaluation.delivery import package

    synthetic, real, viewer, contacts, _ = review_bundle_inputs(tmp_path)
    original = review_bundle.write_playable_gallery

    def change_copy(root, result, viewer_path, **kwargs):
        original(root, result, viewer_path, **kwargs)
        path = root / "clips/controlled/result.json"
        data = read_json(path)
        data["frames"][0]["xyz_m"][0] += 1
        write_json(path, data, overwrite=True)

    monkeypatch.setattr(review_bundle, "write_playable_gallery", change_copy)
    with pytest.raises(ValueError, match="packaged contact gallery artifact differs"):
        package(synthetic, real, viewer, tmp_path / "partial", contacts_path=contacts)


def test_changed_result_copy_is_rejected_before_viewer_generation(
    tmp_path, monkeypatch
):
    from tennis3d.artifacts import read_json
    from tennis3d.evaluation import delivery

    synthetic, real, viewer, contacts, _ = review_bundle_inputs(tmp_path)
    original = delivery.shutil.copyfile

    def changed_copy(source, destination):
        copied = original(source, destination)
        destination = Path(destination)
        if destination.name == "result.json":
            data = read_json(destination)
            data["frames"][0]["xyz_m"][0] += 1
            write_json(destination, data, overwrite=True)
        return copied

    monkeypatch.setattr(delivery.shutil, "copyfile", changed_copy)
    output = tmp_path / "partial"
    with pytest.raises(ValueError, match="packaged contact gallery artifact differs"):
        delivery.package(synthetic, real, viewer, output, contacts_path=contacts)
    assert not (output / "real/clips/controlled/viewer.html").exists()
    assert not (output / "package.json").exists()


@pytest.mark.parametrize("mutation", ["replacement_evidence", "escape_path"])
def test_contact_copy_retains_the_validated_manifest(tmp_path, monkeypatch, mutation):
    from tennis3d.artifacts import read_json
    from tennis3d.evaluation import review_bundle
    from tennis3d.evaluation.delivery import package

    synthetic, real, viewer, contacts, _ = review_bundle_inputs(tmp_path)
    original = review_bundle.checked_supplements

    def replace_after_validation(*args, **kwargs):
        validated = original(*args, **kwargs)
        files = read_json(contacts / "files.json")
        if mutation == "replacement_evidence":
            report = contacts / "report.md"
            report.write_text("Different evidence and matching replacement manifest.")
            files["report.md"] = file_sha256(report)
        else:
            outside = tmp_path / "escaped.json"
            write_json(outside, {"kind": "controlled_escape_fixture"})
            files["../escaped.json"] = file_sha256(outside)
        write_json(contacts / "files.json", files, overwrite=True)
        return validated

    monkeypatch.setattr(review_bundle, "checked_supplements", replace_after_validation)
    output = tmp_path / "partial"
    with pytest.raises(ValueError, match="contact manifest changed"):
        package(synthetic, real, viewer, output, contacts_path=contacts)
    assert not (output / "escaped.json").exists()
    assert not (output / "package.json").exists()


@pytest.mark.parametrize("cohort", ["synthetic", "real", "resolution-stress"])
@pytest.mark.parametrize("timing", ["after_validation", "after_gallery"])
def test_package_metrics_retain_the_validation_snapshot(
    tmp_path, monkeypatch, cohort, timing
):
    from tennis3d.artifacts import read_json
    from tennis3d.evaluation import review_bundle
    from tennis3d.evaluation.delivery import package

    synthetic, real, viewer, contacts, stress = review_bundle_inputs(tmp_path)
    output = tmp_path / "partial"
    sources = {"synthetic": synthetic, "real": real, "resolution-stress": stress}

    def mutate(path):
        metrics = read_json(path)
        metrics["controlled_copy_mutation"] = True
        write_json(path, metrics, overwrite=True)

    if timing == "after_validation":
        original = review_bundle.checked_supplements

        def change_source(*args, **kwargs):
            validated = original(*args, **kwargs)
            mutate(sources[cohort] / "metrics.json")
            return validated

        monkeypatch.setattr(review_bundle, "checked_supplements", change_source)
    else:
        original = review_bundle.write_playable_gallery

        def change_copy(root, result, viewer_path, **kwargs):
            original(root, result, viewer_path, **kwargs)
            if root.name == ("real" if cohort == "synthetic" else cohort):
                mutate(output / cohort / "metrics.json")

        monkeypatch.setattr(review_bundle, "write_playable_gallery", change_copy)
    with pytest.raises(ValueError, match="packaged evaluation metrics changed"):
        package(
            synthetic,
            real,
            viewer,
            output,
            contacts_path=contacts,
            stress_path=stress,
        )
    assert not (output / "package.json").exists()


def test_stress_package_retains_exact_frozen_membership(tmp_path):
    from tennis3d.artifacts import read_json
    from tennis3d.evaluation.delivery import package

    synthetic, real, viewer, _, stress = review_bundle_inputs(tmp_path)
    frozen = inventory(
        [
            Clip.model_validate(row["source"])
            for row in read_json(real / "metrics.json")["records"]
        ]
    )
    write_json(stress / "inventory.json", frozen.model_dump(), overwrite=True)
    metrics = read_json(stress / "metrics.json")
    metrics.update(
        inventory_sha256=file_sha256(stress / "inventory.json"),
        inventory=frozen.summary(),
    )
    # The old one-record metrics cannot silently omit the other frozen clip.
    write_json(stress / "metrics.json", metrics, overwrite=True)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="every frozen inventory clip"):
        package(synthetic, real, viewer, output, stress_path=stress)
    assert not output.exists()


@pytest.mark.parametrize("substituted", [False, True])
def test_external_stress_inventory_must_match_the_run_inventory(tmp_path, substituted):
    from tennis3d.artifacts import read_json
    from tennis3d.evaluation.delivery import package

    synthetic, real, viewer, _, stress = review_bundle_inputs(tmp_path)
    metrics = read_json(stress / "metrics.json")
    selected = Clip.model_validate(metrics["records"][0]["source"])
    if substituted:
        selected = selected.model_copy(update={"clip_id": "substituted_fixture"})
        metrics["records"][0].update(
            clip_id=selected.clip_id, source=selected.model_dump()
        )
    frozen = inventory([selected])
    original = tmp_path / "original.pretty.json"
    original.write_text(json.dumps(frozen.model_dump(), indent=2) + "\n")
    metrics.update(inventory_sha256=file_sha256(original), inventory=frozen.summary())
    write_json(stress / "metrics.json", metrics, overwrite=True)
    output = tmp_path / "package"
    if substituted:
        with pytest.raises(ValueError, match="differs from run-local inventory"):
            package(
                synthetic,
                real,
                viewer,
                output,
                stress_path=stress,
                stress_inventory_path=original,
            )
        assert not output.exists()
    else:
        assert original.read_bytes() != (stress / "inventory.json").read_bytes()
        package(
            synthetic,
            real,
            viewer,
            output,
            stress_path=stress,
            stress_inventory_path=original,
        )
        assert (output / "resolution-stress/inventory.frozen.json").read_bytes() == (
            original.read_bytes()
        )


def test_stress_preserves_declared_manual_lifting_physical_report(tmp_path):
    from tennis3d.artifacts import read_json
    from tennis3d.evaluation.delivery import package

    synthetic, real, viewer, _, stress = review_bundle_inputs(tmp_path)
    metrics = read_json(stress / "metrics.json")
    row = metrics["records"][0]
    relative = f"clips/{row['clip_id']}/labelled-physical.json"
    row["labelled_lifting"] = {"physical_path": relative}
    write_json(stress / "metrics.json", metrics, overwrite=True)
    report = {
        "metadata": {
            "artifact_id": row["clip_id"],
            "mode": "manual_2D_input",
            "source_identity": {"source_sha256": row["source"]["source_sha256"]},
            "prediction_identity": {"checkpoint_sha256": "a" * 64},
        },
        "kind": "controlled_manual_lifting_fixture",
    }
    write_json(stress / relative, report)
    output = tmp_path / "package"
    package(synthetic, real, viewer, output, stress_path=stress)
    assert (output / "resolution-stress" / relative).read_bytes() == (
        stress / relative
    ).read_bytes()
    report["metadata"]["mode"] = "end_to_end"
    write_json(stress / relative, report, overwrite=True)
    with pytest.raises(ValueError, match="physical clip differs"):
        package(synthetic, real, viewer, tmp_path / "bad_mode", stress_path=stress)


@pytest.mark.parametrize("filename", ["report.md", "config.json", "result.json"])
def test_stress_copy_rejects_outside_symlinks(tmp_path, filename):
    from tennis3d.artifacts import read_json
    from tennis3d.evaluation.delivery import package

    synthetic, real, viewer, _, stress = review_bundle_inputs(tmp_path)
    if filename == "result.json":
        metrics = read_json(stress / "metrics.json")
        metrics["records"][0]["result_path"] = "clips/failed_fixture/result.json"
        write_json(stress / "metrics.json", metrics, overwrite=True)
        write_json(
            stress / "clips/failed_fixture/result.json",
            read_json(real / "clips/controlled/result.json"),
        )
    selected = (
        stress / "clips/failed_fixture/result.json"
        if filename == "result.json"
        else stress / filename
    )
    outside = tmp_path / ("outside-" + filename)
    outside.write_bytes(selected.read_bytes() if selected.exists() else b"private")
    selected.unlink(missing_ok=True)
    selected.symlink_to(outside)
    output = tmp_path / "partial"
    with pytest.raises(ValueError, match="copy source escapes"):
        package(synthetic, real, viewer, output, stress_path=stress)
    assert not (output / "package.json").exists()
    assert not (output / "resolution-stress" / selected.relative_to(stress)).exists()


def test_stress_copied_artifact_is_bound_through_gallery_generation(
    tmp_path, monkeypatch
):
    from tennis3d.artifacts import read_json
    from tennis3d.evaluation import review_bundle
    from tennis3d.evaluation.delivery import package

    synthetic, real, viewer, _, stress = review_bundle_inputs(tmp_path)
    metrics = read_json(stress / "metrics.json")
    metrics["records"][0]["result_path"] = "clips/failed_fixture/result.json"
    write_json(stress / "metrics.json", metrics, overwrite=True)
    write_json(
        stress / "clips/failed_fixture/result.json",
        read_json(real / "clips/controlled/result.json"),
    )
    original = review_bundle.write_playable_gallery

    def mutate_after_gallery(root, result, viewer_path, **kwargs):
        original(root, result, viewer_path, **kwargs)
        if root.name == "resolution-stress":
            path = root / "clips/failed_fixture/result.json"
            data = read_json(path)
            data["frames"][0]["xyz_m"][0] += 1
            write_json(path, data, overwrite=True)

    monkeypatch.setattr(review_bundle, "write_playable_gallery", mutate_after_gallery)
    output = tmp_path / "partial"
    with pytest.raises(ValueError, match="packaged stress report changed"):
        package(synthetic, real, viewer, output, stress_path=stress)
    assert not (output / "package.json").exists()


@pytest.mark.parametrize(
    "gallery_ready,produced_result,expected_status",
    [
        (True, False, "no_results"),
        (True, True, "measured"),
        (False, False, "incomplete_inventory"),
    ],
)
def test_real_run_status_requires_result(
    tmp_path, monkeypatch, gallery_ready, produced_result, expected_status
):
    from types import SimpleNamespace

    from tennis3d.evaluation import delivery

    selected = clip(tmp_path)
    frozen = inventory([selected])
    summary = frozen.summary() | {"required_gallery_ready": gallery_ready}
    monkeypatch.setattr(Inventory, "summary", lambda self: summary)
    path = tmp_path / "inventory.json"
    write_json(path, frozen.model_dump())
    release = SimpleNamespace(
        checkpoint=SimpleNamespace(
            path=selected.source_path, sha256=selected.source_sha256
        )
    )
    provenance = {"release": {"checkpoint": {"sha256": selected.source_sha256}}}
    monkeypatch.setattr(
        delivery, "checked_release", lambda _: (release, None, provenance)
    )

    def evaluate(*args):
        if not produced_result:
            raise RuntimeError("Controlled fixture pipeline failure")
        # A failed-quality result still measures execution/zero coverage.
        return {"status": "failed", "result_path": "clips/controlled/result.json"}, None

    monkeypatch.setattr(delivery, "evaluate_real_clip", evaluate)
    result = delivery.real(
        tmp_path / "unused-release.json",
        path,
        ROOT / "configs/stage4-reconstruction.yaml",
        tmp_path / "run",
    )
    assert result["status"] == expected_status
    assert result["inventory"]["required_gallery_ready"] is gallery_ready
    assert len(result["records"]) == 1
    if not produced_result:
        assert result["records"][0]["status"] == "failed"
        assert (
            "Controlled fixture pipeline failure"
            in result["records"][0]["failure_reasons"][0]
        )
