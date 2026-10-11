"""Independent exhaustive match oracle and source-bound fixture delivery controls."""

import copy
import itertools
from fractions import Fraction

import numpy as np
import pytest

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
from tennis3d.evaluation.contact_delivery import evaluate_gallery, read_gallery_clip
from tennis3d.evaluation.contacts import TOLERANCES_S, evaluate_contacts, match_contacts
from tennis3d.evaluation.manual_events import load_manual_snapshot, manual_event_arrays
from tennis3d.evaluation.physical import evaluate_physical
from tennis3d.evaluation.physical_ode import FitConfig
from tennis3d.inference.export import EXPORT_INPUTS
from tennis3d.inference.lift import LIFT_INPUTS
from tennis3d.schemas import Frame, Provenance, Result, Source, Summary, write_result


def test_global_matching_cardinality_then_error_and_ambiguity():
    # Nearest-first takes prediction 0 -> reference 1 and strands prediction 1.
    report = match_contacts([0.09, 0.19], [0.0, 0.1], 0.1)
    assert [(r["prediction_index"], r["reference_index"]) for r in report["pairs"]] == [
        (0, 0),
        (1, 1),
    ]
    minimum = match_contacts([0.0, 0.1], [0.08], 0.1)
    assert minimum["pairs"][0]["prediction_index"] == 1
    tied = match_contacts([0.0, 0.1], [0.2, 0.3], 0.4)
    assert tied["ambiguous"]
    assert [(r["prediction_index"], r["reference_index"]) for r in tied["pairs"]] == [
        (0, 0),
        (1, 1),
    ]
    assert tied == match_contacts([0.0, 0.1], [0.2, 0.3], 0.4)
    assert match_contacts([1], [1.1], 0.1)["pairs"]
    assert not match_contacts([1], [1.10001], 0.1)["pairs"]
    assert match_contacts([], [0.1], 0.1)["unmatched_reference_indices"] == [0]
    assert match_contacts([0.1], [], 0.1)["unmatched_prediction_indices"] == [0]
    for pred, ref, tol in (([np.nan], [], 0.1), ([-1], [], 0.1), ([], [], 0.0)):
        with pytest.raises(ValueError):
            match_contacts(pred, ref, tol)


def test_matcher_against_exhaustive_unordered_oracle():
    rng = np.random.default_rng(78)
    for _ in range(80):
        pred = rng.integers(0, 8, size=3) / 10
        ref = rng.integers(0, 8, size=3) / 10
        options = []
        for assignment in itertools.product(range(4), repeat=3):
            matched = [r for r in assignment if r < 3]
            if len(set(matched)) != len(matched):
                continue
            pairs = [(p, r) for p, r in enumerate(assignment) if r < 3]
            if any(abs(pred[p] - ref[r]) > 0.3 + 1e-12 for p, r in pairs):
                continue
            options.append(
                (len(pairs), sum(abs(pred[p] - ref[r]) for p, r in pairs), assignment)
            )
        cardinality = max(x[0] for x in options)
        cost = min(x[1] for x in options if x[0] == cardinality)
        optima = [
            x for x in options if x[0] == cardinality and abs(x[1] - cost) <= 1e-12
        ]
        expected = min(x[2] for x in optima)
        actual = match_contacts(pred, ref, 0.3)
        assert len(actual["pairs"]) == cardinality
        assert actual["total_absolute_residual_s"] == pytest.approx(cost, abs=1e-12)
        assignment = [3] * 3
        for pair in actual["pairs"]:
            assignment[pair["prediction_index"]] = pair["reference_index"]
        assert tuple(assignment) == expected
        assert actual["ambiguous"] == (len(optima) > 1)


def fixture(tmp_path):
    snapshot, gallery = tmp_path / "snapshot", tmp_path / "gallery"
    clips, labels = [], []
    for c in range(2):
        frames = [
            {
                "frame_index": i,
                "original_frame_index": 50 + i,
                "pts": 2500 + 10 * i,
                "time_base": [1, 250],
                "decoded_rgb_sha256": "d" * 64,
            }
            for i in range(15)
        ]
        path = tmp_path / f"timeline-{c}.json"
        write_json(path, {"source_sha256": "a" * 64, "frames": frames})
        clip = {
            "clip_id": f"clip-{c}",
            "recording_group": "fixture-match",
            "frame_count": 15,
            "original_source_frame_start": 50,
            "source_sha256": "a" * 64,
            "timeline_path": str(path),
            "timeline_sha256": file_sha256(path),
            "timing_note": "locally constructed source-PTS fixture",
            "width": 640,
            "height": 480,
        }
        clips.append(clip)
        events = []
        for i, kind in ((5, "hit"), (10, "bounce")):
            t = Fraction(2500 + 10 * i, 250)
            events.append(
                {
                    **frames[i],
                    "source_frame_index": i,
                    "source": "manual",
                    "type": kind,
                    "source_time": {
                        "numerator": t.numerator,
                        "denominator": t.denominator,
                    },
                    "source_time_seconds": float(t),
                    "elapsed_clip_seconds": float(t - 10),
                    "timing": {"uncertainty_frames": 1},
                }
            )
        labels.append(
            {
                "schema": "tennis3d.manual-events/v1",
                "submitted": True,
                "clip_id": clip["clip_id"],
                "frame_count": 15,
                "revision": 1,
                "events": events,
                "reviewed_event_inventory": {
                    "frame_start": 0,
                    "frame_end_exclusive": 15,
                    "types": ["hit", "bounce"],
                },
            }
        )
    write_json(snapshot / "inventory.private.json", {"clips": clips})
    receipt = {
        "inventory_sha256": file_sha256(snapshot / "inventory.private.json"),
        "clips": 2,
        "event_counts": {"hit": 2, "bounce": 2},
        "files": {},
        "human_timing_caveat": "controlled fixture; no physical contact truth",
        "timing_interpretation": {"physical_contact_time_exact": False},
    }
    for clip, label in zip(clips, labels):
        label["identity"] = {
            "inventory_sha256": receipt["inventory_sha256"],
            "source_sha256": clip["source_sha256"],
            "timeline_sha256": clip["timeline_sha256"],
        }
        path = snapshot / "labels" / f"{clip['clip_id']}.json"
        write_json(path, label)
        receipt["files"][path.name] = {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "revision": 1,
            "events": 2,
        }
    write_json(snapshot / "receipt.json", receipt)
    config = {"lifting": {"fixture": "lifting"}, "quality": {"fixture": "quality"}}
    write_json(gallery / "config.json", config)
    write_json(
        gallery / "metrics.json",
        {
            "config_sha256": json_sha256(config),
            "provenance": {"release": {"checkpoint": {"sha256": "b" * 64}}},
            "records": [
                {
                    "clip_id": c["clip_id"],
                    "source": c,
                    "status": "ok" if i == 0 else "failed",
                    "failure_reasons": [] if i == 0 else ["fixture_absent_stage"],
                }
                for i, c in enumerate(clips)
            ],
        },
    )
    provenance = ArtifactProvenance(
        source_sha256="a" * 64,
        config_sha256="c" * 64,
        upstream_sha256={"checkpoint": "b" * 64},
        code_revision="controlled fixture",
    )
    run = gallery / "clips" / "clip-0"
    pts = np.array([r["pts"] for r in frames])
    t = np.array([float(Fraction(int(p) - 2500, 250)) for p in pts])
    arrays = {
        "t_s": t,
        "frame_index": np.arange(15),
        "pts": pts,
        "time_base": np.tile([1, 250], (15, 1)),
    }
    write_npz(run / "frames.npz", arrays, provenance)
    write_npz(
        run / "ball_track.npz",
        {
            "ball_observed": np.ones(15, dtype=bool),
        },
        provenance,
    )
    write_json(
        run / "detect.json",
        {
            "source_sha256": "a" * 64,
            "artifacts": {
                name: file_sha256(run / name)
                for name in ("frames.npz", "ball_track.npz")
            },
        },
    )
    xyz = np.column_stack([20 * np.abs(t - 0.2), t * 15, 4 + t - 4.905 * t**2])
    write_json(run / "calibrate.json", {"fixture": "calibration"})
    write_npz(run / "cameras.npz", {"fixture": np.zeros(1)}, provenance)
    write_json(run / "lift.config.json", config["lifting"])
    raw_provenance = provenance.model_copy(
        update={
            "config_sha256": json_sha256(config["lifting"]),
            "upstream_sha256": {name: file_sha256(run / name) for name in LIFT_INPUTS}
            | {"checkpoint": "b" * 64},
        }
    )
    write_npz(
        run / "predictions_raw.npz",
        {
            "t_s": t,
            "frame_index": np.arange(15),
            "xyz_m": xyz,
            "covered": np.ones(15, dtype=bool),
        },
        raw_provenance,
    )
    write_json(
        run / "lift.json",
        {
            "checkpoint_sha256": "b" * 64,
            "predictions_sha256": file_sha256(run / "predictions_raw.npz"),
            "weights": "EMA",
            "config_sha256": json_sha256(config["lifting"]),
        },
    )
    result = Result(
        status="partial",
        source=Source(
            name="fixture",
            sha256="a" * 64,
            width=640,
            height=480,
            frame_count=15,
            time_origin_s=10.0,
            time_base=[1, 250],
        ),
        frames=[
            Frame(
                frame_index=i,
                t_s=float(t[i]),
                xyz_m=xyz[i].tolist() if i != 5 else None,
                observation="detected",
                reconstruction="estimated" if i != 5 else "unavailable",
                camera_valid=True,
                reason_codes=[] if i != 5 else ["implausible_prediction"],
            )
            for i in range(15)
        ],
        summary=Summary(valid_frame_fraction=14 / 15, detected_frame_fraction=1.0),
        provenance=Provenance(
            model_label="fixture",
            lifter_checkpoint_sha256="b" * 64,
            config_sha256=json_sha256(config["quality"]),
            calibration_mode="estimated_static",
            calibration_assumptions=[],
        ),
    )
    write_result(run / "result.json", result)
    write_json(run / "export.config.json", config["quality"])
    write_json(run / "metrics.json", {"fixture": "export_metrics"})
    write_json(
        run / "export.json",
        {
            "identity": {
                "source_sha256": "a" * 64,
                "config_sha256": json_sha256(config["quality"]),
                "upstream_sha256": {
                    name: file_sha256(run / name) for name in EXPORT_INPUTS
                },
                "code_identity": None,
            },
            "artifacts": {
                name: file_sha256(run / name)
                for name in ("result.json", "metrics.json", "export.config.json")
            },
        },
    )
    plan = tmp_path / "plan.json"
    write_json(
        plan,
        {
            "frozen_before_model_output_scoring": True,
            "primary_tolerance_seconds": 0.1,
            "sensitivity_tolerance_seconds": list(TOLERANCES_S),
            "manual_segmentation_contract": {
                "primary_exclusion_precision_seconds": 0.1,
                "additive_segmentation_source": "manual_events",
            },
        },
    )
    return snapshot, gallery, plan


def test_fixture_delivery_retains_failed_clip_raw_export_and_separate_ode(tmp_path):
    snapshot, gallery, plan = fixture(tmp_path)
    sources, identity = load_manual_snapshot(
        snapshot, file_sha256(snapshot / "receipt.json")
    )
    assert identity["event_counts"] == {"hit": 2, "bounce": 2}
    output = tmp_path / "report"
    result = evaluate_gallery(
        snapshot=snapshot,
        receipt_sha256=identity["receipt_sha256"],
        plan=plan,
        plan_sha256=file_sha256(plan),
        gallery=gallery,
        output=output,
        checkpoint_sha256="b" * 64,
    )
    assert len(result["records"]) == 2
    # Both clips come from one recording; the limit names that, not the clip count.
    assert any("across 1 recording do" in text for text in result["limitations"])
    raw, exported = result["cohorts"]["raw"], result["cohorts"]["exported"]
    assert raw["coverage"]["source_frames"] == 30
    assert raw["coverage"]["available_frames"] == 15
    assert exported["coverage"]["available_frames"] == 14
    assert raw["scores"][2]["references"] == 4
    assert raw["scores"][2]["true_positive"] == 1
    assert raw["scores"][2]["reference_type_recall"]["hit"]["recall"] == 0.5
    assert exported["scores"][2]["true_positive"] == 0
    assert len(result["physical"]["cohorts"]) == 4
    assert all(
        c["counts"]["source_frames"] == 30 for c in result["physical"]["cohorts"]
    )
    manual = read_json(output / "clips/clip-0/physical-raw-manual_events.json")
    assert manual["config"] == FitConfig().__dict__
    assert [c["precision_s"] for c in manual["segmentation"]["contacts"]] == [0.1, 0.1]
    assert manual["unconditional_xyz_error_m"] is None
    assert manual["reference"]["image"]["kind"] == "unavailable"
    assert all(
        file_sha256(output / p) == sha
        for p, sha in read_json(output / "files.json").items()
    )
    # PTS tampering cannot pass through an elapsed-time coincidence.
    arrays, _ = read_gallery_clip(gallery / "clips/clip-0", sources[0], "b" * 64)
    references = copy.deepcopy(sources[0]["references"])
    references[0]["t_s"] += 0.001
    with pytest.raises(ValueError, match="exact source"):
        evaluate_contacts(arrays["raw"], references)
    with pytest.raises(ValueError, match="checkpoint"):
        read_gallery_clip(gallery / "clips/clip-0", sources[0], "e" * 64)
    with pytest.raises(ValueError, match="hash"):
        load_manual_snapshot(snapshot, "f" * 64)


def test_manual_physical_mode_validates_provenance_and_preserves_defaults(tmp_path):
    snapshot, gallery, _ = fixture(tmp_path)
    sources, _ = load_manual_snapshot(snapshot, file_sha256(snapshot / "receipt.json"))
    data, _ = read_gallery_clip(gallery / "clips/clip-0", sources[0], "b" * 64)
    arrays = {**data["raw"], **manual_event_arrays(sources[0]["references"])}
    meta = {
        "artifact_id": "fixture",
        "domain": "real",
        "split": "test",
        "time_basis": "source_pts",
        "timing_note": "controlled",
        "source_identity": {"source_sha256": "a" * 64},
        "prediction_identity": {"fixture": True},
        "segmentation_source": "manual_events",
        "segmentation_method": "manual-events/v1",
        "manual_event_identity": {
            "source_sha256": "a" * 64,
            "annotation_sha256": "d" * 64,
        },
    }
    evaluate_physical(arrays, metadata=meta)
    for change in (
        {"manual_event_identity": {}},
        {"time_basis": "declared_nominal"},
        {"segmentation_method": "unknown"},
    ):
        with pytest.raises(ValueError, match="manual_events"):
            evaluate_physical(arrays, metadata={**meta, **change})
    for change in (
        {"event_code": np.array([0, 2])},
        {"event_t_s": np.array([0.0, 10.4])},
    ):
        with pytest.raises(ValueError, match="manual_events"):
            evaluate_physical({**arrays, **change}, metadata=meta)
    missing = dict(arrays)
    del missing["event_precision_s"]
    with pytest.raises(ValueError, match="explicit"):
        evaluate_physical(missing, metadata=meta)


def test_episode_midpoint_gaps_and_reference_types_share_one_matching():
    times = np.arange(11) * 0.04
    xyz = np.zeros((11, 3))
    xyz[6:, 0] = 1.0  # Adjacent opposite jumps merge into one episode.
    arrays = {
        "t_s": times,
        "xyz_m": xyz,
        "available": np.ones(11, dtype=bool),
        "input_observed": np.ones(11, dtype=bool),
    }
    references = [
        {"frame_index": 5, "t_s": float(times[5]), "kind": "hit"},
        {"frame_index": 6, "t_s": float(times[6]), "kind": "bounce"},
    ]
    result = evaluate_contacts(arrays, references)
    assert len(result["episodes"]) == 1
    episode = result["episodes"][0]
    assert episode["start_t_s"] == times[5]
    assert episode["end_t_s"] == times[6]
    assert episode["center_t_s"] == pytest.approx(0.22)
    assert episode["kind"] == "unclassified_contact"
    score = result["scores"][2]
    assert score["true_positive"] == 1 and score["false_negative"] == 1
    assert score["matching"]["ambiguous"]
    assert score["reference_type_recall"]["hit"]["recall"] == 1
    assert score["reference_type_recall"]["bounce"]["recall"] == 0
    arrays["available"][5:] = False
    missing = evaluate_contacts(arrays, references)
    assert not missing["episodes"]
    assert missing["scores"][2]["false_negative"] == 2
    assert missing["coverage"]["unavailable_runs"][0]["stop_frame_exclusive"] == 11
    assert missing["coverage"]["unavailable_runs"][0]["end_t_s"] == times[-1]
    arrays["t_s"] = times.copy()
    arrays["t_s"][4:] += 0.2
    assert evaluate_contacts(arrays, [])["coverage"]["source_time_gaps"]


@pytest.mark.parametrize(
    "tamper",
    ["raw_upstream", "lift_config", "export_hash", "export_upstream", "export_result"],
)
def test_gallery_rejects_stale_raw_and_export_provenance(tmp_path, tamper):
    snapshot, gallery, _ = fixture(tmp_path)
    frozen, _ = load_manual_snapshot(snapshot, file_sha256(snapshot / "receipt.json"))
    run = gallery / "clips" / "clip-0"
    if tamper == "raw_upstream":
        arrays, provenance = read_npz(run / "predictions_raw.npz")
        upstream = dict(
            provenance.upstream_sha256,
            **{"frames.npz": "e" * 64, "ball_track.npz": "f" * 64},
        )
        write_npz(
            run / "predictions_raw.npz",
            arrays,
            provenance.model_copy(update={"upstream_sha256": upstream}),
            overwrite=True,
        )
        lift = read_json(run / "lift.json")
        lift["predictions_sha256"] = file_sha256(run / "predictions_raw.npz")
        write_json(run / "lift.json", lift, overwrite=True)
    elif tamper == "lift_config":
        write_json(run / "lift.config.json", {"changed": True}, overwrite=True)
    elif tamper in ("export_hash", "export_upstream"):
        export = read_json(run / "export.json")
        if tamper == "export_hash":
            export["artifacts"]["result.json"] = "e" * 64
        else:
            export["identity"]["upstream_sha256"]["predictions_raw.npz"] = "e" * 64
        write_json(run / "export.json", export, overwrite=True)
    else:
        # Re-enable the filtered contact while keeping a schema-valid raw subset.
        result = read_json(run / "result.json")
        raw, _ = read_npz(run / "predictions_raw.npz")
        result["frames"][5].update(
            xyz_m=raw["xyz_m"][5].tolist(), reconstruction="estimated", reason_codes=[]
        )
        result["summary"]["valid_frame_fraction"] = 1.0
        result["status"] = "ok"
        write_json(run / "result.json", result, overwrite=True)
    with pytest.raises(ValueError, match="lift config|export config"):
        read_gallery_clip(run, frozen[0], "b" * 64)
