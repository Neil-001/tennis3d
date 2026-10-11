"""Read-only frozen-gallery contact evaluation; writes only a new private output."""

import argparse
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from tennis3d.artifacts import file_sha256, json_sha256, read_json, read_npz, write_json
from tennis3d.evaluation.contacts import (
    MANUAL_EXCLUSION_S,
    PRIMARY_TOLERANCE_S,
    evaluate_contacts,
    summarize_contacts,
)
from tennis3d.evaluation.manual_events import (
    load_contact_plan,
    load_manual_snapshot,
    manual_event_arrays,
)
from tennis3d.evaluation.physical import evaluate_physical
from tennis3d.evaluation.physical_controls import summarize_physical
from tennis3d.inference.export import EXPORT_INPUTS
from tennis3d.inference.lift import LIFT_INPUTS
from tennis3d.schemas import read_result


def read_gallery_clip(run, frozen, checkpoint_sha256, *, expected_config=None):
    """Read existing delivery artifacts, retaining absent stages as zero coverage.

    The detector stores elapsed clip seconds. Verify integer source PTS and
    timebase on every row before using the frozen absolute source-time axis.
    Existing malformed or inconsistent files raise; absent files are explicit.
    """
    run = Path(run)
    clip, times = frozen["clip"], frozen["t_s"]
    n = len(times)
    numeric = {
        "t_s": times,
        "xyz_m": np.full((n, 3), np.nan),
        "available": np.zeros(n, dtype=bool),
        "input_observed": np.zeros(n, dtype=bool),
    }
    raw, exported = dict(numeric), dict(numeric)
    files, absent = {}, []

    def present(name):
        path = run / name
        if path.is_file():
            files[name] = file_sha256(path)
            return True
        absent.append(name)
        return False

    def required_hashes(names):
        for name in names:
            if not present(name):
                raise ValueError(
                    f"gallery stage lacks required provenance artifact: {name}"
                )
        return {name: files[name] for name in names}

    frames = None
    frame_identity = None
    if present("frames.npz"):
        frames, identity = read_npz(run / "frames.npz")
        frame_identity = identity
        rows = frozen["frames"]
        elapsed = np.array(
            [
                float(
                    Fraction(r["pts"] * r["time_base"][0], r["time_base"][1])
                    - Fraction(
                        rows[0]["pts"] * rows[0]["time_base"][0],
                        rows[0]["time_base"][1],
                    )
                )
                for r in rows
            ]
        )
        if (
            identity.source_sha256 != clip["source_sha256"]
            or not np.array_equal(frames["frame_index"], np.arange(n))
            or not np.array_equal(frames["pts"], [r["pts"] for r in rows])
            or not np.array_equal(frames["time_base"], [r["time_base"] for r in rows])
            or frames["t_s"].shape != (n,)
            or not np.allclose(frames["t_s"], elapsed, atol=1e-9, rtol=0)
        ):
            raise ValueError("gallery frames differ from frozen source PTS")
    if present("ball_track.npz"):
        track, identity = read_npz(run / "ball_track.npz")
        if (
            frames is None
            or identity.source_sha256 != clip["source_sha256"]
            or identity != frame_identity
            or track["ball_observed"].shape != (n,)
            or track["ball_observed"].dtype != np.bool_
        ):
            raise ValueError("gallery observations differ from frozen source rows")
        raw["input_observed"] = exported["input_observed"] = track["ball_observed"]
    if frames is not None:
        if not present("detect.json"):
            raise ValueError("gallery frame rows lack detection hash provenance")
        detection = read_json(run / "detect.json")
        if detection["source_sha256"] != clip["source_sha256"] or any(
            detection["artifacts"].get(name) != files[name]
            for name in ("frames.npz", "ball_track.npz")
            if name in files
        ):
            raise ValueError("gallery detection source/artifact hashes differ")
    if present("predictions_raw.npz"):
        values, identity = read_npz(run / "predictions_raw.npz")
        if (
            frames is None
            or identity.source_sha256 != clip["source_sha256"]
            or identity.upstream_sha256.get("checkpoint") != checkpoint_sha256
            or not np.array_equal(values["frame_index"], np.arange(n))
            or not np.array_equal(values["t_s"], frames["t_s"])
        ):
            raise ValueError(
                "raw gallery prediction source/checkpoint identity mismatch"
            )
        if not present("lift.json"):
            raise ValueError("raw gallery predictions lack lift provenance")
        lifting = read_json(run / "lift.json")
        upstream = required_hashes(LIFT_INPUTS) | {"checkpoint": checkpoint_sha256}
        required_hashes(("lift.config.json",))
        lift_config = read_json(run / "lift.config.json")
        if (
            lifting["checkpoint_sha256"] != checkpoint_sha256
            or lifting["predictions_sha256"] != files["predictions_raw.npz"]
            or lifting["weights"] != "EMA"
            or lifting["config_sha256"] != json_sha256(lift_config)
            or identity.config_sha256 != lifting["config_sha256"]
            or identity.upstream_sha256 != upstream
            or (
                expected_config is not None
                and lift_config != expected_config["lifting"]
            )
        ):
            raise ValueError(
                "gallery lift config/upstream/hash/checkpoint/EMA identity mismatch"
            )
        raw["xyz_m"], raw["available"] = values["xyz_m"], values["covered"]
    if present("result.json"):
        required_hashes(("export.json", "export.config.json", "metrics.json"))
        export = read_json(run / "export.json")
        export_config = read_json(run / "export.config.json")
        export_identity = export["identity"]
        if (
            export_identity["source_sha256"] != clip["source_sha256"]
            or export_identity["config_sha256"] != json_sha256(export_config)
            or export_identity["upstream_sha256"] != required_hashes(EXPORT_INPUTS)
            or export["artifacts"]
            != {
                name: files[name]
                for name in ("result.json", "metrics.json", "export.config.json")
            }
            or (
                expected_config is not None
                and export_config != expected_config["quality"]
            )
        ):
            raise ValueError("gallery export config/upstream/artifact hashes differ")
        result = read_result(run / "result.json")
        if (
            frames is None
            or result.source.sha256 != clip["source_sha256"]
            or result.source.frame_count != n
            or (result.source.width, result.source.height)
            != (clip["width"], clip["height"])
            or result.provenance.lifter_checkpoint_sha256 != checkpoint_sha256
            or result.provenance.config_sha256 != export_identity["config_sha256"]
            or result.provenance.code_revision != export_identity["code_identity"]
            or not np.array_equal([r.t_s for r in result.frames], frames["t_s"])
        ):
            raise ValueError("exported gallery source/checkpoint identity mismatch")
        exported["xyz_m"] = np.array(
            [r.xyz_m if r.xyz_m is not None else [np.nan] * 3 for r in result.frames],
            dtype=float,
        )
        exported["available"] = np.array(
            [r.xyz_m is not None for r in result.frames], dtype=bool
        )
        if np.any(exported["available"] & ~raw["available"]) or not np.array_equal(
            exported["xyz_m"][exported["available"]],
            raw["xyz_m"][exported["available"]],
        ):
            raise ValueError("exported gallery differs from its raw predictions")
    return {"raw": raw, "exported": exported}, {
        "files_sha256": files,
        "absent_artifacts": absent,
        "time_axis": "verified elapsed detector time joined to immutable integer source PTS; contact scoring and ODE use absolute source seconds",
    }


def evaluate_gallery(
    *, snapshot, receipt_sha256, plan, plan_sha256, gallery, output, checkpoint_sha256
):
    started = time.perf_counter()
    output, gallery = Path(output), Path(gallery)
    if output.exists() and any(output.iterdir()):
        raise ValueError("contact output must be a new or empty directory")
    frozen_plan = load_contact_plan(plan, plan_sha256)
    sources, snapshot_identity = load_manual_snapshot(snapshot, receipt_sha256)
    # The complete frozen snapshot is verified before the first output read.
    gallery_metrics = read_json(gallery / "metrics.json")
    gallery_hash = file_sha256(gallery / "metrics.json")
    gallery_config = read_json(gallery / "config.json")
    if json_sha256(gallery_config) != gallery_metrics["config_sha256"]:
        raise ValueError("gallery reconstruction config hash mismatch")
    records = gallery_metrics["records"]
    ids = [s["clip"]["clip_id"] for s in sources]
    if len(records) != len(ids) or {r["clip_id"] for r in records} != set(ids):
        raise ValueError("gallery must report every frozen clip exactly once")
    if (
        gallery_metrics["provenance"]["release"]["checkpoint"]["sha256"]
        != checkpoint_sha256
    ):
        raise ValueError("gallery uses a different released checkpoint")
    by_id = {r["clip_id"]: r for r in records}
    for frozen in sources:
        clip = frozen["clip"]
        record_source = by_id[clip["clip_id"]]["source"]
        if any(
            record_source[key] != clip[key]
            for key in (
                "source_sha256",
                "timeline_sha256",
                "frame_count",
                "width",
                "height",
            )
        ):
            raise ValueError("gallery record differs from frozen source domain")
    contacts = {"raw": [], "exported": []}
    physical_paths, rows = [], []
    output.mkdir(parents=True, exist_ok=True)
    for frozen in sources:
        clip = frozen["clip"]
        clip_id = clip["clip_id"]
        numeric, artifact_identity = read_gallery_clip(
            gallery / "clips" / clip_id,
            frozen,
            checkpoint_sha256,
            expected_config=gallery_config,
        )
        row = {
            "clip_id": clip_id,
            "gallery_status": by_id[clip_id]["status"],
            "gallery_failure_reasons": by_id[clip_id].get("failure_reasons", []),
            "source_identity": {
                "source_sha256": clip["source_sha256"],
                "timeline_sha256": clip["timeline_sha256"],
                "annotation_sha256": frozen["annotation_sha256"],
            },
            "artifact_identity": artifact_identity,
            "cohorts": {},
        }
        for cohort, arrays in numeric.items():
            report = evaluate_contacts(arrays, frozen["references"])
            contacts[cohort].append(report)
            path = Path("clips") / clip_id / f"contacts-{cohort}.json"
            write_json(output / path, report)
            row["cohorts"][cohort] = {
                "contact_path": str(path),
                "coverage": report["coverage"],
                "primary": next(
                    s
                    for s in report["scores"]
                    if s["tolerance_s"] == PRIMARY_TOLERANCE_S
                ),
                "physical_paths": {},
            }
            metadata = {
                "artifact_id": clip_id,
                "domain": "real",
                "split": "test",
                "mode": f"end_to_end_{cohort}",
                "time_basis": "source_pts",
                "timing_note": clip["timing_note"],
                "source_identity": row["source_identity"],
                "prediction_identity": {
                    "checkpoint_sha256": checkpoint_sha256,
                    "weights": "EMA",
                    "filtering": cohort,
                    **artifact_identity,
                },
                "segmentation_source": "automatic",
                "segmentation_method": "velocity_jump_v1",
            }
            for segmentation in ("automatic", "manual_events"):
                data, meta = dict(arrays), dict(metadata)
                if segmentation == "manual_events":
                    data.update(manual_event_arrays(frozen["references"]))
                    meta.update(
                        segmentation_source="manual_events",
                        segmentation_method="manual-events/v1",
                        manual_event_identity={
                            "source_sha256": clip["source_sha256"],
                            "annotation_sha256": frozen["annotation_sha256"],
                            "receipt_sha256": receipt_sha256,
                            "method": "source-bound manual contact marks",
                            "engineering_exclusion_precision_s": MANUAL_EXCLUSION_S,
                            "precision_note": "predeclared engineering exclusion, not calibrated annotator uncertainty",
                        },
                    )
                fit = evaluate_physical(data, metadata=meta)
                fit_path = (
                    Path("clips") / clip_id / f"physical-{cohort}-{segmentation}.json"
                )
                write_json(output / fit_path, fit)
                physical_paths.append(fit_path)
                row["cohorts"][cohort]["physical_paths"][segmentation] = str(fit_path)
        rows.append(row)
    if file_sha256(gallery / "metrics.json") != gallery_hash:
        raise ValueError("gallery metrics changed during contact evaluation")
    for row in rows:
        if any(
            file_sha256(gallery / "clips" / row["clip_id"] / name) != digest
            for name, digest in row["artifact_identity"]["files_sha256"].items()
        ):
            raise ValueError("gallery artifact changed during contact evaluation")
    groups = len({s["clip"]["recording_group"] for s in sources})
    recordings = f"{groups} recording{'' if groups == 1 else 's'}"
    result = {
        "schema_version": "1.0",
        "scientific_status": "manual_label_relative_untyped_contact_evaluation_and_ODE_consistency",
        "checkpoint_sha256": checkpoint_sha256,
        "snapshot": snapshot_identity,
        "plan_sha256": plan_sha256,
        "plan": frozen_plan,
        "gallery_metrics_sha256": gallery_hash,
        "gallery_provenance": gallery_metrics["provenance"],
        "primary_tolerance_s": PRIMARY_TOLERANCE_S,
        "cohorts": {
            name: summarize_contacts(reports) for name, reports in contacts.items()
        },
        "records": rows,
        "physical": summarize_physical(
            read_json(output / path) for path in physical_paths
        ),
        "elapsed_s": time.perf_counter() - started,
        "limitations": [
            snapshot_identity["human_timing_caveat"],
            "Saved one-frame uncertainty is not calibrated physical timing; blur uncertainty and broadcaster exposure are unknown.",
            "velocity_jump_v1 is an unclassified prediction-derived heuristic; only reference-type recall is reported, never typed precision.",
            "All frozen references remain in recall denominators, including failed clips and unavailable source rows; precision counts every predicted episode within each fully reviewed clip.",
            "Episode center is the arithmetic midpoint of source-PTS onset/end, possibly between sampled frames; episode extent and detector precision remain separate.",
            "Manual ODE exclusions use 100 ms engineering precision plus the unchanged 10 ms guard. Automatic ODE remains a separate cohort with unchanged FitConfig and FlightParameters.",
            "Event-only references provide no UV, calibration, landing XY or independent real XYZ truth. ODE residual is consistency, not XYZ accuracy.",
            f"High-resolution convenience clips across {recordings} do not establish population performance; occlusion category and detector-training independence remain unverified.",
        ],
    }
    write_json(output / "metrics.json", result)
    lines = [
        "# Frozen manual contact evaluation",
        "",
        f"All {len(rows)} clips and {sum(snapshot_identity['event_counts'].values())} manual contact references are retained.",
        "",
        "| cohort | TP | FP | FN | precision | recall | F1 | hit-reference recall | bounce-reference recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cohort, report in result["cohorts"].items():
        s = next(s for s in report["scores"] if s["tolerance_s"] == PRIMARY_TOLERANCE_S)
        lines.append(
            f"| {cohort} | {s['true_positive']} | {s['false_positive']} | {s['false_negative']} | {s['precision']} | {s['recall']} | {s['f1']} | {s['reference_type_recall']['hit']['recall']} | {s['reference_type_recall']['bounce']['recall']} |"
        )
    lines += [
        "",
        "Primary matching tolerance is 100 ms. All declared sensitivities, matches, timing residuals, ambiguity witnesses, gaps, source hashes and separate ODE cohorts are in [metrics.json](metrics.json) and the per-clip reports.",
        "",
        *[f"- {limit}" for limit in result["limitations"]],
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    write_json(
        output / "files.json",
        {
            str(p.relative_to(output)): file_sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "plan", "gallery", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("receipt-sha256", "plan-sha256", "checkpoint-sha256"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    result = evaluate_gallery(**vars(args))
    print(f"Evaluated {len(result['records'])} frozen clips; reports in {args.output}")


if __name__ == "__main__":
    main()
