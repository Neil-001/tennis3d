"""Segment-level synthetic discrimination and explicit frozen validation policy."""

import hashlib
import json

import numpy as np
from scipy.stats import spearmanr

from tennis3d.evaluation.real import error_summary


def _signature(report):
    payload = {
        key: report[key]
        for key in ("schema_version", "physics", "config", "definitions")
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def summarize_physical(
    reports, *, error_bin_edges_m=(0.0, 0.25, 0.5, 1.0, 2.0)
) -> dict:
    """Consume reports once, retaining scalar segment errors, never curves or full reports."""
    edges = np.asarray(error_bin_edges_m, dtype=float)
    if (
        edges.ndim != 1
        or not len(edges)
        or edges[0] != 0
        or not np.isfinite(edges).all()
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError(
            "true-error bins must start at zero and strictly increase in metres"
        )
    signature, cohorts = None, {}
    for report in reports:
        current_signature = _signature(report)
        if signature is not None and signature != current_signature:
            raise ValueError(
                "side-by-side controls require identical physics, fit settings and definitions"
            )
        signature = current_signature
        m = report["metadata"]
        key = json.dumps(
            [
                m["domain"],
                m["split"],
                m.get("mode"),
                m["time_basis"],
                m["segmentation_source"],
                m["segmentation_method"],
            ]
        )
        cohort = cohorts.setdefault(
            key,
            {
                "domain": m["domain"],
                "split": m["split"],
                "mode": m.get("mode"),
                "time_basis": m["time_basis"],
                "segmentation_source": m["segmentation_source"],
                "segmentation_method": m["segmentation_method"],
                "artifact_ids": [],
                "artifacts": [],
                "counts": {},
                "reason_counts": {},
                "_ids": set(),
                "_residuals": {"rmse_m": [], "max_discrepancy_m": []},
                "_pairs": [],
                "_reference": {
                    "reprojection_error_px": [0, 0.0, 0.0],
                    "landing_error_xy_m": [0, 0.0, 0.0],
                },
            },
        )
        if m["artifact_id"] in cohort["_ids"]:
            raise ValueError("duplicate artifact_id in a physical cohort")
        cohort["_ids"].add(m["artifact_id"])
        cohort["artifact_ids"].append(m["artifact_id"])
        cohort["artifacts"].append(
            {"metadata": m, "reference": report.get("reference")}
        )
        for field in ("counts", "reason_counts"):
            for name, value in report.get(field, {}).items():
                cohort[field][name] = cohort[field].get(name, 0) + value
        for segment in report["segments"]:
            if segment["status"] != "fitted":
                continue
            rmse, maximum = (
                float(segment["rmse_m"]),
                float(segment["max_discrepancy_m"]),
            )
            cohort["_residuals"]["rmse_m"].append(rmse)
            cohort["_residuals"]["max_discrepancy_m"].append(maximum)
            true = segment.get("true_xyz_error_m", {}).get("rmse_m")
            if segment.get("true_xyz_supported_fraction") == 1.0 and true is not None:
                cohort["_pairs"].append((float(true), rmse, maximum))
        for name, total in cohort["_reference"].items():
            stats = report.get("independent_agreement", {}).get(name)
            if stats is not None and stats["count"]:
                total[0] += stats["count"]
                total[1] += stats["count"] * stats["mean"]
                total[2] += stats["count"] * stats["rmse"] ** 2
    results = []
    for cohort in cohorts.values():
        del cohort["_ids"]
        residuals, pairs = cohort.pop("_residuals"), cohort.pop("_pairs")
        true_error = np.array([s[0] for s in pairs])
        discrimination = {}
        for column, metric in enumerate(("rmse_m", "max_discrepancy_m"), start=1):
            residual = np.array([s[column] for s in pairs])
            rho, reason = None, "too_few_segments"
            if len(pairs) >= 3:
                reason = "constant_values"
                if np.ptp(true_error) > 1e-12 and np.ptp(residual) > 1e-12:
                    rho, reason = float(spearmanr(true_error, residual).statistic), None
            bins = []
            for i, lo in enumerate(edges):
                hi = edges[i + 1] if i + 1 < len(edges) else None
                mask = (true_error >= lo) & (
                    (true_error < hi) if hi is not None else True
                )
                bins.append(
                    {
                        "true_rmse_lower_m": float(lo),
                        "true_rmse_upper_m": float(hi) if hi is not None else None,
                        "residual_m": error_summary(residual[mask]),
                    }
                )
            discrimination[metric] = {
                "paired_segments": len(pairs),
                "spearman_rho": rho,
                "unavailable_reason": reason,
                "by_true_xyz_rmse_bin": bins,
            }
        reference = {
            name: {
                "count": count,
                "mean": total / count if count else None,
                "rmse": float(np.sqrt(square / count)) if count else None,
            }
            for name, (count, total, square) in cohort.pop("_reference").items()
        }
        results.append(
            {
                **cohort,
                "residual_distributions_m": {
                    k: {**error_summary(v), "max": max(v, default=None)}
                    for k, v in residuals.items()
                },
                "discrimination": discrimination,
                "independent_agreement": reference,
            }
        )
    return {
        "schema_version": "1.0",
        "evaluator_signature": signature,
        "cohorts": results,
        "discrimination_definition": "segment-weighted Spearman and fixed error bins: ODE residual versus predicted-to-true Euclidean XYZ RMSE, only segments with full truth support; no p-value independence assumption",
        "reference_aggregation": "frame/endpoint-weighted mean and RMSE from per-report independent summaries; quantiles remain in individual reports",
        "interpretation": "Low ODE residual can coexist with large true error. Report weak/nonpositive discrimination; consistency alone does not establish transfer or real XYZ accuracy.",
    }


def freeze_validation_policy(
    reports, *, metric: str, quantile: float, validation_identity: dict
) -> dict:
    """Explicit opt-in only, after authorization; no default pass threshold or automatic calibration."""
    reports = list(reports)
    if (
        metric not in ("rmse_m", "max_discrepancy_m")
        or not np.isfinite(quantile)
        or not 0 < quantile < 1
        or not validation_identity
    ):
        raise ValueError(
            "declare metric, quantile in (0,1), and reserved-validation identity"
        )
    if not reports or any(
        r["metadata"]["domain"] != "synthetic"
        or r["metadata"]["split"] != "reserved_validation"
        or r["metadata"].get("mode") != "corrupted"
        or r["metadata"]["time_basis"] != "simulation"
        for r in reports
    ):
        raise ValueError(
            "threshold derives only from corrupted reserved synthetic validation"
        )
    summary = summarize_physical(reports)
    values = [
        s[metric] for r in reports for s in r["segments"] if s["status"] == "fitted"
    ]
    if not values:
        raise ValueError("validation has no fitted segments")
    policy = {
        "schema_version": "1.0",
        "metric": metric,
        "threshold_m": float(np.quantile(values, quantile)),
        "quantile": quantile,
        "quantile_method": "numpy linear",
        "segments": len(values),
        "evaluator_signature": summary["evaluator_signature"],
        "validation_identity": validation_identity,
        "validation_report_sha256": [
            hashlib.sha256(
                json.dumps(r, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            for r in reports
        ],
        "rule": "scored segment passes iff chosen Euclidean residual <= frozen threshold; failures/exclusions are not passes",
    }
    policy["policy_sha256"] = hashlib.sha256(
        json.dumps(policy, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return policy


def apply_frozen_policy(report, policy) -> dict:
    payload = {k: v for k, v in policy.items() if k != "policy_sha256"}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if digest != policy.get("policy_sha256") or _signature(report) != policy.get(
        "evaluator_signature"
    ):
        raise ValueError("frozen policy hash or evaluator signature differs")
    metric, threshold = policy["metric"], policy["threshold_m"]
    if (
        metric not in ("rmse_m", "max_discrepancy_m")
        or not np.isfinite(threshold)
        or threshold < 0
    ):
        raise ValueError("invalid frozen metric/threshold")
    records = [
        {
            "segment_id": s["segment_id"],
            "status": s["status"],
            "passes": s[metric] <= threshold if s["status"] == "fitted" else None,
        }
        for s in report["segments"]
    ]
    return {
        "policy_sha256": digest,
        "metric": metric,
        "threshold_m": threshold,
        "passed_segments": sum(r["passes"] is True for r in records),
        "scored_segments": sum(r["passes"] is not None for r in records),
        "records": records,
    }
