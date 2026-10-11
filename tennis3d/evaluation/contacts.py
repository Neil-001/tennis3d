"""Frozen untyped contact scoring. References never influence contact detection."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from tennis3d.evaluation.physical import _automatic_contacts, _mask, _xyz
from tennis3d.evaluation.physical_ode import DEFAULT_CONFIG
from tennis3d.evaluation.real import error_summary
from tennis3d.evaluation.runs import mask_runs, timeline

TOLERANCES_S = (1 / 30, 2 / 30, 0.1, 0.2)
PRIMARY_TOLERANCE_S = 0.1
MANUAL_EXCLUSION_S = 0.1
OPTIMUM_EQUIVALENCE_S = 1e-12


def match_contacts(prediction_t_s, reference_t_s, tolerance_s):
    """Maximum cardinality, then minimum total |dt|, then lexicographic indices.

    A private dummy column for every prediction permits all unmatched outcomes.
    The unmatched cost exceeds every possible sum of valid edge costs. Re-solving
    with each chosen edge forbidden detects alternative optima, including crossing
    assignments. Costs within 1e-12 seconds count as computational ties only.
    """
    pred = np.asarray(prediction_t_s, dtype=float)
    ref = np.asarray(reference_t_s, dtype=float)
    if any(
        v.ndim != 1 or not np.isfinite(v).all() or np.any(v < 0) for v in (pred, ref)
    ):
        raise ValueError("contact times must be finite nonnegative vectors")
    if not np.isfinite(tolerance_s) or tolerance_s <= 0:
        raise ValueError("contact tolerance must be finite and positive")
    errors = np.abs(pred[:, None] - ref)
    # Only absorb floating arithmetic at a declared inclusive boundary.
    allowed = errors <= tolerance_s + OPTIMUM_EQUIVALENCE_S
    penalty = (len(pred) + 1) * (tolerance_s + 1)

    def solve(pred_ids, ref_ids, forbidden=None):
        if not pred_ids:
            return []
        cost = np.full((len(pred_ids), len(ref_ids) + len(pred_ids)), penalty)
        for i, p in enumerate(pred_ids):
            for j, r in enumerate(ref_ids):
                cost[i, j] = (
                    errors[p, r] if allowed[p, r] and (p, r) != forbidden else np.inf
                )
        rows, columns = linear_sum_assignment(cost)
        return [
            (pred_ids[i], ref_ids[j]) for i, j in zip(rows, columns) if j < len(ref_ids)
        ]

    def objective(pairs):
        return len(pairs), sum(float(errors[p, r]) for p, r in pairs)

    optimum = solve(list(range(len(pred))), list(range(len(ref))))
    count, total = objective(optimum)

    def equivalent(pairs):
        n, cost = objective(pairs)
        return n == count and abs(cost - total) <= OPTIMUM_EQUIVALENCE_S

    # Fix the lowest reference index compatible with the optimum for each
    # original prediction index. Unmatched sorts after every reference index.
    chosen, remaining = [], list(range(len(ref)))
    for p in range(len(pred)):
        for r in [r for r in remaining if allowed[p, r]] + [None]:
            trial = chosen + ([(p, r)] if r is not None else [])
            rest = [j for j in remaining if j != r]
            trial += solve(list(range(p + 1, len(pred))), rest)
            if equivalent(trial):
                if r is not None:
                    chosen.append((p, r))
                    remaining.remove(r)
                break
        else:
            raise RuntimeError("contact assignment lost its optimum")
    alternatives = []
    for pair in chosen:
        alternative = solve(list(range(len(pred))), list(range(len(ref))), pair)
        if equivalent(alternative):
            alternatives.append(
                {
                    "chosen_edge": list(pair),
                    "alternative_pairs": [list(x) for x in alternative],
                }
            )
    matched_p, matched_r = {p for p, _ in chosen}, {r for _, r in chosen}
    return {
        "pairs": [
            {
                "prediction_index": p,
                "reference_index": r,
                "residual_s": float(pred[p] - ref[r]),
            }
            for p, r in chosen
        ],
        "unmatched_prediction_indices": [
            p for p in range(len(pred)) if p not in matched_p
        ],
        "unmatched_reference_indices": [
            r for r in range(len(ref)) if r not in matched_r
        ],
        "total_absolute_residual_s": sum(float(errors[p, r]) for p, r in chosen),
        "ambiguous": bool(alternatives),
        "alternative_optimum_witnesses": alternatives,
        "tie_rule": "lexicographic original prediction then reference indices; unmatched last",
        "computational_tie_epsilon_s": OPTIMUM_EQUIVALENCE_S,
        "inclusive_boundary_epsilon_s": OPTIMUM_EQUIVALENCE_S,
    }


def contact_metrics(tp, predictions, references):
    precision = tp / predictions if predictions else None
    recall = tp / references if references else None
    return {
        "true_positive": tp,
        "false_positive": predictions - tp,
        "false_negative": references - tp,
        "predictions": predictions,
        "references": references,
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / (predictions + references) if predictions + references else None,
    }


def timing_summary(residuals):
    signed = error_summary(residuals)
    return {"signed_s": signed, "absolute_s": error_summary(np.abs(residuals))}


def evaluate_contacts(arrays, references, *, tolerances_s=TOLERANCES_S):
    """Retain all typed references in recall, even on unavailable source rows.

    References have exact source frame_index, t_s and kind, plus any original
    annotation fields. The heuristic's internal hit/bounce guesses are discarded.
    Episode centers are the arithmetic midpoint of onset/end source-PTS times.
    """
    times = timeline(arrays["t_s"])
    n = len(times)
    available = _mask(arrays, "available", n)
    xyz = _xyz(arrays, "xyz_m", n, available)
    _mask(arrays, "input_observed", n)
    tolerances = list(tolerances_s)
    if not tolerances or len(set(tolerances)) != len(tolerances):
        raise ValueError("contact tolerances must be nonempty and unique")
    for row in references:
        i = row["frame_index"]
        if (
            type(i) is not int
            or not 0 <= i < n
            or row["t_s"] != float(times[i])
            or row["kind"] not in ("hit", "bounce")
        ):
            raise ValueError(
                "contact reference must match exact source row/time and type"
            )
    events = _automatic_contacts(times, xyz, available, DEFAULT_CONFIG)
    episodes = [
        {
            "start_t_s": e["start_t_s"],
            "end_t_s": e["end_t_s"],
            "center_t_s": (e["start_t_s"] + e["end_t_s"]) / 2,
            "detector_precision_s": e["precision_s"],
            "kind": "unclassified_contact",
        }
        for e in events
    ]
    center_support = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        center_support[i] = (
            available[i - 1 : i + 2].all()
            and np.max(np.diff(times[i - 1 : i + 2])) <= DEFAULT_CONFIG.max_time_gap_s
        )
    scored = []
    for tolerance in tolerances:
        match = match_contacts(
            [e["center_t_s"] for e in episodes],
            [r["t_s"] for r in references],
            tolerance,
        )
        matched = {p["reference_index"] for p in match["pairs"]}
        by_type = {}
        for kind in ("hit", "bounce"):
            indices = [i for i, r in enumerate(references) if r["kind"] == kind]
            by_type[kind] = {
                "matched_references": len(matched.intersection(indices)),
                "references": len(indices),
                "recall": len(matched.intersection(indices)) / len(indices)
                if indices
                else None,
            }
        scored.append(
            {
                "tolerance_s": tolerance,
                **contact_metrics(len(match["pairs"]), len(episodes), len(references)),
                "reference_type_recall": by_type,
                "timing": timing_summary([p["residual_s"] for p in match["pairs"]]),
                "matching": match,
                "references_with_heuristic_support_in_window": sum(
                    bool(
                        np.any(center_support & (np.abs(times - r["t_s"]) <= tolerance))
                    )
                    for r in references
                ),
            }
        )
    gaps = [
        {
            "start_frame": start,
            "stop_frame_exclusive": stop,
            "start_t_s": float(times[start]),
            "end_t_s": float(times[stop]) if stop < n else float(times[-1]),
            "duration_s": float(
                (times[stop] if stop < n else times[-1]) - times[start]
            ),
        }
        for start, stop, value in mask_runs(available)
        if not value
    ]
    return {
        "method": "velocity_jump_v1",
        "prediction_type": "unclassified_contact",
        "center_rule": "arithmetic midpoint of episode onset/end source PTS; never label-selected or nominal FPS",
        "episodes": episodes,
        "references": references,
        "coverage": {
            "source_frames": n,
            "available_frames": int(available.sum()),
            "heuristic_supported_center_frames": int(center_support.sum()),
            "reference_frames_available": sum(
                bool(available[r["frame_index"]]) for r in references
            ),
            "references": len(references),
            "unavailable_runs": gaps,
            "source_time_gaps": [
                {
                    "before_frame": int(i),
                    "after_frame": int(i + 1),
                    "gap_s": float(times[i + 1] - times[i]),
                }
                for i in np.flatnonzero(np.diff(times) > DEFAULT_CONFIG.max_time_gap_s)
            ],
            "gap_duration_convention": "sample hold to next row; final endpoint is last source PTS without invented exposure",
        },
        "scores": scored,
    }


def summarize_contacts(reports):
    """Micro counts on all clips; timing uses every matched pair, including ties."""
    reports = list(reports)
    if not reports:
        raise ValueError("contact summary needs all preselected clip reports")
    tolerances = [s["tolerance_s"] for s in reports[0]["scores"]]
    if any([s["tolerance_s"] for s in r["scores"]] != tolerances for r in reports):
        raise ValueError("contact reports require identical tolerances")
    scores = []
    for i, tolerance in enumerate(tolerances):
        rows = [r["scores"][i] for r in reports]
        result = {
            "tolerance_s": tolerance,
            **contact_metrics(
                sum(r["true_positive"] for r in rows),
                sum(r["predictions"] for r in rows),
                sum(r["references"] for r in rows),
            ),
        }
        result["reference_type_recall"] = {}
        for kind in ("hit", "bounce"):
            count = sum(r["reference_type_recall"][kind]["references"] for r in rows)
            matched = sum(
                r["reference_type_recall"][kind]["matched_references"] for r in rows
            )
            result["reference_type_recall"][kind] = {
                "references": count,
                "matched_references": matched,
                "recall": matched / count if count else None,
            }
        result["timing"] = timing_summary(
            [p["residual_s"] for r in rows for p in r["matching"]["pairs"]]
        )
        result["ambiguous_clips"] = sum(r["matching"]["ambiguous"] for r in rows)
        result["references_with_heuristic_support_in_window"] = sum(
            r["references_with_heuristic_support_in_window"] for r in rows
        )
        scores.append(result)
    return {
        "clips": len(reports),
        "coverage": {
            k: sum(r["coverage"][k] for r in reports)
            for k in (
                "source_frames",
                "available_frames",
                "heuristic_supported_center_frames",
                "reference_frames_available",
                "references",
            )
        },
        "scores": scores,
    }
