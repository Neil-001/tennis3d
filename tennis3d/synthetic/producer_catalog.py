"""Generation-time summaries using the frozen evaluator masks and bins."""

from pathlib import Path

import numpy as np

from tennis3d.artifacts import file_sha256, json_sha256
from tennis3d.datasets import subset
from tennis3d.evaluation import metrics
from tennis3d.synthetic import KINDS

GAPS = ("gap_up_to_0.15s", "gap_0.15_to_0.5s", "gap_over_0.5s")
DISTANCES = ("distance_under_20m", "distance_20_to_40m", "distance_40m_plus")
DEFINITIONS = {
    "version": "producer-coverage-v1",
    "supported": "label_valid & camera_valid",
    "occluded": "supported & ~ball_observed",
    "gap_bins": list(GAPS),
    "distance_bins": list(DISTANCES),
    "duration_s": "full physical t_s[-1] - t_s[0]",
    "shot_kind_bits": list(KINDS),
}


def definition_hash():
    return json_sha256(
        {
            "producer_catalog.py": file_sha256(Path(__file__)),
            "metrics.py": file_sha256(Path(metrics.__file__)),
            "subset.py": file_sha256(Path(subset.__file__)),
        }
    )


def summarize(record):
    views = []
    for counts in subset.supported_counts(record):
        views.append(
            {
                "view_id": counts.view_id,
                "supported": counts.supported,
                "occluded": counts.supported_occluded,
                "gap_counts": [counts.gap_bins[key] for key in GAPS],
                "distance_counts": [counts.distance_bins[key] for key in DISTANCES],
            }
        )
    return {
        "index": record.index().model_dump(),
        "duration_s": float(record.physical["t_s"][-1] - record.physical["t_s"][0]),
        "shot_kind_bits": [
            bool(np.any(record.physical["segment_kind"] == i))
            for i in range(len(KINDS))
        ],
        "views": views,
    }


STRATA_DEFINITIONS = {
    "family": "Exact recorded family_id.",
    "duration": "Physical duration seconds: under_5s (<5), 5_to_15s (>=5,<15), 15s_plus (>=15).",
    "fps": "Exact recorded integer frames per second, encoded as decimal text.",
    "occlusion": "Across all views, supported occluded fraction: no_supported, none (0), under_half (>0,<0.5), half_plus (>=0.5).",
    "distance": "Largest total supported distance-bin count across all views; evaluator bin order breaks ties; no_supported if all zero.",
    **{
        f"shot_{kind}": f"Recorded {kind} shot-kind presence: present or absent."
        for kind in KINDS
    },
}


def stratum_labels(index, duration, bits, counts):
    """Frozen labels computed only from existing physical/index/count fields."""
    supported = sum(v.supported for v in counts)
    occluded = sum(v.supported_occluded for v in counts)
    distances = [sum(v.distance_bins[key] for v in counts) for key in DISTANCES]
    return {
        "family": index.family_id,
        "duration": "under_5s"
        if duration < 5
        else "5_to_15s"
        if duration < 15
        else "15s_plus",
        "fps": str(index.fps),
        "occlusion": (
            "no_supported"
            if not supported
            else "none"
            if not occluded
            else "under_half"
            if 2 * occluded < supported
            else "half_plus"
        ),
        "distance": DISTANCES[max(range(len(DISTANCES)), key=distances.__getitem__)]
        if supported
        else "no_supported",
        **{
            f"shot_{kind}": "present" if bits & (1 << i) else "absent"
            for i, kind in enumerate(KINDS)
        },
    }
