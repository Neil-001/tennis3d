"""Bounded generation qualification summaries, including scoped prior reuse."""

from collections import Counter, defaultdict
from itertools import pairwise

import numpy as np

from tennis3d.synthetic import EVENT_CODES, KINDS
from tennis3d.synthetic.producer_records import admission_failures

BIN_EDGES = {
    "speed_m_s": [10, 20, 30, 40, 60],
    "spin_rad_s": [50, 100, 200, 400, 800],
    "height_m": [1, 3, 6, 10],
}


class NumericSummary:
    def __init__(self, cohort_size):
        self.cohort_size = cohort_size
        self.cohort = None
        self.active = {}
        self.families = []
        self.counts = Counter()
        self.kinds, self.events, self.fps, self.returns = (
            Counter(),
            Counter(),
            Counter(),
            Counter(),
        )
        self.bins = defaultdict(Counter)
        self.proposals = defaultdict(lambda: defaultdict(Counter))

    def _flush(self):
        for family, counters in sorted(self.active.items()):
            prior_counts, pairs = counters["priors"], counters["pairs"]
            self.families.append(
                {
                    "family_id": family,
                    "rallies": counters["rallies"],
                    "kinds": {
                        kind: {
                            "uses": sum(
                                count
                                for (name, _), count in prior_counts.items()
                                if name == kind
                            ),
                            "unique_priors": sum(
                                name == kind for name, _ in prior_counts
                            ),
                            "max_prior_uses": max(
                                (
                                    count
                                    for (name, _), count in prior_counts.items()
                                    if name == kind
                                ),
                                default=0,
                            ),
                        }
                        for kind in KINDS
                    },
                    "transitions": sum(pairs.values()),
                    "unique_transitions": len(pairs),
                    "extra_transition_reuses": sum(
                        max(0, count - 1) for count in pairs.values()
                    ),
                    "max_transition_uses": max(pairs.values(), default=0),
                }
            )
        self.active.clear()

    def add(self, record, diagnostic):
        family = record.metadata.family_id
        cohort = (
            record.metadata.split,
            int(family.split("-family-")[1].split("-")[0]) // self.cohort_size,
        )
        if self.cohort != cohort:
            self._flush()
            self.cohort = cohort
        scoped = self.active.setdefault(
            family, {"rallies": 0, "priors": Counter(), "pairs": Counter()}
        )
        scoped["rallies"] += 1
        physical = record.physical
        tokens = [
            (KINDS[int(kind)], int(prior))
            for kind, prior in zip(physical["segment_kind"], physical["prior_id"])
        ]
        scoped["priors"].update(tokens)
        scoped["pairs"].update(pairwise(tokens))
        self.kinds.update(kind for kind, _ in tokens)
        self.events.update(str(int(code)) for code in physical["event_code"])
        self.fps[str(record.metadata.fps)] += 1
        self.returns[str(max(0, len(tokens) - 2))] += 1
        self.counts["physical_rallies"] += 1
        self.counts["physical_frames"] += len(physical["t_s"])
        self.counts["typed_numeric_bytes"] += sum(a.nbytes for a in physical.values())
        for name, values in (
            ("speed_m_s", np.linalg.norm(physical["velocity_m_s"], axis=1)),
            ("spin_rad_s", np.linalg.norm(physical["spin_rad_s"], axis=1)),
            ("height_m", physical["xyz_m"][:, 2]),
        ):
            edges = BIN_EDGES[name]
            hist = np.bincount(
                np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1
            )
            self.bins[name].update({str(i): int(count) for i, count in enumerate(hist)})
        if (
            diagnostic["rally_id"] != record.metadata.rally_id
            or diagnostic["rally_seed"] != record.metadata.seed
        ):
            raise ValueError("admission identity differs from numeric record")
        if len(diagnostic["views"]) != len(record.views):
            raise ValueError("admission view membership differs")
        for view, admitted in zip(record.views, diagnostic["views"]):
            if (
                admitted["accepted_seed"] != view.metadata.view_seed
                or admitted["view_id"] != view.metadata.view_id
            ):
                raise ValueError("accepted proposal seed differs from saved view_seed")
            if not admission_failures(view.arrays)["accepted"]:
                raise ValueError("saved view fails frozen clean/corrupted admission")
            self.counts["views"] += 1
            self.counts["view_frames"] += len(view.arrays["t_s"])
            self.counts["typed_numeric_bytes"] += sum(
                a.nbytes for a in view.arrays.values()
            )
            split = record.metadata.split
            self.proposals[split]["attempt_counts"][str(admitted["attempt_count"])] += 1
            for proposal in admitted["proposals"]:
                for phase in (
                    ["proposed"]
                    + (["attempt_zero"] if proposal["attempt"] == 0 else [])
                    + (["admitted"] if proposal["accepted"] else [])
                ):
                    counts = self.proposals[split][phase]
                    summary = proposal["summary"]
                    counts["views"] += 1
                    for key in (
                        "frames",
                        "camera_valid",
                        "supported",
                        "in_frame",
                        "observed",
                    ):
                        counts[key] += summary[key]
                    counts[f"camera_component/{summary['camera_component']}"] += 1
                    counts["all_missing_clean_windows"] += sum(
                        "no_clean_evidence" in w["reasons"]
                        for w in proposal["rejected_windows"]
                    )
                    counts["all_missing_corrupt_windows"] += sum(
                        "no_corrupted_evidence" in w["reasons"]
                        for w in proposal["rejected_windows"]
                    )
                    for key in (
                        "camera_height_m",
                        "camera_distance_m",
                        "focal_px",
                        "missing_runs",
                    ):
                        value = summary[key]
                        counts[f"{key}/count"] += value["count"]
                        counts[f"{key}/sum"] += (value["mean"] or 0) * value["count"]

    def finish(self):
        self._flush()
        return {
            "counts": dict(self.counts),
            "family_reuse": self.families,
            "shot_uses": dict(self.kinds),
            "event_codes": EVENT_CODES,
            "event_counts": dict(self.events),
            "fps": dict(self.fps),
            "returns": dict(self.returns),
            "motion_bins": dict(self.bins),
            "bin_edges": BIN_EDGES,
            "admission_distributions": {
                split: {phase: dict(values) for phase, values in phases.items()}
                for split, phases in self.proposals.items()
            },
        }
