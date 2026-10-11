"""Keep independent Hough components ambiguous; never invent an interpolated detection."""

import numpy as np

# Numeric observation states shared by detection artifacts.
MISSING, DETECTED, AMBIGUOUS, DUPLICATE = range(4)


def associate(candidates, duplicate_of=-1, merge_radius=8.0):
    if duplicate_of >= 0:
        return np.full(2, np.nan), DUPLICATE
    # Ignore unsupported Hough edge centers for selection only. Raw candidates
    # remain intact in the artifact. Scores are local heatmap intensity, not probability.
    candidates = candidates[candidates[:, 3] >= 0.1]
    if not len(candidates):
        return np.full(2, np.nan), MISSING
    # Hough produces several overlapping circles for one heatmap blob. Collapse
    # only that local cluster; disconnected evidence remains explicitly ambiguous.
    first = candidates[0, :2]
    if np.any(np.linalg.norm(candidates[:, :2] - first, axis=1) > merge_radius):
        return np.full(2, np.nan), AMBIGUOUS
    return first.copy(), DETECTED
