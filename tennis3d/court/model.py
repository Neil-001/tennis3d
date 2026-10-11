"""Metres, court-centre origin, X right, Y far, Z up.

Intersections use ideal dimension lines, measured to the outside court edges.
Paint widths are not modelled. Detector heatmap centres approximate these lines;
that paint/annotation offset belongs in calibration residuals, not a scale change.
"""

import numpy as np

LENGTH_M = 23.77
SINGLES_WIDTH_M = 8.23
DOUBLES_WIDTH_M = 10.97
SERVICE_LINE_DISTANCE_M = 6.40
BALL_RADIUS_M = 0.0335

# TennisCourtDetector court_reference.py key_points, top = far = positive Y.
LANDMARK_NAMES = (
    "far_doubles_left",
    "far_doubles_right",
    "near_doubles_left",
    "near_doubles_right",
    "far_singles_left",
    "near_singles_left",
    "far_singles_right",
    "near_singles_right",
    "far_service_left",
    "far_service_right",
    "near_service_left",
    "near_service_right",
    "far_service_center",
    "near_service_center",
)


def court_landmarks() -> np.ndarray:
    """Return a new float64 (14, 3) array in the canonical detector order."""
    d, s, y, h = (
        DOUBLES_WIDTH_M / 2,
        SINGLES_WIDTH_M / 2,
        LENGTH_M / 2,
        SERVICE_LINE_DISTANCE_M,
    )
    return np.array(
        [
            [-d, y, 0],
            [d, y, 0],
            [-d, -y, 0],
            [d, -y, 0],
            [-s, y, 0],
            [-s, -y, 0],
            [s, y, 0],
            [s, -y, 0],
            [-s, h, 0],
            [s, h, 0],
            [-s, -h, 0],
            [s, -h, 0],
            [0, h, 0],
            [0, -h, 0],
        ],
        dtype=np.float64,
    )
