"""Published Hough parameters, with every candidate retained in network coordinates."""

import cv2
import numpy as np


def circles(heatmap, *, court=False):
    _, binary = cv2.threshold(heatmap, 170 if court else 127, 255, cv2.THRESH_BINARY)
    found = cv2.HoughCircles(
        binary,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20 if court else 1,
        param1=50,
        param2=2,
        minRadius=10 if court else 2,
        maxRadius=25 if court else 7,
    )
    if found is None:
        return np.empty((0, 4), dtype=np.float64)
    result = []
    for x, y, radius in found[0]:
        ix, iy = round(float(x)), round(float(y))
        patch = heatmap[max(0, iy - 2) : iy + 3, max(0, ix - 2) : ix + 3]
        score = float(patch.mean() / 255) if patch.size else 0.0
        result.append([float(x), float(y), float(radius), score])
    return np.array(result)


def ball_map(labels):
    # Preserve the reference's uint8 wraparound rather than claiming a corrected
    # probability interpretation; detector scores are not calibrated probabilities.
    return (labels.astype(np.int64) * 255).astype(np.uint8)
