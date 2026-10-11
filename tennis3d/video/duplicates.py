"""Conservative repeated-image evidence, including motion-region checks.

Whole-frame SSIM alone never declares a duplicate. Exact decoded pixel equality
is sufficient. Re-encoded holds require unchanged previously moving pixels too.
"""

import hashlib

import cv2
import numpy as np
from skimage.metrics import structural_similarity


class DuplicateDetector:
    def __init__(self, ssim=0.9995, motion_error=0.7, cut_difference=55.0):
        self.threshold = ssim
        self.motion_error = motion_error
        self.cut_difference = cut_difference
        self.previous = self.before = self.digest = None
        self.previous_source = -1

    def update(self, bgr, index):
        digest = hashlib.sha256(bgr.tobytes()).digest()
        gray = cv2.cvtColor(cv2.resize(bgr, (320, 180)), cv2.COLOR_BGR2GRAY)
        self.exact_match = digest == self.digest
        duplicate = -1
        similarity, moving_error, cut = 1.0, float("nan"), False
        if self.previous is not None:
            delta = np.abs(gray.astype(float) - self.previous)
            similarity = float(
                structural_similarity(gray, self.previous, data_range=255)
            )
            cut = bool(delta.mean() > self.cut_difference and similarity < 0.5)
            if digest == self.digest:
                duplicate = self.previous_source
            elif self.before is not None:
                motion = np.abs(self.previous.astype(float) - self.before) > 8
                if motion.sum() >= 30:
                    moving_error = float(np.mean(delta[motion]))
                    if (
                        similarity >= self.threshold
                        and moving_error <= self.motion_error
                    ):
                        duplicate = self.previous_source
        self.before, self.previous, self.digest = self.previous, gray, digest
        # Near-match proposals may fail the later cadence check. Only exact
        # equality can safely collapse a reference before that confirmation.
        self.previous_source = (
            duplicate if self.exact_match and duplicate >= 0 else index
        )
        return duplicate, similarity, moving_error, cut


def confirm_repetition(proposed, exact):
    """Confirm re-encoded holds only in a recurring cadence of at least three.

    Exact decoded matches need no cadence. This intentionally misses isolated
    near-duplicates rather than declaring every static-looking frame a hold.
    """
    proposed = np.asarray(proposed, dtype=np.int64).copy()
    indices = np.flatnonzero((proposed >= 0) & ~np.asarray(exact, dtype=bool))
    confirmed = set()
    for period in range(2, 13):
        for phase in range(period):
            group = indices[indices % period == phase]
            breaks = np.flatnonzero(np.diff(group) != period) + 1
            for run in np.split(group, breaks):
                if len(run) >= 3:
                    confirmed.update(run.tolist())
    for index in indices:
        if index not in confirmed:
            proposed[index] = -1
    return proposed
