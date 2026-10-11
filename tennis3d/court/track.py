"""Timestamp-aware pykalman filtering of center, log focal and local rotation.

This is a product adapter to pykalman's filter_update, not a new Kalman solver.
Orientation lives in a shot-local tangent chart; absolute Rodrigues components
are never averaged across their pi branch cut.
"""

import numpy as np
from pykalman import KalmanFilter
from scipy.spatial.transform import Rotation

from tennis3d.court.calibrate import intrinsic, quality
from tennis3d.geometry.camera import Camera


class CameraFilter:
    def __init__(self, config):
        self.config = config
        self.mean = self.cov = self.anchor = None
        self.last_time = self.last_measurement = None
        self.rejections = 0
        # Center metres, log focal, local rotation radians; velocity follows.
        self.noise = np.array([0.4, 0.8, 0.4, 0.025, 0.01, 0.01, 0.01]) ** 2
        self.h = np.hstack([np.eye(7), np.zeros((7, 7))])
        self.kf = KalmanFilter(
            n_dim_state=14,
            n_dim_obs=7,
            observation_matrices=self.h,
            observation_covariance=np.diag(self.noise),
        )

    def update(self, time, estimate, uv, size, *, cut=False):
        if self.last_time is not None and time <= self.last_time:
            raise ValueError("Camera filter times must increase strictly")
        if cut or estimate.reason == "ambiguous_focal_depth":
            self.mean = None
            self.last_measurement = None
        accepted = estimate.valid and not cut
        measurement = None
        camera = estimate.camera
        if accepted and camera is not None:
            rotation = Rotation.from_matrix(camera.R)
            if self.mean is None:
                self.anchor = rotation
            assert self.anchor is not None
            local = (rotation * self.anchor.inv()).as_rotvec()
            measurement = np.r_[camera.center, np.log(camera.K[0, 0]), local]
        if self.mean is None:
            self.rejections = 0
            self.last_time = time
            if measurement is None:
                return None, "unavailable"
            self.mean = np.r_[measurement, np.zeros(7)]
            self.cov = np.diag(np.r_[self.noise, self.noise * 4])
            self.last_measurement = time
            return camera, "measured"
        dt = time - self.last_time
        transition = np.eye(14)
        transition[:7, 7:] = np.eye(7) * dt
        process = np.diag(np.r_[self.noise * dt**3 / 3, self.noise * dt])
        process[:7, 7:] = process[7:, :7] = np.diag(self.noise * dt**2 / 2)
        predicted = transition @ self.mean
        assert self.cov is not None
        covariance = transition @ self.cov @ transition.T + process
        if measurement is not None:
            innovation = measurement - self.h @ predicted
            s = self.h @ covariance @ self.h.T + np.diag(self.noise)
            distance = float(innovation @ np.linalg.solve(s, innovation))
            if distance > self.config.innovation_gate:
                measurement = None
                self.rejections += 1
            else:
                self.rejections = 0
        self.mean, self.cov = self.kf.filter_update(
            self.mean,
            self.cov,
            observation=measurement,
            transition_matrix=transition,
            transition_covariance=process,
        )
        self.mean = np.asarray(self.mean)
        self.last_time = time
        if measurement is not None:
            self.last_measurement = time
        if self.rejections >= self.config.reset_after_rejections:
            self.mean = None
            return None, "innovation_reset"
        if (
            self.last_measurement is None
            or time - self.last_measurement > self.config.max_gap_s
        ):
            self.mean = None
            return None, "unsupported_gap"
        rotation = Rotation.from_rotvec(self.mean[4:7]) * self.anchor
        r = rotation.as_matrix()
        filtered = Camera(
            intrinsic(np.exp(self.mean[3]), size), r, -r @ self.mean[:3], size
        )
        # A smoothed camera must still agree with available independent evidence.
        if (
            estimate.valid
            and not quality(filtered, uv, estimate.inliers, self.config)[0]
        ):
            if measurement is not None:
                # This update entered the state but failed the independent pixel
                # check. It cannot support later predictions through missing data.
                self.mean = None
                self.last_measurement = None
                self.rejections = 0
            return None, "filtered_reprojection_error"
        return filtered, "measured" if measurement is not None else "predicted"
