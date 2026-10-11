import numpy as np
import pytest

from tennis3d.court.model import LANDMARK_NAMES, court_landmarks
from tennis3d.geometry.camera import Camera, ImageSize
from tennis3d.geometry.image import crop_resize, normalize_pixels
from tennis3d.geometry.projection import project, world_rays


def broadcast_camera():
    centre = np.array([2.0, -24.0, 12.0])
    forward = -centre / np.linalg.norm(centre)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    return Camera(
        np.array([[1300, 11, 960], [0, 1250, 540], [0, 0, 1]]),
        rotation,
        -rotation @ centre,
        ImageSize(1920, 1080),
    )


def test_landmark_order_and_metric_frame():
    xyz = court_landmarks()
    assert xyz.shape == (14, 3)
    assert len(LANDMARK_NAMES) == 14
    np.testing.assert_allclose(
        xyz[[0, 3, 4, 5, 8, 13]],
        [
            [-5.485, 11.885, 0],
            [5.485, -11.885, 0],
            [-4.115, 11.885, 0],
            [-4.115, -11.885, 0],
            [-4.115, 6.4, 0],
            [0, -6.4, 0],
        ],
    )
    xyz[:] = 0
    assert court_landmarks()[0, 1] > 0


def test_projection_pose_and_rays_round_trip():
    camera = broadcast_camera()
    xyz = np.concatenate([court_landmarks(), [[0.4, -8.2, 1.3], [2, 0, 4]]])
    uv, valid = project(camera, xyz)
    assert valid.all()
    np.testing.assert_allclose(
        camera.camera_to_world(camera.world_to_camera(xyz)), xyz, atol=1e-13
    )
    centre, directions = world_rays(camera, uv)
    distance = np.linalg.norm(xyz - centre, axis=-1)
    np.testing.assert_allclose(centre + directions * distance[:, None], xyz, atol=1e-12)
    np.testing.assert_allclose(project(camera, [0, 0, 0])[0], [960, 540], atol=1e-12)
    behind = camera.camera_to_world([[0, 0, -1], [0, 0, 0]])
    pixels, mask = project(camera, behind)
    assert not mask.any()
    assert np.isnan(pixels).all()


def test_independent_opencv_coordinate_example():
    camera = Camera(
        np.array([[100, 10, 320], [0, 200, 240], [0, 0, 1]]),
        np.eye(3),
        np.array([0, 0, 10]),
        ImageSize(640, 480),
    )
    np.testing.assert_allclose(project(camera, [[2, 3, 0]])[0], [[343, 300]])
    np.testing.assert_allclose(camera.center, [0, 0, -10])


def test_crop_resize_updates_full_intrinsics_and_pixels():
    camera = broadcast_camera()
    transform = crop_resize(
        camera.image_size, ImageSize(640, 360), left=120, top=60, width=1600, height=900
    )
    uv, _ = project(camera, court_landmarks())
    changed, _ = project(transform.camera(camera), court_landmarks())
    np.testing.assert_allclose(changed, transform.apply(uv))
    np.testing.assert_allclose(transform.inverse().apply(changed), uv)
    np.testing.assert_allclose(
        normalize_pixels([[1920, 1080]], camera.image_size), [[1, 1]]
    )
    with pytest.raises(ValueError):
        crop_resize(camera.image_size, ImageSize(10, 10), left=-1)
    with pytest.raises(ValueError):
        transform.camera(Camera(camera.K, camera.R, camera.t, ImageSize(10, 10)))


def test_invalid_camera_conventions():
    with pytest.raises(ValueError, match="rotation"):
        Camera(np.eye(3), np.diag([1, 1, -1]), np.zeros(3), ImageSize(10, 10))
    with pytest.raises(ValueError, match="positive"):
        ImageSize(True, 10)
    with pytest.raises(ValueError, match="distortion"):
        Camera(np.eye(3), np.eye(3), np.zeros(3), ImageSize(10, 10), "radial")  # ty: ignore[invalid-argument-type]
