import numpy as np
import pytest

from aero_webcam_teleop.tracking import HandMocap, build_points, depth_at, physical_side

IMAGE = [(0.1 * i / 20, 0.5, 0.0) for i in range(21)]
WORLD = [(float(i), 0.0, 0.0) for i in range(21)]


def test_mediapipe_handedness_is_mirrored_for_raw_frames():
    assert physical_side("Right") == "left"
    assert physical_side("Left") == "right"
    assert physical_side("Right", mirrored=True) == "right"


def test_world_landmarks_preferred_over_image():
    pts, source = build_points(IMAGE, WORLD, (480, 640, 3))
    assert source == "world"
    np.testing.assert_allclose(pts, np.asarray(WORLD))


def test_image_fallback_corrects_aspect_ratio():
    pts, source = build_points([(0.5, 0.5, 0.1)] * 21, None, (480, 640, 3))
    assert source == "image"
    assert pts[0] == pytest.approx([0.5, 0.5 * 480 / 640, 0.1])


def test_depth_points_used_only_when_every_landmark_has_depth():
    depth = np.full((480, 640), 0.3, dtype=np.float32)
    deproject = lambda u, v, d: np.array([u * d, v * d, d])  # noqa: E731
    pts, source = build_points(IMAGE, WORLD, depth.shape, depth, deproject)
    assert source == "depth"
    assert pts[:, 2] == pytest.approx(0.3)

    depth[:, :] = 0.0  # sensor dropout
    _, source = build_points(IMAGE, WORLD, depth.shape, depth, deproject)
    assert source == "world"


def test_depth_at_ignores_zero_pixels():
    depth = np.zeros((10, 10), dtype=np.float32)
    depth[5, 5] = 0.4
    assert depth_at(depth, 5, 5) == pytest.approx(0.4)
    assert depth_at(np.zeros((10, 10)), 5, 5) == 0.0


def test_default_keypoints_are_independent():
    m = HandMocap()
    m.keypoints[0].x = 1.0
    assert m.keypoints[1].x == 0.0
