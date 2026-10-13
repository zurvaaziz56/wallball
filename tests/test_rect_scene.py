"""Measured-rectangle scene mode (no markers): pose from four clicked corners, focal refinement,
optical-flow tracking through camera motion, no drift when static."""
import cv2
import numpy as np
import pytest

from tests.synth import project, synthetic_camera
from wallball.config import CameraConfig, SceneRect
from wallball.scene import RectSceneTracker, focal_from_quad, look_at, rotation_angle_deg

CAM = synthetic_camera()
RECT = SceneRect(width_m=0.9, height_m=2.05, bottom_m=0.0, x0_m=0.6)  # a door frame


_TEXTURE = None


def wall_texture():
    """A fixed random texture living on the wall plane (X -1..4 m, Y 0..3 m), like paint marks and dirt."""
    global _TEXTURE
    if _TEXTURE is None:
        rng = np.random.default_rng(12345)
        tex = np.full((600, 1000), 190, np.uint8)
        for _ in range(900):
            x, y = rng.integers(0, 1000), rng.integers(0, 600)
            r = int(rng.integers(2, 9))
            cv2.circle(tex, (int(x), int(y)), r, int(rng.integers(120, 235)), -1)
        _TEXTURE = cv2.GaussianBlur(tex, (3, 3), 0)
    return _TEXTURE


def render_door(cam: CameraConfig, centre, target=(1.0, 1.2, 0.0), seed=0):
    """Textured wall with a door frame at RECT and a floor line; returns (gray, R, t, corners_px)."""
    R, t = look_at(centre, target)
    rng = np.random.default_rng(seed)
    img = np.full((cam.height, cam.width), 190, np.uint8)
    # texture mapped onto the wall plane so features repeat across viewpoints
    tex = wall_texture()
    wall_corners = np.array([[-1.0, 3.0, 0], [4.0, 3.0, 0], [4.0, 0.0, 0], [-1.0, 0.0, 0]])
    dst = project(cam.K, R, t, wall_corners).astype(np.float32)
    src = np.array([[0, 0], [tex.shape[1], 0], [tex.shape[1], tex.shape[0]], [0, tex.shape[0]]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(tex, M, (cam.width, cam.height), borderMode=cv2.BORDER_CONSTANT, borderValue=190)
    img = warped
    img = cv2.add(img, rng.integers(0, 6, img.shape, dtype=np.uint8))  # sensor noise
    corners = project(cam.K, R, t, RECT.corners_world())
    poly = np.round(corners).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [poly], True, 40, 9)
    inner = project(cam.K, R, t, RECT.corners_world() + np.array([[0.05, -0.05, 0], [-0.05, -0.05, 0], [-0.05, 0.05, 0], [0.05, 0.05, 0]]))
    cv2.polylines(img, [np.round(inner).astype(np.int32).reshape(-1, 1, 2)], True, 120, 3)
    a = project(cam.K, R, t, [[-3.0, 0.0, 0.0]])[0]
    b = project(cam.K, R, t, [[5.0, 0.0, 0.0]])[0]
    cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), 90, 2)
    return img, R, t, corners


VIEWS = [(2.4, 1.7, 6.8), (5.5, 1.5, 4.0), (-3.5, 1.4, 3.5), (2.6, 0.15, 6.5)]


@pytest.mark.parametrize("centre", VIEWS)
def test_pose_from_clicked_corners(centre):
    img, R, t, corners = render_door(CAM, centre)
    clicks = corners + np.random.default_rng(1).normal(0, 1.0, corners.shape)  # a human clicks to about a pixel
    tracker = RectSceneTracker(RECT, CAM, clicks, refine_focal=False)
    pose = tracker.update(img, 0.0)
    assert pose is not None
    assert np.linalg.norm(pose.C - np.array(centre)) < 0.25, pose.C  # a door at 7 m: about 2 deg of tilt ambiguity
    assert rotation_angle_deg(pose.R, R) < 2.5
    # placement: a point on the door maps back to its wall coordinates
    P = np.array([[1.2, 1.0, 0.0]])
    assert np.allclose(pose.image_to_wall(project(CAM.K, R, t, P)), P[:, :2], atol=0.03)


def test_focal_refined_from_oblique_view():
    centre = (5.5, 1.5, 4.0)  # strongly oblique: vanishing points are well defined
    img, R, t, corners = render_door(CAM, centre)
    f_est = focal_from_quad(corners, CAM.cx, CAM.cy, RECT.width_m, RECT.height_m)
    assert f_est is not None and abs(f_est - CAM.fx) / CAM.fx < 0.03, f_est
    wrong = CameraConfig(CAM.width, CAM.height, CAM.fx * 1.2, CAM.fy * 1.2, CAM.cx, CAM.cy, name="nominal")
    tracker = RectSceneTracker(RECT, wrong, corners)
    assert tracker.focal_refined and abs(tracker.cam.fx - CAM.fx) / CAM.fx < 0.03
    pose = tracker.update(img, 0.0)
    assert pose is not None and np.linalg.norm(pose.C - np.array(centre)) < 0.15


def test_frontal_view_keeps_nominal_focal():
    img, R, t, corners = render_door(CAM, (1.0, 1.2, 5.0), target=(1.0, 1.2, 0.0))
    f = focal_from_quad(corners, CAM.cx, CAM.cy, RECT.width_m, RECT.height_m, nominal=CAM.fx)
    assert f is None or abs(f - CAM.fx) / CAM.fx < 0.1


def test_static_camera_does_not_drift():
    centre = (2.4, 1.7, 6.8)
    tracker = None
    pose = None
    for i in range(120):
        img, R, t, corners = render_door(CAM, centre, seed=i)  # fresh sensor noise each frame
        if tracker is None:
            tracker = RectSceneTracker(RECT, CAM, corners, refine_focal=False)
        pose = tracker.update(img, i / 60.0)
    assert pose is not None and pose.motion == "STATIC"
    assert np.linalg.norm(pose.C - np.array(centre)) < 0.03
    assert np.linalg.norm(tracker.quad - tracker.quad_ref) < 1.0


def test_tracks_through_camera_motion():
    tracker = None
    for i in range(40):
        centre = (2.4 + 0.02 * i, 1.7 + 0.005 * i, 6.8 - 0.015 * i)  # handheld drift, 2 cm per frame
        img, R, t, corners = render_door(CAM, centre, seed=i)
        if tracker is None:
            tracker = RectSceneTracker(RECT, CAM, corners, refine_focal=False)
        pose = tracker.update(img, i / 60.0)
        assert pose is not None, f"lost at frame {i}"
        assert np.linalg.norm(pose.C - np.array(centre)) < 0.25, (i, pose.C, centre)
    assert pose.motion == "MOVING"


BIG = SceneRect(width_m=3.0, height_m=2.4, bottom_m=0.0, x0_m=-0.5)  # the whole wall section


def test_big_rectangle_is_accurate():
    from tests import test_rect_scene as m
    old = m.RECT
    m.RECT = BIG
    try:
        for centre in [(2.4, 1.7, 6.8), (5.5, 1.5, 4.0)]:
            img, R, t, corners = render_door(CAM, centre)
            clicks = corners + np.random.default_rng(2).normal(0, 1.0, corners.shape)
            pose = RectSceneTracker(BIG, CAM, clicks, refine_focal=False).update(img, 0.0)
            assert pose is not None
            assert np.linalg.norm(pose.C - np.array(centre)) < 0.15, pose.C
            assert rotation_angle_deg(pose.R, R) < 1.5
    finally:
        m.RECT = old


def test_noisy_transferred_corners_do_not_corrupt_focal():
    """Corners a few pixels off (as after a feature transfer) must not produce a wild focal length."""
    img, R, t, corners = render_door(CAM, (5.5, 1.5, 4.0))
    rng = np.random.default_rng(4)
    bad = corners + rng.normal(0, 5.0, corners.shape)
    nominal = CameraConfig(CAM.width, CAM.height, CAM.fx, CAM.fy, CAM.cx, CAM.cy, name="nominal")
    tracker = RectSceneTracker(RECT, nominal, bad)
    assert abs(tracker.cam.fx - CAM.fx) / CAM.fx < 0.15
