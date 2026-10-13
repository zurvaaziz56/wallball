import math

import numpy as np
import pytest

from wallball.config import WallConfig
from wallball.scene import BEHIND, DIAGONAL, GROUND, LEFT, NORMAL, RIGHT, SIDE, SceneTracker, rotation_angle_deg
from tests.synth import project, render_scene, synthetic_camera

WALL = WallConfig.default()
CAM = synthetic_camera()


def solve(centre, **kw):
    img, R, t = render_scene(WALL, CAM, centre, **kw)
    tracker = SceneTracker(WALL, CAM)
    pose = tracker.update(img, 0.0)
    return pose, R, t, img


@pytest.mark.parametrize(
    "centre, category, side, height",
    [
        ((1.0, 1.7, 6.0), BEHIND, RIGHT, NORMAL),       # tripod straight behind
        ((5.0, 1.6, 4.0), DIAGONAL, RIGHT, NORMAL),     # 45 deg diagonal from the right
        ((-3.5, 1.5, 2.0), SIDE, LEFT, NORMAL),         # side view from the left, close to the wall
        ((1.0, 0.12, 4.0), BEHIND, RIGHT, GROUND),      # phone lying on the floor
        ((-1.0, 1.8, 7.5), BEHIND, LEFT, NORMAL),       # far and slightly left
    ],
)
def test_pose_recovery_and_viewpoint(centre, category, side, height):
    pose, R, t, _ = solve(centre)
    assert pose is not None, "scene should anchor from this viewpoint"
    assert pose.n_markers >= 3
    dist = float(np.linalg.norm(np.array(centre)))
    assert np.linalg.norm(pose.C - np.array(centre)) < 0.01 + 0.005 * dist, pose.C
    assert rotation_angle_deg(pose.R, R) < 0.3
    assert pose.rms_px < 1.0
    assert pose.category == category
    assert pose.side == side
    assert pose.height_class == height


def test_azimuth_value():
    pose, _, _, _ = solve((1.0 + 5.0, 1.55, 5.0))  # 45 degrees from the rect centre
    assert pose is not None
    assert abs(pose.phi_deg - 45.0) < 1.0


def test_not_enough_markers():
    pose, _, _, _ = solve((1.0, 1.7, 6.0), marker_ids=[0, 2])
    assert pose is None


def test_homography_round_trips():
    pose, R, t, _ = solve((2.5, 1.4, 5.5))
    assert pose is not None
    wall_pts = np.array([[0.3, 1.2, 0.0], [1.8, 2.1, 0.0], [1.0, 0.9, 0.0]])
    px = project(CAM.K, R, t, wall_pts)
    # against ground truth (pose estimated from a rendered image: centimetre / few-pixel accuracy)
    assert np.allclose(pose.image_to_wall(px), wall_pts[:, :2], atol=0.01)
    assert np.allclose(pose.wall_to_image(wall_pts[:, :2]), px, atol=3.0)
    floor_pts = np.array([[0.5, 0.0, 1.0], [1.5, 0.0, 2.5]])
    px_f = project(CAM.K, R, t, floor_pts)
    assert np.allclose(pose.image_to_floor(px_f), floor_pts[:, [0, 2]], atol=0.03)
    assert np.allclose(pose.floor_to_image(floor_pts[:, [0, 2]]), px_f, atol=3.0)
    # exact self-consistency of the inverse mappings
    assert np.allclose(pose.wall_to_image(pose.image_to_wall(px)), px, atol=1e-3)
    assert np.allclose(pose.floor_to_image(pose.image_to_floor(px_f)), px_f, atol=1e-3)


def test_rays_and_plane_intersections():
    pose, R, t, _ = solve((-2.0, 1.4, 4.0))
    assert pose is not None
    P = np.array([[1.2, 1.0, 2.5]])  # a ball in flight
    px = project(CAM.K, R, t, P)
    C, d = pose.ray(px)
    # the ray must pass through P
    s = np.dot(P[0] - C, d[0])
    assert np.linalg.norm(C + s * d[0] - P[0]) < 0.01
    # intersecting with the plane Z = 2.5 recovers P
    assert np.allclose(pose.point_at_plane_z(px, 2.5)[0], P[0], atol=0.01)
    assert np.allclose(pose.point_at_plane_y(px, 1.0)[0], P[0], atol=0.03)  # grazing plane amplifies pose noise


def test_static_then_moving_states():
    tracker = SceneTracker(WALL, CAM)
    img, _, _ = render_scene(WALL, CAM, (1.0, 1.7, 6.0))
    pose = None
    for i in range(80):  # 1.33 s at 60 fps, same frame each time
        pose = tracker.update(img, i / 60.0)
    assert pose is not None and pose.motion == "STATIC"
    # now the camera jumps: state must flip to MOVING within a few frames
    img2, _, _ = render_scene(WALL, CAM, (1.4, 1.6, 5.6))
    for i in range(80, 84):
        pose = tracker.update(img2, i / 60.0)
    assert pose is not None and pose.motion == "MOVING"
    assert np.linalg.norm(pose.C - np.array([1.4, 1.6, 5.6])) < 0.05
