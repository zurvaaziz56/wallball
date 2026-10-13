import numpy as np
import pytest

from tests.synth import project, synthetic_camera
from tests.test_rect_scene import RECT, render_door
from wallball.config import CameraConfig
from wallball.measure import RACKET_M, aspect_from_perspective, measure_rectangle, transfer_corners
from wallball.scene import look_at

CAM = synthetic_camera()


def racket_px(R, t, K):
    """A racket held flat against the wall inside the door: 68.6 cm horizontal at 1.2 m height."""
    a = np.array([0.7, 1.2, 0.0])
    b = a + np.array([RACKET_M, 0.0, 0.0])
    return project(K, R, t, np.vstack([a, b]))


@pytest.mark.parametrize("centre", [(2.4, 1.7, 6.8), (5.5, 1.5, 4.0), (1.0, 1.3, 5.0), (-2.5, 1.2, 4.0)])
def test_rectangle_size_from_racket(centre):
    img, R, t, corners = render_door(CAM, centre)
    rng = np.random.default_rng(3)
    clicks = corners + rng.normal(0, 1.0, corners.shape)
    ref = racket_px(R, t, CAM.K) + rng.normal(0, 1.0, (2, 2))
    floor = project(CAM.K, R, t, np.array([[1.0, 0.0, 0.0]]))[0]  # a point on the floor line
    m = measure_rectangle(clicks, CAM.K, ref, RACKET_M, floor_px=floor)
    assert abs(m.width_m - RECT.width_m) / RECT.width_m < 0.04, (centre, m)
    assert abs(m.height_m - RECT.height_m) / RECT.height_m < 0.04, (centre, m)
    assert m.bottom_m < 0.08, m


def test_size_with_nominal_lens_is_still_reasonable():
    centre = (5.5, 1.5, 4.0)
    img, R, t, corners = render_door(CAM, centre)
    nominal = CameraConfig(CAM.width, CAM.height, CAM.fx * 1.15, CAM.fy * 1.15, CAM.cx, CAM.cy)
    m = measure_rectangle(corners, nominal.K, racket_px(R, t, CAM.K), RACKET_M)
    assert abs(m.width_m - RECT.width_m) / RECT.width_m < 0.12
    assert abs(m.height_m - RECT.height_m) / RECT.height_m < 0.12


def test_window_height_above_floor():
    win = np.array([[0.4, 2.0, 0], [1.6, 2.0, 0], [1.6, 1.0, 0], [0.4, 1.0, 0]])  # a 1.2 x 1.0 window, sill at 1.0 m
    centre = (3.0, 1.5, 5.0)
    R, t = look_at(centre, (1.0, 1.5, 0.0))
    quad = project(CAM.K, R, t, win)
    ref = project(CAM.K, R, t, np.array([[0.5, 1.5, 0], [0.5 + RACKET_M, 1.5, 0]]))
    floor = project(CAM.K, R, t, np.array([[1.0, 0.0, 0.0]]))[0]
    m = measure_rectangle(quad, CAM.K, ref, RACKET_M, floor_px=floor)
    assert abs(m.width_m - 1.2) < 0.03 and abs(m.height_m - 1.0) < 0.03
    assert abs(m.bottom_m - 1.0) < 0.04, m


def test_corner_transfer_between_views():
    ref_img, R0, t0, q0 = render_door(CAM, (2.4, 1.7, 6.8), seed=1)
    img, R1, t1, q1 = render_door(CAM, (3.5, 1.4, 5.5), seed=2)
    quad, inliers, err = transfer_corners(ref_img, q0, img)
    assert quad is not None and inliers >= 12, (inliers, err)
    assert np.abs(quad - q1).max() < 4.0, (quad, q1)
