"""Synthetic scene renderer shared by the tests: draws the wall markers as seen from a chosen camera."""
from __future__ import annotations

import cv2
import numpy as np

from wallball.config import CameraConfig, WallConfig
from wallball.scene import ARUCO_DICTS, look_at


def synthetic_camera(width=1920, height=1440, f=1370.0) -> CameraConfig:
    return CameraConfig(width, height, f, f, width / 2.0, height / 2.0, name="synthetic")


def project(K, R, t, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    cam = (R @ pts.T).T + t
    uv = cam[:, :2] / cam[:, 2:3]
    return uv @ K[:2, :2].T + K[:2, 2]


def render_scene(wall: WallConfig, cam: CameraConfig, centre, target=None, *, marker_ids=None,
                 background=200, floor_line=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render a grayscale frame of the marker wall from camera `centre`. Returns (image, R, t)."""
    if target is None:
        cx, cy = wall.practice_rect.centre
        target = (cx, cy, 0.0)
    R, t = look_at(centre, target)
    K = cam.K
    img = np.full((cam.height, cam.width), background, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[wall.dictionary])
    for m in wall.markers:
        if marker_ids is not None and m.id not in marker_ids:
            continue
        side_px = 400
        # marker with a white quiet zone (2 cells) around it
        cells = 4 + 2
        bitmap = cv2.aruco.generateImageMarker(dictionary, m.id, side_px, borderBits=1)
        quiet = int(side_px * 2 / cells)
        canvas = np.full((side_px + 2 * quiet, side_px + 2 * quiet), 255, dtype=np.uint8)
        canvas[quiet:quiet + side_px, quiet:quiet + side_px] = bitmap
        # world corners of the canvas (quiet zone included)
        h = m.size_m / 2.0
        q = m.size_m * 2 / cells
        world = np.array([
            [m.x_m - h - q, m.y_m + h + q, 0.0],
            [m.x_m + h + q, m.y_m + h + q, 0.0],
            [m.x_m + h + q, m.y_m - h - q, 0.0],
            [m.x_m - h - q, m.y_m - h - q, 0.0],
        ])
        dst = project(K, R, t, world).astype(np.float32)
        S = canvas.shape[0]
        src = np.array([[0, 0], [S, 0], [S, S], [0, S]], dtype=np.float32)
        Hm = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(canvas, Hm, (cam.width, cam.height), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpPerspective(np.full_like(canvas, 255), Hm, (cam.width, cam.height),
                                   flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        img[mask > 0] = warped[mask > 0]
    if floor_line:
        a = project(K, R, t, [[-3.0, 0.0, 0.0]])[0]
        b = project(K, R, t, [[5.0, 0.0, 0.0]])[0]
        cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), 90, 2)
    return img, R, t
