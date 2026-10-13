"""Scene anchor: ArUco markers on the wall -> per-frame camera pose, homographies, viewpoint.

Everything downstream (ball events, placement, speed, forehand/backhand) works in the wall
world frame, so the camera can be behind the player, on either side, on the ground or in
someone's hand. The only requirement is that enough markers are in view.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import CameraConfig, WallConfig

ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}

BEHIND, DIAGONAL, SIDE = "BEHIND", "DIAGONAL", "SIDE"
LEFT, RIGHT = "LEFT", "RIGHT"
GROUND, NORMAL = "GROUND", "NORMAL"
STATIC, MOVING = "STATIC", "MOVING"


def look_at(centre, target, up=(0.0, 1.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """World->camera (R, t) for a camera at `centre` looking at `target` (OpenCV axes: x right, y down, z forward)."""
    c = np.asarray(centre, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64) - c
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, dtype=np.float64))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    t = -R @ c
    return R, t


def rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angle of the relative rotation between two rotation matrices, degrees."""
    Rrel = R1 @ R2.T
    c = (np.trace(Rrel) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


@dataclass
class ScenePose:
    """Camera pose for one frame plus everything derived from it."""

    t: float
    rvec: np.ndarray
    tvec: np.ndarray
    R: np.ndarray
    C: np.ndarray  # camera centre in world (X, Y, Z)
    K: np.ndarray
    dist: np.ndarray
    H_wall: np.ndarray  # (X, Y, 1) on the wall plane -> ideal image (u, v, 1)
    H_floor: np.ndarray  # (X, Z, 1) on the floor plane -> ideal image (u, v, 1)
    n_markers: int
    span_m: float
    rms_px: float
    phi_deg: float  # azimuth from the wall normal (0 = straight on, 90 = along the wall)
    side: str  # LEFT | RIGHT
    height_class: str  # GROUND | NORMAL
    category: str  # BEHIND | DIAGONAL | SIDE
    motion: str  # STATIC | MOVING
    rot_rate_dps: float
    marker_corners: dict[int, np.ndarray] = field(default_factory=dict)  # id -> 4x2 image corners
    stale_s: float = 0.0  # age of the underlying detection (0 when detected this frame)

    # ---- projections ------------------------------------------------------
    def project(self, pts_world: np.ndarray) -> np.ndarray:
        """Nx3 world points -> Nx2 (distorted) pixels."""
        pts = np.asarray(pts_world, dtype=np.float64).reshape(-1, 1, 3)
        img, _ = cv2.projectPoints(pts, self.rvec, self.tvec, self.K, self.dist)
        return img.reshape(-1, 2)

    def undistort(self, uv) -> np.ndarray:
        """Nx2 distorted pixels -> Nx2 ideal pinhole pixels."""
        pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.undistortPoints(pts, self.K, self.dist, P=self.K)
        return out.reshape(-1, 2)

    def _plane_from_image(self, H: np.ndarray, uv) -> np.ndarray:
        ideal = self.undistort(uv)
        Hinv = np.linalg.inv(H)
        homog = np.hstack([ideal, np.ones((ideal.shape[0], 1))]) @ Hinv.T
        return homog[:, :2] / homog[:, 2:3]

    def image_to_wall(self, uv) -> np.ndarray:
        """Pixels -> (X, Y) on the wall plane, metres. Nx2."""
        return self._plane_from_image(self.H_wall, uv)

    def image_to_floor(self, uv) -> np.ndarray:
        """Pixels -> (X, Z) on the floor plane, metres. Nx2."""
        return self._plane_from_image(self.H_floor, uv)

    def wall_to_image(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        pts = np.hstack([xy, np.zeros((xy.shape[0], 1))])
        return self.project(pts)

    def floor_to_image(self, xz) -> np.ndarray:
        xz = np.asarray(xz, dtype=np.float64).reshape(-1, 2)
        pts = np.stack([xz[:, 0], np.zeros(xz.shape[0]), xz[:, 1]], axis=1)
        return self.project(pts)

    def ray(self, uv) -> tuple[np.ndarray, np.ndarray]:
        """Pixel -> (origin, unit direction) of its viewing ray in world coordinates."""
        pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        norm = cv2.undistortPoints(pts, self.K, self.dist).reshape(-1, 2)  # normalised camera coords
        d_cam = np.hstack([norm, np.ones((norm.shape[0], 1))])
        d_world = d_cam @ self.R  # R^T d for each row
        d_world /= np.linalg.norm(d_world, axis=1, keepdims=True)
        return self.C.copy(), d_world

    def point_at_plane_y(self, uv, y_m: float) -> np.ndarray:
        """Intersect pixel rays with the horizontal plane Y = y_m. Returns Nx3 world points."""
        C, d = self.ray(uv)
        s = (y_m - C[1]) / d[:, 1]
        return C[None, :] + s[:, None] * d

    def point_at_plane_z(self, uv, z_m: float) -> np.ndarray:
        """Intersect pixel rays with the plane Z = z_m (parallel to the wall). Returns Nx3."""
        C, d = self.ray(uv)
        s = (z_m - C[2]) / d[:, 2]
        return C[None, :] + s[:, None] * d

    @property
    def distance_to_wall_m(self) -> float:
        return float(self.C[2])

    @property
    def height_m(self) -> float:
        return float(self.C[1])

    @property
    def badge(self) -> str:
        return f"{self.category}-{self.side} {self.height_m:.1f}m high, {self.distance_to_wall_m:.1f}m to wall, {self.motion}"


class SceneTracker:
    """Detects the wall markers and maintains a smoothed, validated camera pose."""

    def __init__(
        self,
        wall: WallConfig,
        cam: CameraConfig,
        *,
        full_every: int = 12,
        lost_every: int = 6,
        static_px: float = 0.75,
        static_window_s: float = 1.0,
        median_window_s: float = 2.0,
        ground_height_m: float = 0.4,
    ):
        self.wall = wall
        self.cam = cam
        self.K = cam.K
        self.dist = cam.dist_coeffs
        self.full_every = full_every
        self.lost_every = lost_every
        self.static_px = static_px
        self.static_window_s = static_window_s
        self.median_window_s = median_window_s
        self.ground_height_m = ground_height_m

        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[wall.dictionary])
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)

        self.obj_by_id = {m.id: m.corners_world() for m in wall.markers}
        self.centre_by_id = {m.id: np.array([m.x_m, m.y_m]) for m in wall.markers}

        self.frame_i = 0
        self.last: ScenePose | None = None
        self.last_corners: dict[int, np.ndarray] = {}
        self.raw_history: deque[tuple[float, np.ndarray, np.ndarray]] = deque()  # (t, rvec, tvec)
        self.corner_history: deque[tuple[float, dict[int, np.ndarray]]] = deque()
        self.category = BEHIND
        self._category_init = False

    # ---- detection --------------------------------------------------------
    def _detect(self, gray: np.ndarray, roi: tuple[int, int, int, int] | None) -> dict[int, np.ndarray]:
        if roi is not None:
            x0, y0, x1, y1 = roi
            sub = gray[y0:y1, x0:x1]
            corners, ids, _ = self.detector.detectMarkers(sub)
            offset = np.array([x0, y0], dtype=np.float32)
        else:
            corners, ids, _ = self.detector.detectMarkers(gray)
            offset = np.zeros(2, dtype=np.float32)
        found: dict[int, np.ndarray] = {}
        if ids is None:
            return found
        for c, i in zip(corners, ids.flatten()):
            i = int(i)
            if i in self.obj_by_id and i not in found:
                found[i] = c.reshape(4, 2) + offset
        return found

    def _roi(self, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
        if not self.last_corners:
            return None
        pts = np.vstack(list(self.last_corners.values()))
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        w, h = x1 - x0, y1 - y0
        mx, my = 0.25 * w + 40, 0.25 * h + 40
        H, W = shape[:2]
        return (
            int(max(0, x0 - mx)),
            int(max(0, y0 - my)),
            int(min(W, x1 + mx)),
            int(min(H, y1 + my)),
        )

    def detect(self, gray: np.ndarray) -> dict[int, np.ndarray]:
        """Marker detection with the ROI/full-frame schedule. Returns id -> 4x2 corners."""
        roi = None
        if not self.last_corners and self.frame_i > 1 and self.frame_i % self.lost_every != 0:
            return {}  # nothing anchored yet: a full-frame search a few times per second is enough
        if self.last_corners and self.frame_i % self.full_every != 0:
            roi = self._roi(gray.shape)
        found = self._detect(gray, roi)
        if roi is not None and len(found) < self.wall.min_markers:
            found = self._detect(gray, None)
        return found

    # ---- pose -------------------------------------------------------------
    def _solve(self, found: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float] | None:
        obj = np.vstack([self.obj_by_id[i] for i in found])
        img = np.vstack([found[i] for i in found]).astype(np.float64)
        ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok or len(rvecs) == 0:
            return None
        candidates = []
        for r, t, e in zip(rvecs, tvecs, np.asarray(errs).flatten()):
            candidates.append((float(e), r.reshape(3), t.reshape(3)))
        candidates.sort(key=lambda c: c[0])
        best = candidates[0]
        if len(candidates) > 1 and self.last is not None:
            second = candidates[1]
            if second[0] < 1.2 * best[0] + 0.1:  # ambiguous: prefer continuity with the previous pose
                R_prev = self.last.R
                d_best = rotation_angle_deg(cv2.Rodrigues(best[1])[0], R_prev)
                d_second = rotation_angle_deg(cv2.Rodrigues(second[1])[0], R_prev)
                if d_second < d_best:
                    best = second
        rvec, tvec = best[1].copy(), best[2].copy()
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, self.K, self.dist, rvec.reshape(3, 1), tvec.reshape(3, 1))
        rvec, tvec = rvec.reshape(3), tvec.reshape(3)
        proj, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, self.K, self.dist)
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        return rvec, tvec, rms

    def _span(self, found: dict[int, np.ndarray]) -> float:
        ids = list(found)
        best = 0.0
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                best = max(best, float(np.linalg.norm(self.centre_by_id[ids[a]] - self.centre_by_id[ids[b]])))
        return best

    def _motion(self, t: float, found: dict[int, np.ndarray]) -> tuple[str, float]:
        """Motion state from corner displacement over the last window, and rotation rate."""
        self.corner_history.append((t, {i: c.copy() for i, c in found.items()}))
        while self.corner_history and t - self.corner_history[0][0] > self.static_window_s:
            self.corner_history.popleft()
        max_disp = 0.0
        t0, old = self.corner_history[0]
        for i, c in found.items():
            if i in old:
                max_disp = max(max_disp, float(np.linalg.norm(c - old[i], axis=1).mean()))
        prev = self.last.motion if self.last is not None else MOVING
        thr = self.static_px * (3.0 if prev == STATIC else 1.0)  # hysteresis: leaving STATIC needs a real move
        motion = STATIC if (max_disp < thr and t - t0 >= 0.5 * self.static_window_s) else MOVING
        if len(self.corner_history) < 2:
            motion = MOVING  # unknown yet: be conservative
        return motion, max_disp

    def _rot_rate(self, t: float) -> float:
        if len(self.raw_history) < 2:
            return 0.0
        t1, r1, _ = self.raw_history[-1]
        for t0, r0, _ in reversed(self.raw_history):
            if t1 - t0 >= 0.08:
                ang = rotation_angle_deg(cv2.Rodrigues(r1)[0], cv2.Rodrigues(r0)[0])
                return ang / (t1 - t0)
        return 0.0

    def _smoothed(self, t: float, motion: str) -> tuple[np.ndarray, np.ndarray]:
        while self.raw_history and t - self.raw_history[0][0] > self.median_window_s:
            self.raw_history.popleft()
        if motion == STATIC and len(self.raw_history) >= 3:
            rv = np.median(np.stack([r for _, r, _ in self.raw_history]), axis=0)
            tv = np.median(np.stack([tt for _, _, tt in self.raw_history]), axis=0)
            return rv, tv
        _, rv, tv = self.raw_history[-1]
        return rv.copy(), tv.copy()

    def _classify(self, C: np.ndarray) -> tuple[float, str, str, str]:
        wx, wy = self.wall.practice_rect.centre
        dx = wx - C[0]
        dz = 0.0 - C[2]
        phi = math.degrees(math.atan2(abs(dx), abs(dz)))
        side = RIGHT if C[0] > wx else LEFT
        height_class = GROUND if C[1] < self.ground_height_m else NORMAL
        # hysteresis on the azimuth category
        if not self._category_init:
            self.category = BEHIND if phi < 35 else (DIAGONAL if phi < 65 else SIDE)
            self._category_init = True
        else:
            if self.category == BEHIND and phi > 40:
                self.category = DIAGONAL
            elif self.category == DIAGONAL and phi < 30:
                self.category = BEHIND
            elif self.category == DIAGONAL and phi > 70:
                self.category = SIDE
            elif self.category == SIDE and phi < 60:
                self.category = DIAGONAL
        return phi, side, height_class, self.category

    def _build(self, t: float, rvec, tvec, found, span, rms, motion, rot_rate) -> ScenePose:
        R, _ = cv2.Rodrigues(rvec)
        C = -R.T @ tvec
        H_wall = self.K @ np.column_stack([R[:, 0], R[:, 1], tvec])
        H_floor = self.K @ np.column_stack([R[:, 0], R[:, 2], tvec])
        phi, side, height_class, category = self._classify(C)
        return ScenePose(
            t=t, rvec=rvec, tvec=tvec, R=R, C=C, K=self.K, dist=self.dist,
            H_wall=H_wall, H_floor=H_floor, n_markers=len(found), span_m=span, rms_px=rms,
            phi_deg=phi, side=side, height_class=height_class, category=category,
            motion=motion, rot_rate_dps=rot_rate, marker_corners=found,
        )

    def update(self, gray: np.ndarray, t: float) -> ScenePose | None:
        """Process one grayscale frame. Returns the current pose, or None when the scene is not anchored."""
        self.frame_i += 1
        found = self.detect(gray)
        span = self._span(found) if found else 0.0
        if len(found) < self.wall.min_markers or span < self.wall.min_span_m:
            self.last_corners = found if found else {}
            return self._stale(t)
        solved = self._solve(found)
        if solved is None:
            return self._stale(t)
        rvec, tvec, rms = solved
        if rms > self.wall.max_reproj_px:
            self.last_corners = found
            return self._stale(t)
        self.last_corners = found
        self.raw_history.append((t, rvec, tvec))
        motion, _ = self._motion(t, found)
        rot_rate = self._rot_rate(t)
        rv, tv = self._smoothed(t, motion)
        pose = self._build(t, rv, tv, found, span, rms, motion, rot_rate)
        self.last = pose
        return pose

    def _stale(self, t: float) -> ScenePose | None:
        """Without a fresh detection: a static camera keeps its frozen pose, a moving one has none."""
        if self.last is None or self.last.motion != STATIC:
            return None
        self.last.stale_s = t - self.last.t
        return self.last if self.last.stale_s < 1.0 else None


# =========================================================================== measured rectangle (no markers)

RECT_ID = 1000


def focal_estimates_from_quad(quad: np.ndarray, cx: float, cy: float, rect_w: float = 1.0, rect_h: float = 1.0) -> list[float]:
    """Focal length estimates (px) from a metric rectangle seen in perspective: one from the
    orthogonality of the rotation columns, one from their equal length. Empty for a frontal view."""
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    world = np.array([[0, rect_h], [rect_w, rect_h], [rect_w, 0], [0, 0]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(world, q).astype(np.float64)
    h1, h2 = H[:, 0], H[:, 1]
    a = np.array([h1[0] - cx * h1[2], h1[1] - cy * h1[2]])
    b = np.array([h2[0] - cx * h2[2], h2[1] - cy * h2[2]])
    span = float(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[3] - q[0])))
    scale = float(np.linalg.norm(H[:2, :2])) + 1e-12
    est = []
    if abs(h1[2] * h2[2]) > (1e-4 * scale / span) ** 2:
        f2 = -float(np.dot(a, b)) / float(h1[2] * h2[2])
        if f2 > 0:
            est.append(float(np.sqrt(f2)))
    den = float(h2[2] ** 2 - h1[2] ** 2)
    if abs(den) > (1e-4 * scale / span) ** 2:
        f2 = (float(np.dot(a, a)) - float(np.dot(b, b))) / den
        if f2 > 0:
            est.append(float(np.sqrt(f2)))
    return est


def focal_from_quad(quad: np.ndarray, cx: float, cy: float, rect_w: float = 1.0, rect_h: float = 1.0,
                    nominal: float | None = None) -> float | None:
    """A trustworthy focal length from the rectangle, or None.

    Both estimates must exist and agree within 10 %, and the result must stay within 25 % of the
    nominal lens when one is given: a few pixels of corner error can otherwise swing a single
    estimate by a factor of two, which would silently corrupt every speed.
    """
    est = focal_estimates_from_quad(quad, cx, cy, rect_w, rect_h)
    if len(est) < 2:
        return None
    lo, hi = min(est), max(est)
    if hi / lo > 1.10:
        return None
    f = float(np.sqrt(lo * hi))
    if nominal is not None and not (0.75 * nominal <= f <= 1.35 * nominal):
        return None
    return f


class RectSceneTracker(SceneTracker):
    """Camera pose from a measured rectangle whose corners were clicked once, then tracked with
    optical flow (relative to the reference frame, so a static camera never drifts)."""

    def __init__(self, rect, cam: CameraConfig, corners_px: np.ndarray, *, refine_focal: bool = True, **kw):
        wall = rect.wall_config()
        super().__init__(wall, cam, **kw)
        self.rect = rect
        self.obj_by_id = {RECT_ID: rect.corners_world()}
        c = rect.practice_rect().centre
        self.centre_by_id = {RECT_ID: np.array(c)}
        self.quad = np.asarray(corners_px, dtype=np.float32).reshape(4, 2)
        self.quad_ref = self.quad.copy()
        self.ref_gray: np.ndarray | None = None
        self.prev_gray: np.ndarray | None = None
        self.lost_frames = 0
        self.focal_refined = False
        if rect.f_px:
            self._set_focal(float(rect.f_px) * cam.width / max(1, rect.frame_width or cam.width))
        elif refine_focal and cam.name.startswith("nominal"):  # never override a real calibration
            f = focal_from_quad(self.quad, cam.cx, cam.cy, rect.width_m, rect.height_m, nominal=cam.fx)
            if f is not None:
                self._set_focal(f)
                self.focal_refined = True
        self.lk = dict(winSize=(25, 25), maxLevel=4,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def _set_focal(self, f: float) -> None:
        self.cam = CameraConfig(self.cam.width, self.cam.height, f, f, self.cam.cx, self.cam.cy, list(self.cam.dist),
                                self.cam.name, self.cam.rms_px)
        self.K = self.cam.K

    def _track(self, gray: np.ndarray) -> bool:
        """Update self.quad from the reference frame (no drift); fall back to frame-to-frame."""
        if self.ref_gray is None:
            self.ref_gray = gray.copy()
            self.prev_gray = gray.copy()
            return True
        p0 = self.quad_ref.reshape(-1, 1, 2)
        guess = self.quad.reshape(-1, 1, 2).copy()
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.ref_gray, gray, p0, guess, flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
        ok = st.flatten() == 1
        if ok.all():
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.ref_gray, p1, p0.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
            fb = np.linalg.norm(back.reshape(-1, 2) - p0.reshape(-1, 2), axis=1)
            if (st2.flatten() == 1).all() and fb.max() < 2.0:
                self.quad = p1.reshape(4, 2)
                self.prev_gray = gray.copy()
                self.lost_frames = 0
                return True
        # frame-to-frame fallback (handles larger appearance changes, may drift slowly)
        assert self.prev_gray is not None
        p0f = self.quad.reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0f, None, **self.lk)
        if (st.flatten() == 1).all():
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p1, None, **self.lk)
            fb = np.linalg.norm(back.reshape(-1, 2) - p0f.reshape(-1, 2), axis=1)
            if (st2.flatten() == 1).all() and fb.max() < 2.0:
                self.quad = p1.reshape(4, 2)
                self.prev_gray = gray.copy()
                self.lost_frames = 0
                return True
        self.lost_frames += 1
        self.prev_gray = gray.copy()
        return False

    def update(self, gray: np.ndarray, t: float) -> ScenePose | None:
        self.frame_i += 1
        tracked = self._track(gray)
        if not tracked and (self.last is None or self.last.motion != STATIC or self.lost_frames > 60):
            return self._stale(t)
        found = {RECT_ID: self.quad.astype(np.float64)}
        solved = self._solve(found)
        if solved is None:
            return self._stale(t)
        rvec, tvec, rms = solved
        self.last_corners = found
        self.raw_history.append((t, rvec, tvec))
        motion, _ = self._motion(t, found)
        rot_rate = self._rot_rate(t)
        rv, tv = self._smoothed(t, motion)
        pose = self._build(t, rv, tv, found, self.rect.width_m, rms, motion, rot_rate)
        self.last = pose
        return pose
