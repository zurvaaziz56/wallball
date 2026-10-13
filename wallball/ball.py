"""Ball detection and per-leg tracking.

Static camera: causal three-frame differencing on blurred luminance, streak-tolerant blob
filter, player box masked out. Moving camera: the caller passes previous frames already
warped into the current view (wall-plane homography), so the same differencing applies.
A learned detector (CoreML) can replace `detect` behind the same interface later.

Tracking is per leg: a leg ends on a gap or a sharp direction reversal, and never
interpolates across one, because reversals are exactly the events we want to find.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Detection:
    t: float
    index: int
    x: float
    y: float
    radius: float  # half the minor axis (ball diameter even when streaked)
    length: float  # major axis
    area: float
    score: float = 0.0

    @property
    def elongation(self) -> float:
        return self.length / max(2.0 * self.radius, 1e-6)


class ClassicBallDetector:
    def __init__(self, *, blur: int = 5, diff_thr: int = 16, min_radius: float = 2.5, max_radius: float = 45.0,
                 max_elongation: float = 7.0, max_candidates: int = 6):
        self.blur = blur | 1
        self.diff_thr = diff_thr
        self.min_radius, self.max_radius = min_radius, max_radius
        self.max_elongation = max_elongation
        self.max_candidates = max_candidates
        self.prev: deque[np.ndarray] = deque(maxlen=2)
        self.last_mask: np.ndarray | None = None
        # exponential background for static cameras (float32); warm-up counted in frames
        self.bg: np.ndarray | None = None
        self.bg_alpha = 0.04
        self.bg_frames = 0
        self.bg_warmup = 20

    def reset(self) -> None:
        self.prev.clear()
        self.bg = None
        self.bg_frames = 0

    def _update_background(self, g: np.ndarray, mask_boxes) -> None:
        if self.bg is None:
            self.bg = g.astype(np.float32)
            self.bg_frames = 1
            return
        mask = None
        if mask_boxes:
            mask = np.full(g.shape[:2], 255, np.uint8)
            for (x0, y0, x1, y1) in mask_boxes:
                w, h = x1 - x0, y1 - y0
                cv2.rectangle(mask, (int(x0 - 0.15 * w), int(y0 - 0.15 * h)), (int(x1 + 0.15 * w), int(y1 + 0.15 * h)), 0, -1)
        cv2.accumulateWeighted(g, self.bg, self.bg_alpha, mask)
        self.bg_frames += 1

    def _background_diff(self, g: np.ndarray) -> np.ndarray | None:
        if self.bg is None or self.bg_frames < self.bg_warmup:
            return None
        bg_u8 = cv2.convertScaleAbs(self.bg)
        # exposure normalisation: match the frame mean to the background mean
        m_bg = float(np.mean(cv2.mean(bg_u8)[:3]))
        m_g = float(np.mean(cv2.mean(g)[:3]))
        gn = g
        if m_g > 1e-3 and abs(m_bg / m_g - 1.0) > 0.01:
            gn = cv2.convertScaleAbs(g, alpha=m_bg / m_g)
        return self._strongest_channel(cv2.absdiff(gn, bg_u8))

    def prepare(self, frame: np.ndarray) -> np.ndarray:
        """Blurred frame (BGR or grey). Colour frames keep chroma, so a yellow ball on a light wall
        still differs strongly in the blue channel even when its luminance matches the wall."""
        return cv2.GaussianBlur(frame, (self.blur, self.blur), 0)

    @staticmethod
    def _strongest_channel(d: np.ndarray) -> np.ndarray:
        if d.ndim == 2:
            return d
        b, g, r = cv2.split(d)
        return cv2.max(cv2.max(b, g), r)

    def detect(self, frame: np.ndarray, t: float, index: int, *, mask_boxes=None, predicted=None,
               radius_range: tuple[float, float] | None = None, prev_override: list[np.ndarray] | None = None,
               roi_mask: np.ndarray | None = None, static: bool = False) -> list[Detection]:
        """Candidates for this frame, best first. `mask_boxes`: (x0, y0, x1, y1) regions to ignore
        (the player). `predicted`: (x, y) from the tracker to rank candidates. `prev_override`: the
        two previous frames warped into this view (moving camera)."""
        g = self.prepare(frame)
        prev = prev_override if prev_override is not None else list(self.prev)
        self.prev.append(g)
        d_bg = None
        if static:
            d_bg = self._background_diff(g)
            self._update_background(g, mask_boxes)
        else:
            self.bg = None
            self.bg_frames = 0
        motion = None
        if len(prev) >= 2:
            motion = self._strongest_channel(cv2.absdiff(g, prev[-1]))  # any change since the last frame
        if d_bg is not None:
            d = d_bg  # static camera: the whole ball blob regardless of how far it moved
        else:
            if len(prev) < 2:
                return []
            d = self._strongest_channel(cv2.min(cv2.absdiff(g, prev[-1]), cv2.absdiff(g, prev[-2])))
        _, binary = cv2.threshold(d, self.diff_thr, 255, cv2.THRESH_BINARY)
        motion_mask = None
        if d_bg is not None and motion is not None:
            _, motion_mask = cv2.threshold(motion, max(6, self.diff_thr // 2), 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if mask_boxes:
            for (x0, y0, x1, y1) in mask_boxes:
                w, h = x1 - x0, y1 - y0
                cv2.rectangle(binary, (int(x0 - 0.15 * w), int(y0 - 0.15 * h)), (int(x1 + 0.15 * w), int(y1 + 0.15 * h)), 0, -1)
        if roi_mask is not None:
            binary = cv2.bitwise_and(binary, roi_mask)
        self.last_mask = binary
        rmin, rmax = radius_range or (self.min_radius, self.max_radius)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: list[Detection] = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < 3.0:
                continue
            (cx, cy), (w, h), _ = cv2.minAreaRect(c)
            minor, major = min(w, h), max(w, h)
            r = max(minor, 1.0) / 2.0
            if r < rmin * 0.7 or r > rmax * 1.5:
                continue
            elong = major / max(minor, 1.0)
            if elong > self.max_elongation:
                continue
            fill = area / max(w * h, 1.0)
            if fill < 0.35:
                continue
            if motion_mask is not None:  # a ball never hovers: the blob must contain fresh motion
                x, y, bw, bh = cv2.boundingRect(c)
                if cv2.countNonZero(motion_mask[y:y + bh, x:x + bw]) < max(2, int(0.05 * area)):
                    continue
            score = 1.0
            # size plausibility
            mid = np.sqrt(rmin * rmax)
            score *= float(np.exp(-abs(np.log(r / mid)) ** 2 / 2.0))
            if predicted is not None:
                dist = float(np.hypot(cx - predicted[0], cy - predicted[1]))
                score *= float(np.exp(-dist / (8.0 * max(r, 3.0))))
            out.append(Detection(t, index, float(cx), float(cy), float(r), float(major), area, score))
        out.sort(key=lambda d: d.score, reverse=True)
        return out[: self.max_candidates]


@dataclass
class TrackPoint:
    t: float
    index: int
    x: float
    y: float
    r: float
    score: float = 0.0

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class Leg:
    points: list[TrackPoint] = field(default_factory=list)
    reason_ended: str | None = None  # "gap" | "reversal"

    @property
    def n(self) -> int:
        return len(self.points)

    def velocity(self, k: int = 3) -> np.ndarray | None:
        """Mean image velocity (px/s) over the last k points."""
        if self.n < 2:
            return None
        pts = self.points[-min(k, self.n):]
        dt = pts[-1].t - pts[0].t
        if dt <= 0:
            return None
        return (pts[-1].xy - pts[0].xy) / dt


class LegTracker:
    """Nearest-neighbour association with a constant-velocity prediction; legs split on reversals."""

    def __init__(self, *, base_gate_px: float = 40.0, first_gate_px: float = 260.0, max_gap_frames: int = 3,
                 reversal_cos: float = -0.3, min_speed_px_s: float = 150.0, trail_len: int = 40):
        self.base_gate = base_gate_px
        self.first_gate = first_gate_px  # no velocity yet: the ball may have moved a lot since the first point
        self.max_gap = max_gap_frames
        self.reversal_cos = reversal_cos
        self.min_speed = min_speed_px_s
        self.legs: list[Leg] = []
        self.gap = 0
        self.trail: deque[TrackPoint] = deque(maxlen=trail_len)
        self.last_reversal: tuple[TrackPoint, TrackPoint] | None = None

    @property
    def current(self) -> Leg | None:
        return self.legs[-1] if self.legs and self.legs[-1].reason_ended is None else None

    def predict(self, t: float) -> tuple[float, float] | None:
        leg = self.current
        if leg is None or leg.n == 0:
            return None
        v = leg.velocity()
        p = leg.points[-1]
        if v is None:
            return (p.x, p.y)
        return (float(p.x + v[0] * (t - p.t)), float(p.y + v[1] * (t - p.t)))

    def _end_leg(self, reason: str) -> None:
        if self.legs and self.legs[-1].reason_ended is None:
            self.legs[-1].reason_ended = reason

    def _is_break(self, v: np.ndarray | None, disp: np.ndarray, dt: float) -> bool:
        """Sharp direction change: a reversal against the leg's velocity, or a down-then-up vertical flip."""
        if v is None or dt <= 0:
            return False
        speed = float(np.linalg.norm(v))
        if speed < self.min_speed or float(np.linalg.norm(disp)) < 1e-6:
            return False
        cos = float(np.dot(disp, v) / (np.linalg.norm(disp) * speed))
        if cos < self.reversal_cos:
            return True
        vy_after = disp[1] / dt
        return v[1] > self.min_speed and vy_after < -self.min_speed

    def _start_leg(self, det: Detection, t: float, index: int, reversal_from: TrackPoint | None) -> TrackPoint:
        pt = TrackPoint(t, index, det.x, det.y, det.radius, det.score)
        if reversal_from is not None:
            self._end_leg("reversal")
            self.last_reversal = (reversal_from, pt)
        self.legs.append(Leg([pt]))
        self.trail.append(pt)
        self.gap = 0
        return pt

    def update(self, detections: list[Detection], t: float, index: int) -> TrackPoint | None:
        leg = self.current
        if leg is None or leg.n == 0:
            if not detections:
                return None
            return self._start_leg(detections[0], t, index, None)
        v = leg.velocity()
        p = leg.points[-1]
        dt = t - p.t
        if v is not None:
            pred = p.xy + v * dt
            gate = self.base_gate + 0.6 * float(np.linalg.norm(v)) * dt
            gate_break = self.base_gate + 1.5 * float(np.linalg.norm(v)) * dt
        else:
            pred = p.xy
            gate = gate_break = self.first_gate
        # hypothesis 1: continuation near the prediction
        best, best_d = None, None
        for d in detections:
            dist = float(np.hypot(d.x - pred[0], d.y - pred[1]))
            if dist <= gate and (best_d is None or dist < best_d):
                best, best_d = d, dist
        if best is not None and not self._is_break(v, np.array([best.x, best.y]) - p.xy, dt):
            pt = TrackPoint(t, index, best.x, best.y, best.radius, best.score)
            leg.points.append(pt)
            self.trail.append(pt)
            self.gap = 0
            return pt
        # hypothesis 2: a break (reversal / bounce) near the last point
        cand, cand_d = None, None
        for d in detections:
            dist = float(np.hypot(d.x - p.x, d.y - p.y))
            if dist <= gate_break and self._is_break(v, np.array([d.x, d.y]) - p.xy, dt):
                if cand_d is None or dist < cand_d:
                    cand, cand_d = d, dist
        if cand is not None:
            return self._start_leg(cand, t, index, p)
        self.gap += 1
        if self.gap > self.max_gap:
            self._end_leg("gap")
        return None

    def trail_xy(self) -> list[tuple[float, float]]:
        return [(p.x, p.y) for p in self.trail]

    def prune(self, before_t: float) -> None:
        self.legs = [l for l in self.legs if l.points and l.points[-1].t >= before_t]
