"""Dashboard rendering: wall-anchored overlays on the video plus the stats panel (OpenCV only).

Layout mirrors the reference reel: video on top; a row of three tiles (strokes,
forehand/backhand, speed); a row with the per-stroke speed bars and the placement map.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import WallConfig
from .pose import SKELETON
from .scene import ScenePose

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LABEL = (170, 68, 68)  # dark blue-ish, BGR
GREY = (170, 170, 170)
DARK = (40, 40, 40)
GREEN = (70, 200, 70)
CYAN = (230, 200, 40)
YELLOW = (0, 230, 255)
MAGENTA = (200, 60, 200)
FONT = cv2.FONT_HERSHEY_SIMPLEX
MONO = cv2.FONT_HERSHEY_PLAIN


def speed_colour(kmh: float | None) -> tuple[int, int, int]:
    if kmh is None:
        return GREY
    if kmh < 45:
        return (90, 200, 90)
    if kmh < 60:
        return (60, 210, 210)
    if kmh < 75:
        return (0, 150, 255)
    return (0, 70, 255)


@dataclass
class DashboardState:
    strokes: int = 0
    unconfirmed: int = 0
    forehands: int = 0
    backhands: int = 0
    unknown: int = 0
    speed_kmh: float | None = None
    peak_kmh: float | None = None
    fh_avg_kmh: float | None = None
    bh_avg_kmh: float | None = None
    speed_history: list[tuple[int, float | None, str]] = field(default_factory=list)  # (idx, kmh, kind)
    placements: list[tuple[int, float, float]] = field(default_factory=list)  # (idx, u, v), v grows upward
    latest_idx: int | None = None
    clock_s: float = 0.0
    fps_capture: float = 0.0
    fps_proc: float = 0.0
    badge: str = ""
    status: str = ""
    speed_label: str = "launch speed"
    low_confidence: bool = False


# --------------------------------------------------------------------------- overlays (full-res frame)


def _th(frame: np.ndarray) -> int:
    return max(1, int(round(frame.shape[1] / 800)))


def _pt(p, scale: float) -> tuple[int, int]:
    return (int(round(p[0] * scale)), int(round(p[1] * scale)))


def draw_scene(frame: np.ndarray, pose: ScenePose, wall: WallConfig, scale: float = 1.0) -> None:
    """Marker outlines, practice rectangle and the wall-floor line; coordinates are full-res * scale."""
    th = _th(frame)
    for mid, corners in pose.marker_corners.items():
        pts = np.round(corners * scale).astype(np.int32).reshape(-1, 1, 2)
        if mid >= 1000:  # the clicked/tracked rectangle: corner dots, no id
            for q in pts.reshape(-1, 2):
                cv2.circle(frame, tuple(int(v) for v in q), 4 * th, MAGENTA, -1)
            continue
        cv2.polylines(frame, [pts], True, GREEN, th)
        c = corners.mean(axis=0) * scale
        cv2.putText(frame, str(mid), (int(c[0]) - 6 * th, int(c[1]) + 4 * th), FONT, 0.5 * th, GREEN, th)
    rect = pose.project(wall.practice_rect.corners_world()) * scale
    cv2.polylines(frame, [np.round(rect).astype(np.int32).reshape(-1, 1, 2)], True, CYAN, th)
    x0, x1 = wall.practice_rect.x0_m - 0.5, wall.practice_rect.x1_m + 0.5
    line = pose.project(np.array([[x0, 0.0, 0.0], [x1, 0.0, 0.0]]))
    cv2.line(frame, _pt(line[0], scale), _pt(line[1], scale), CYAN, max(1, th // 2))


def draw_skeleton(frame: np.ndarray, kps: np.ndarray, thr: float = 0.4, colour=(80, 230, 80), scale: float = 1.0) -> None:
    th = _th(frame)
    for a, b in SKELETON:
        if kps[a, 2] >= thr and kps[b, 2] >= thr:
            cv2.line(frame, _pt(kps[a, :2], scale), _pt(kps[b, :2], scale), colour, th)
    for i in range(kps.shape[0]):
        if kps[i, 2] >= thr:
            cv2.circle(frame, _pt(kps[i, :2], scale), 2 * th, colour, -1)


def draw_trail(frame: np.ndarray, pts: list[tuple[float, float]], radius: float | None = None, scale: float = 1.0) -> None:
    """Gradient trail from yellow (old) to blue (new), circle on the latest point."""
    th = _th(frame)
    n = len(pts)
    for i in range(1, n):
        a = 1.0 - (i / max(1, n - 1))
        colour = (int(255 * (1 - a) + 0 * a), int(60 * (1 - a) + 230 * a), int(30 * (1 - a) + 255 * a))
        cv2.line(frame, _pt(pts[i - 1], scale), _pt(pts[i], scale), colour, th)
    if n:
        r = int(radius * scale) if radius else 8 * th
        cv2.circle(frame, _pt(pts[-1], scale), max(r, 4 * th), (255, 80, 30), th)


def draw_impacts(frame: np.ndarray, pose: ScenePose, impacts: list[tuple[int, float, float]],
                 latest_idx: int | None, scale: float = 1.0) -> None:
    """Numbered circles anchored to the wall (impacts as (idx, X_m, Y_m))."""
    if not impacts:
        return
    th = _th(frame)
    xy = np.array([[x, y] for _, x, y in impacts])
    px = pose.wall_to_image(xy) * scale
    r = 9 * th
    for (idx, _, _), p in zip(impacts, px):
        centre = tuple(np.round(p).astype(int))
        colour = GREEN if idx == latest_idx else (230, 120, 40)
        cv2.circle(frame, centre, r, WHITE, -1)
        cv2.circle(frame, centre, r, colour, th)
        label = str(idx)
        (tw, tht), _ = cv2.getTextSize(label, FONT, 0.45 * th, th)
        cv2.putText(frame, label, (centre[0] - tw // 2, centre[1] + tht // 2), FONT, 0.45 * th, colour, th)


# --------------------------------------------------------------------------- dashboard


class Renderer:
    def __init__(self, display_width: int = 1280):
        self.W = int(display_width)
        self.stats_h = int(round(self.W * 0.34))
        self.s = self.W / 1280.0  # text scale
        self._stats_key: tuple | None = None
        self._stats_img: np.ndarray | None = None

    def _text(self, img, text, xy, scale, colour=BLACK, thick=1, font=FONT):
        cv2.putText(img, text, (int(xy[0]), int(xy[1])), font, scale * self.s, colour, max(1, int(round(thick * self.s))),
                    cv2.LINE_AA)

    def _tile(self, img, x0, y0, x1, y1, label):
        cv2.rectangle(img, (x0, y0), (x1, y1), BLACK, max(1, int(2 * self.s)))
        self._text(img, label, (x0 + 14 * self.s, y0 + 26 * self.s), 0.62, LABEL, 1, MONO)

    def _stats(self, st: DashboardState) -> np.ndarray:
        W, H = self.W, self.stats_h
        img = np.full((H, W, 3), 255, dtype=np.uint8)
        row1 = int(H * 0.42)
        # --- row 1: three tiles
        tw = W // 3
        self._tile(img, 0, 0, tw, row1, "strokes")
        self._text(img, str(st.strokes), (18 * self.s, row1 * 0.72), 2.2, BLACK, 5)
        if st.unconfirmed:
            self._text(img, f"+{st.unconfirmed} unconfirmed", (18 * self.s, row1 * 0.92), 0.5, GREY, 1)
        self._tile(img, tw, 0, 2 * tw, row1, "forehand / backhand")
        self._text(img, f"{st.forehands} / {st.backhands}", (tw + 18 * self.s, row1 * 0.72), 2.2, BLACK, 5)
        if st.unknown:
            self._text(img, f"+{st.unknown} unknown", (tw + 18 * self.s, row1 * 0.92), 0.5, GREY, 1)
        self._tile(img, 2 * tw, 0, W - 1, row1, "shot speed")
        speed_txt = f"{st.speed_kmh:.0f}" if st.speed_kmh is not None else "--"
        self._text(img, speed_txt, (2 * tw + 18 * self.s, row1 * 0.68), 2.2, BLACK, 5)
        self._text(img, "km/h", (2 * tw + 18 * self.s + (len(speed_txt) * 42 + 10) * self.s, row1 * 0.68), 0.6, DARK, 1)
        peak = f"{st.peak_kmh:.0f}" if st.peak_kmh is not None else "--"
        fh = f"{st.fh_avg_kmh:.0f}" if st.fh_avg_kmh is not None else "--"
        bh = f"{st.bh_avg_kmh:.0f}" if st.bh_avg_kmh is not None else "--"
        base_x = 2 * tw + 18 * self.s
        self._text(img, f"peak: {peak} km/h", (base_x, row1 * 0.80), 0.55, DARK, 1, MONO)
        self._text(img, f"forehand avg: {fh} km/h", (base_x, row1 * 0.89), 0.55, DARK, 1, MONO)
        self._text(img, f"backhand avg: {bh} km/h", (base_x, row1 * 0.98), 0.55, DARK, 1, MONO)
        if st.low_confidence:
            self._text(img, "low confidence", (2 * tw + tw * 0.62, row1 * 0.30), 0.5, (0, 90, 230), 1, MONO)
        self._text(img, st.speed_label, (2 * tw + tw * 0.62, row1 * 0.16), 0.5, GREY, 1, MONO)
        # --- row 2: speed history | placement
        y0 = row1
        half = W // 2
        self._tile(img, 0, y0, half, H - 1, "speed history")
        self._speed_bars(img, st, 0, y0, half, H - 1)
        self._tile(img, half, y0, W - 1, H - 1, "shot placement")
        self._placement(img, st, half, y0, W - 1, H - 1)
        return img

    def _speed_bars(self, img, st: DashboardState, x0, y0, x1, y1):
        hist = st.speed_history[-16:]
        peak = max([s for _, s, _ in hist if s is not None] + [st.peak_kmh or 0.0, 60.0])
        ymax = float(int(np.ceil(peak / 20.0)) * 20)
        self._text(img, f"max {peak:.0f} km/h" if hist else "", (x1 - 150 * self.s, y0 + 26 * self.s), 0.55, DARK, 1, MONO)
        ax0, ay0 = int(x0 + 62 * self.s), int(y0 + 40 * self.s)
        ax1, ay1 = int(x1 - 20 * self.s), int(y1 - 34 * self.s)
        cv2.line(img, (ax0, ay0), (ax0, ay1), DARK, 1)
        cv2.line(img, (ax0, ay1), (ax1, ay1), DARK, 1)
        for v in (0, ymax / 2, ymax):
            y = int(ay1 - (ay1 - ay0) * v / ymax)
            self._text(img, f"{v:.0f}", (x0 + 24 * self.s, y + 5 * self.s), 0.5, DARK, 1, MONO)
            cv2.line(img, (ax0 - 4, y), (ax0, y), DARK, 1)
        self._text(img, "stroke #", ((ax0 + ax1) / 2 - 30 * self.s, y1 - 10 * self.s), 0.5, DARK, 1, MONO)
        if not hist:
            return
        n = len(hist)
        slot = (ax1 - ax0) / max(n, 8)
        bw = int(slot * 0.62)
        for k, (idx, kmh, kind) in enumerate(hist):
            bx = int(ax0 + slot * k + slot * 0.19)
            if kmh is None:
                top = ay1 - int((ay1 - ay0) * 0.06)
                cv2.rectangle(img, (bx, top), (bx + bw, ay1), GREY, -1)
                self._text(img, "?", (bx + bw / 2 - 5 * self.s, top - 4 * self.s), 0.45, GREY, 1, MONO)
            else:
                top = int(ay1 - (ay1 - ay0) * min(kmh, ymax) / ymax)
                cv2.rectangle(img, (bx, top), (bx + bw, ay1), speed_colour(kmh), -1)
            if idx == st.latest_idx:
                cv2.rectangle(img, (bx - 1, ay0), (bx + bw + 1, ay1), (200, 200, 200), 1)
            self._text(img, str(idx), (bx + bw / 2 - 5 * self.s, ay1 + 16 * self.s), 0.5, DARK, 1, MONO)
            if kind in ("forehand", "backhand"):
                self._text(img, kind[0].upper(), (bx + bw / 2 - 5 * self.s, ay0 - 4 * self.s), 0.45, DARK, 1, MONO)

    def _placement(self, img, st: DashboardState, x0, y0, x1, y1):
        px0, py0 = int(x0 + 30 * self.s), int(y0 + 44 * self.s)
        px1, py1 = int(x1 - 30 * self.s), int(y1 - 22 * self.s)
        cv2.rectangle(img, (px0, py0), (px1, py1), DARK, 1)
        r = int(11 * self.s)
        for idx, u, v in st.placements:
            cx = int(px0 + (px1 - px0) * min(max(u, -0.05), 1.05))
            cy = int(py1 - (py1 - py0) * min(max(v, -0.05), 1.05))
            colour = GREEN if idx == st.latest_idx else DARK
            cv2.circle(img, (cx, cy), r, WHITE, -1)
            cv2.circle(img, (cx, cy), r, colour, max(1, int(2 * self.s)))
            label = str(idx)
            (tw, th), _ = cv2.getTextSize(label, MONO, 0.9 * self.s, 1)
            cv2.putText(img, label, (cx - tw // 2, cy + th // 2), MONO, 0.9 * self.s, colour, 1, cv2.LINE_AA)

    def _stats_cached(self, st: DashboardState) -> np.ndarray:
        key = (st.strokes, st.unconfirmed, st.forehands, st.backhands, st.unknown, st.speed_kmh, st.peak_kmh,
               st.fh_avg_kmh, st.bh_avg_kmh, len(st.speed_history), len(st.placements), st.latest_idx, st.low_confidence)
        if key != self._stats_key or self._stats_img is None:
            self._stats_img = self._stats(st)
            self._stats_key = key
        return self._stats_img

    def compose(self, frame_bgr: np.ndarray, st: DashboardState, overlays=None) -> np.ndarray:
        """Downscale the video, draw overlays at that scale (cheap), add the HUD and the cached stats panel."""
        h, w = frame_bgr.shape[:2]
        vh = int(round(h * self.W / w))
        video = cv2.resize(frame_bgr, (self.W, vh), interpolation=cv2.INTER_LINEAR)
        if overlays is not None:
            overlays(video, self.W / w)
        # HUD
        lines = [l for l in (st.badge, st.status) if l]
        lines.append(f"{st.clock_s:6.1f}s  cap {st.fps_capture:4.1f} fps  proc {st.fps_proc:4.1f} fps")
        pad = int(8 * self.s)
        lh = int(22 * self.s)
        box_w = int(max(len(l) for l in lines) * 11.5 * self.s) + 2 * pad
        box_h = min(video.shape[0], lh * len(lines) + pad)
        box_w = min(video.shape[1], box_w)
        roi = video[0:box_h, 0:box_w]
        overlay = np.full_like(roi, 20)
        cv2.addWeighted(overlay, 0.55, roi, 0.45, 0, roi)
        for i, line in enumerate(lines):
            colour = (80, 80, 255) if line.startswith("NO SCENE") else WHITE
            cv2.putText(video, line, (pad, pad + lh * (i + 1) - int(6 * self.s)), MONO, 1.1 * self.s, colour, 1, cv2.LINE_AA)
        return np.vstack([video, self._stats_cached(st)])
