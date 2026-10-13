"""Event engine, audio part: onsets -> strokes (racket pop paired with the wall thud).

A stroke is opened by a racket pop and closed by the next wall thud inside a window derived
from the player's distance to the wall. Floor bounces are attached to the stroke they belong
to (before the wall = no speed for that stroke; after = return bounce). Echoes and spurious
onsets are rejected by timing and class scores. The video side (ball events) is added on top
of this in later phases.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .audio import FLOOR, POP, THUD, UNKNOWN, Onset

NOISE = "noise"


@dataclass
class AudioStroke:
    idx: int
    t_pop: float
    t_thud: float | None = None
    bounce_before_wall: bool = False
    bounces_after: int = 0
    pop_score: float = 0.0
    thud_score: float = 0.0
    closed: bool = False
    reinterpreted: bool = False

    @property
    def flight_s(self) -> float | None:
        return None if self.t_thud is None else self.t_thud - self.t_pop


@dataclass
class ImpactSequencer:
    """Greedy decoder with class scores and distance-aware timing windows."""

    distance_m: float = 4.0  # player-to-wall; updated live from the scene/pose
    echo_gap_s: float = 0.15  # same-class onset this soon after an impact is an echo
    max_open_s: float = 1.6  # a pop with no thud by then is a miss
    noise_prior: float = 0.15
    strokes: list[AudioStroke] = field(default_factory=list)
    bounces: list[float] = field(default_factory=list)
    _last_type: str | None = None
    _last_t: float = -1e9
    _next_idx: int = 1

    # ---- timing model -------------------------------------------------------
    @property
    def pop_thud_window(self) -> tuple[float, float]:
        d = max(1.0, self.distance_m)
        return (d / 40.0, d / 5.0 + 0.1)

    @property
    def thud_to_pop_min_s(self) -> float:
        return max(1.0, self.distance_m) / 40.0  # the return flight is at least as long as the outbound one

    @property
    def pop_to_pop_min_s(self) -> float:
        return 2.0 * self.thud_to_pop_min_s + 0.05

    def _timing(self, label: str, dt: float) -> float:
        """Plausibility of `label` occurring `dt` after the last accepted impact."""
        last = self._last_type
        lo, hi = self.pop_thud_window
        if last is None:
            return 1.0 if label == POP else (0.4 if label == FLOOR else 0.25)
        if label == POP:
            if last == POP:
                return 0.6 if dt >= self.pop_to_pop_min_s else 0.02
            return 1.0 if dt >= self.thud_to_pop_min_s else 0.05
        if label == THUD:
            if last == POP:
                return 1.0 if lo <= dt <= hi else 0.08
            if last == FLOOR:
                return 0.9 if 0.04 <= dt <= hi else 0.08
            # thud after thud: only plausible as "the first one was really the floor" (still inside the window)
            if self.strokes and self.strokes[-1].t_thud is not None and lo <= self._last_t + dt - self.strokes[-1].t_pop <= hi:
                return 0.7
            return 0.05
        if label == FLOOR:
            if last == POP:
                return 0.55 if 0.04 <= dt <= 1.2 else 0.08  # most shots reach the wall without bouncing
            if last == THUD:
                return 0.8 if 0.04 <= dt <= 1.2 else 0.08
            return 0.5 if dt >= 0.08 else 0.05  # double bounce
        return 0.1

    # ---- decoding -----------------------------------------------------------
    def _open(self) -> AudioStroke | None:
        if self.strokes and not self.strokes[-1].closed:
            return self.strokes[-1]
        return None

    def _close_stale(self, now: float) -> list[AudioStroke]:
        s = self._open()
        if s is not None and now - s.t_pop > self.max_open_s:
            s.closed = True
            return [s]
        return []

    def push(self, onset: Onset) -> list[AudioStroke]:
        """Feed one onset; returns strokes completed by it (with or without a thud)."""
        done: list[AudioStroke] = []
        done += self._close_stale(onset.t)
        dt = onset.t - self._last_t
        lo, hi = self.pop_thud_window
        scores = dict(onset.scores) if onset.scores else {POP: 0.34, THUD: 0.33, FLOOR: 0.33}
        cand = {lab: scores.get(lab, 0.0) * self._timing(lab, dt) for lab in (POP, THUD, FLOOR)}
        noise = self.noise_prior
        if self._last_type is not None and dt < self.echo_gap_s and onset.label == self._last_type:
            noise = 0.9  # echo of the previous impact
        best = max(cand, key=cand.get)
        if cand[best] <= noise:
            return done  # spurious
        s = self._open()
        if best == POP:
            if s is not None:  # previous stroke never reached the wall
                s.closed = True
                done.append(s)
            self.strokes.append(AudioStroke(self._next_idx, onset.t, pop_score=scores.get(POP, 0.0)))
            self._next_idx += 1
        elif best == THUD:
            if s is not None:
                s.t_thud = onset.t
                s.thud_score = scores.get(THUD, 0.0)
                s.closed = True
                done.append(s)
            elif self.strokes and self.strokes[-1].t_thud is not None and self._last_type == THUD \
                    and lo <= onset.t - self.strokes[-1].t_pop <= hi:
                # a second low impact still inside the window: the first one was the floor, this is the wall
                prev = self.strokes[-1]
                self.bounces.append(prev.t_thud)
                prev.bounce_before_wall = True
                prev.t_thud = onset.t
                prev.thud_score = scores.get(THUD, 0.0)
                prev.reinterpreted = True
            # a thud with no open pop otherwise: the pop was missed; do not invent a stroke
        else:  # FLOOR
            self.bounces.append(onset.t)
            if s is not None:
                s.bounce_before_wall = True
            elif self.strokes:
                self.strokes[-1].bounces_after += 1
        self._last_type, self._last_t = best, onset.t
        return done

    def flush(self, now: float | None = None) -> list[AudioStroke]:
        s = self._open()
        if s is None:
            return []
        s.closed = True
        return [s]

    # ---- stats --------------------------------------------------------------
    @property
    def count(self) -> int:
        return len(self.strokes)


# =========================================================================== video events

import numpy as np  # noqa: E402

from .ball import Leg, LegTracker, TrackPoint  # noqa: E402
from .config import WallConfig as _WallConfig  # noqa: E402
from .scene import ScenePose  # noqa: E402

WALL_HIT, FLOOR_BOUNCE, RACKET, UNSURE = "wall", "floor", "racket", "unsure"


@dataclass
class BallEvent:
    kind: str
    t: float
    x: float
    y: float
    wall_xy: tuple[float, float] | None = None
    floor_xz: tuple[float, float] | None = None
    score_wall: float = 0.0
    score_floor: float = 0.0
    cues: dict[str, float] = field(default_factory=dict)


def _linefit(points: list[TrackPoint]) -> tuple[np.ndarray, np.ndarray]:
    """Position(t) = a + v t over a few points (least squares)."""
    t = np.array([p.t for p in points])
    xy = np.array([[p.x, p.y] for p in points])
    if len(points) < 2 or t[-1] - t[0] <= 0:
        return xy.mean(axis=0), np.zeros(2)
    A = np.stack([np.ones_like(t), t], axis=1)
    coef, *_ = np.linalg.lstsq(A, xy, rcond=None)
    return coef[0], coef[1]


def reversal_point(before: Leg, after: Leg, k: int = 4) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Sub-frame time and position where the two legs meet, plus the velocities before and after."""
    pb = before.points[-k:]
    pa = after.points[:k]
    a_b, v_b = _linefit(pb)
    a_a, v_a = _linefit(pa)
    dv = v_b - v_a
    da = a_b - a_a
    if np.dot(dv, dv) > 1e-9:
        t_star = -float(np.dot(da, dv) / np.dot(dv, dv))
        t_star = min(max(t_star, pb[-1].t - 0.05), pa[0].t + 0.05)
    else:
        t_star = 0.5 * (pb[-1].t + pa[0].t)
    pos = 0.5 * ((a_b + v_b * t_star) + (a_a + v_a * t_star))
    return t_star, float(pos[0]), float(pos[1]), v_b, v_a


class VideoEventDetector:
    """Classifies each leg reversal as a wall hit, a floor bounce, a racket contact or unsure."""

    def __init__(self, wall: _WallConfig, *, confirm_points: int = 3, wall_margin_m: float = 1.0,
                 wall_top_m: float = 3.5, audio_window_s: float = 0.06):
        self.wall = wall
        self.confirm_points = confirm_points
        self.margin = wall_margin_m
        self.top = wall_top_m
        self.audio_window = audio_window_s
        self.events: list[BallEvent] = []
        self._seen_reversal: tuple[int, int] | None = None

    def update(self, tracker: LegTracker, spose: ScenePose | None, recent_onsets, player_box=None) -> BallEvent | None:
        rev = tracker.last_reversal
        if rev is None or spose is None or len(tracker.legs) < 2:
            return None
        key = (rev[0].index, rev[1].index)
        if key == self._seen_reversal:
            return None
        before, after = tracker.legs[-2], tracker.legs[-1]
        if after.n < self.confirm_points and after.reason_ended is None:
            return None  # wait for a few points after the reversal
        if before.points[-1].index != rev[0].index:
            return None
        self._seen_reversal = key
        ev = self._classify(before, after, spose, recent_onsets, player_box)
        self.events.append(ev)
        return ev

    def _classify(self, before: Leg, after: Leg, spose: ScenePose, recent_onsets, player_box) -> BallEvent:
        t_star, x, y, v_b, v_a = reversal_point(before, after)
        cues: dict[str, float] = {}
        wall_xy = spose.image_to_wall(np.array([[x, y]]))[0]
        floor_xz = spose.image_to_floor(np.array([[x, y]]))[0]
        rect = self.wall.practice_rect
        in_wall_x = (rect.x0_m - self.margin) <= wall_xy[0] <= (rect.x1_m + self.margin)
        above_junction = wall_xy[1] > 0.05
        on_wall = in_wall_x and above_junction and wall_xy[1] < self.top
        cues["wall_y"] = float(wall_xy[1])
        # radius cue (strong from behind): the ball is smallest at the wall
        rb = np.mean([p.r for p in before.points[-3:]])
        ra = np.mean([p.r for p in after.points[:3]])
        r_at = min(before.points[-1].r, after.points[0].r)
        radius_cue = float(((rb - r_at) + (ra - r_at)) / max(r_at, 1e-6))
        cues["radius"] = radius_cue
        phi = np.radians(spose.phi_deg)
        # direction cue (strong from the side): approached along -Z, left along +Z
        P = np.array([wall_xy[0], max(wall_xy[1], 0.2), 0.0])
        base = spose.project(P.reshape(1, 3))[0]
        dz = spose.project((P + np.array([0, 0, 0.5])).reshape(1, 3))[0] - base
        dx = spose.project((P + np.array([0.5, 0, 0])).reshape(1, 3))[0] - base
        w_dir = float(np.linalg.norm(dz) / (np.linalg.norm(dz) + np.linalg.norm(dx) + 1e-9))
        nz = dz / (np.linalg.norm(dz) + 1e-9)
        dir_cue = 1.0 if (np.dot(v_b, -nz) > 0 and np.dot(v_a, nz) > 0) else 0.0
        cues["dir"] = dir_cue * w_dir
        # floor cues: vertical flip (down then up in the image) with the horizontal motion continuing
        vflip = 1.0 if (v_b[1] > 0 and v_a[1] < 0) else 0.0
        hcont = 1.0 if (v_b[0] * v_a[0] > 0) else 0.0
        cues["vflip"] = vflip
        cues["hcont"] = hcont
        # audio
        thud = 0.0
        floor_snd = 0.0
        for o in recent_onsets:
            if abs(o.t - t_star) <= self.audio_window:
                thud = max(thud, o.scores.get("thud", 0.0))
                floor_snd = max(floor_snd, o.scores.get("floor", 0.0))
        cues["thud"] = thud
        # racket contact: reversal inside the (dilated) player box
        racket = False
        if player_box is not None:
            x0, y0, x1, y1 = player_box
            w, h = x1 - x0, y1 - y0
            racket = (x0 - 0.3 * w) <= x <= (x1 + 0.3 * w) and (y0 - 0.2 * h) <= y <= (y1 + 0.2 * h)
        s_wall = (0.9 * float(on_wall) + 0.6 * max(0.0, min(1.0, radius_cue / 0.15)) * float(np.cos(phi))
                  + 0.7 * dir_cue * w_dir + 0.8 * thud)
        s_floor = (0.9 * float(not above_junction) + 0.6 * vflip * hcont + 0.6 * floor_snd
                   + 0.3 * float(0.2 < floor_xz[1] < max(0.3, spose.distance_to_wall_m + 0.5)))
        if racket:
            kind = RACKET
        elif s_wall >= 0.8 and s_wall > s_floor:
            kind = WALL_HIT
        elif s_floor >= 0.8 and s_floor > s_wall:
            kind = FLOOR_BOUNCE
        else:
            kind = UNSURE
        return BallEvent(kind, t_star, x, y, (float(wall_xy[0]), float(wall_xy[1])) if kind == WALL_HIT else None,
                         (float(floor_xz[0]), float(floor_xz[1])) if kind == FLOOR_BOUNCE else None,
                         s_wall, s_floor, cues)
