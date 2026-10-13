"""Stroke store: fuses audio strokes, video wall events and pose samples into analysed strokes,
keeps the dashboard statistics and exports the session as JSON."""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import PlayerConfig, WallConfig
from .events import FLOOR_BOUNCE, WALL_HIT, AudioStroke, BallEvent, ImpactSequencer
from .render import DashboardState
from .scene import ScenePose
from .strokes import (BACKHAND, FOREHAND, SOUND, UNKNOWN, KpSample, Stroke, classify_stroke, fit_velocity,
                      geometric_speed, ms_to_kmh, player_floor_xz, propagation_correction, speeds_from_fit)


@dataclass
class BallSample:
    t: float
    x: float
    y: float
    r: float
    leg_id: int


class StrokeStore:
    def __init__(self, wall: WallConfig, player: PlayerConfig, sequencer: ImpactSequencer, *, av_offset_s: float = 0.0,
                 history_s: float = 3.0, max_fit_rms_px: float = 3.0, min_fit_points: int = 3,
                 min_swing: float = 2.5):
        self.wall, self.player, self.sequencer = wall, player, sequencer
        self.min_swing = min_swing  # torso lengths per second
        self.av_offset = av_offset_s  # audio clock minus video clock
        self.history_s = history_s
        self.max_fit_rms = max_fit_rms_px
        self.min_fit_points = min_fit_points
        self.strokes: list[Stroke] = []
        self.kp: deque[KpSample] = deque()
        self.balls: deque[BallSample] = deque()
        self.poses: deque[tuple[float, ScenePose]] = deque()
        self.wall_events: deque[BallEvent] = deque()
        self.floor_events: deque[BallEvent] = deque()
        self.started = time.time()

    # ---- inputs ---------------------------------------------------------------
    def add_pose(self, t: float, kps: np.ndarray, spose: ScenePose | None) -> None:
        self.kp.append(KpSample(t, kps.copy(), spose))
        self._trim(t)

    def _swing_speed(self, t_pop_v: float, window: float = 0.15) -> float | None:
        """Peak speed of the hitting wrist around contact, in torso lengths per second (no scene needed)."""
        from .pose import hitting_wrist, torso_scale

        wi = hitting_wrist(self.player.handedness)
        pts = [(k.t, k.kps) for k in self.kp if abs(k.t - t_pop_v) <= window and k.kps[wi, 2] >= 0.4]
        if len(pts) < 2:
            return None
        best = 0.0
        for (t0, k0), (t1, k1) in zip(pts, pts[1:]):
            dt = t1 - t0
            if dt <= 0:
                continue
            ts = max(torso_scale(k1), 1e-6)
            v = float(np.linalg.norm(k1[wi, :2] - k0[wi, :2])) / ts / dt
            best = max(best, v)
        return best

    def add_scene(self, t: float, spose: ScenePose) -> None:
        self.poses.append((t, spose))
        xz = self._player_xz(t)
        if xz is not None and 0.5 < xz[1] < 12.0:
            self.sequencer.distance_m = float(xz[1])

    def add_ball(self, t: float, x: float, y: float, r: float, leg_id: int) -> None:
        self.balls.append(BallSample(t, x, y, r, leg_id))

    def add_event(self, ev: BallEvent) -> None:
        if ev.kind == FLOOR_BOUNCE:
            self.floor_events.append(ev)
        if ev.kind == WALL_HIT:
            self.wall_events.append(ev)
            # a wall hit seen in video can close a stroke whose thud the mic missed
            self._attach_late_video_hit(ev)

    def _trim(self, t: float) -> None:
        for dq in (self.kp, self.balls, self.poses, self.wall_events, self.floor_events):
            while dq and (dq[0].t if not isinstance(dq[0], tuple) else dq[0][0]) < t - self.history_s:
                dq.popleft()

    # ---- helpers ----------------------------------------------------------------
    def _scene_at(self, t: float) -> ScenePose | None:
        best = None
        for tt, p in self.poses:
            if best is None or abs(tt - t) < abs(best[0] - t):
                best = (tt, p)
        return best[1] if best is not None and abs(best[0] - t) < 0.5 else None

    def _player_xz(self, t: float) -> np.ndarray | None:
        best = None
        for s in self.kp:
            if abs(s.t - t) < 0.25 and (best is None or abs(s.t - t) < abs(best.t - t)):
                best = s
        if best is None:
            return None
        return player_floor_xz(best.kps, best.spose)

    def _ball_at(self, t: float, tol: float = 0.03) -> BallSample | None:
        best = None
        for b in self.balls:
            if abs(b.t - t) <= tol and (best is None or abs(b.t - t) < abs(best.t - t)):
                best = b
        return best

    def _video_time(self, t_audio: float) -> float:
        return t_audio - self.av_offset

    def _attach_late_video_hit(self, ev: BallEvent) -> None:
        for s in reversed(self.strokes):
            if s.t_thud is None and s.wall_xy is None and 0.03 < ev.t - self._video_time(s.t_pop) < 1.6:
                s.t_thud = ev.t + self.av_offset
                s.notes.append("thud time from video (mic missed it)")
                s.wall_xy = ev.wall_xy
                s.placement_uv = self.wall.practice_rect.normalise(*ev.wall_xy)
                self._speed(s, ev)
                return

    # ---- strokes ----------------------------------------------------------------
    def on_audio_stroke(self, a: AudioStroke) -> Stroke:
        s = Stroke(a.idx, a.t_pop, a.t_thud, bounce_before_wall=a.bounce_before_wall)
        if a.t_thud is not None:
            s.flight_s = a.t_thud - a.t_pop
        else:
            s.notes.append("no wall impact heard")
        t_pop_v = self._video_time(a.t_pop)
        # forehand / backhand from the pose around contact (needs the scene for the world axes)
        samples = [k for k in self.kp if t_pop_v - 0.5 <= k.t <= t_pop_v + 0.08 and k.spose is not None]
        ball = self._ball_at(t_pop_v, 0.04)
        s.kind, s.kind_conf, s.cues = classify_stroke(samples, t_pop_v, self.player,
                                                      (ball.x, ball.y) if ball else None)
        if s.kind == UNKNOWN and not samples:
            s.notes.append("no pose around contact")
        swing = self._swing_speed(t_pop_v)
        if swing is not None:
            s.cues["swing_speed"] = swing
        # placement from the matching video wall hit
        ev = None
        if a.t_thud is not None:
            t_thud_v = self._video_time(a.t_thud)
            cands = [e for e in self.wall_events if abs(e.t - t_thud_v) < 0.12]
            if not cands:
                # the mic's "thud" may have been a floor bounce: is there a floor event there and a wall hit later?
                floor_near = any(abs(e.t - t_thud_v) < 0.12 for e in self.floor_events)
                later = [e for e in self.wall_events if 0.05 < e.t - t_thud_v < 0.9]
                if floor_near and later:
                    s.bounce_before_wall = True
                    cands = [later[0]]
                    s.t_thud = later[0].t + self.av_offset
                    s.flight_s = s.t_thud - s.t_pop
                    s.notes.append("floor bounce before the wall (video)")
            if cands:
                ev = min(cands, key=lambda e: abs(e.t - self._video_time(s.t_thud)))
                s.wall_xy = ev.wall_xy
                s.placement_uv = self.wall.practice_rect.normalise(*ev.wall_xy)
            else:
                s.notes.append("ball not seen at the wall")
        self._speed(s, ev)
        # a stroke counts when the wall saw the ball or the skeleton swung a racket; sounds alone do not
        s.confirmed = ev is not None or (swing is not None and swing >= self.min_swing)
        if not s.confirmed:
            s.notes.append("unconfirmed: no wall hit seen and no swing seen")
        self.strokes.append(s)
        return s

    def _speed(self, s: Stroke, ev: BallEvent | None) -> None:
        if s.t_thud is None:
            return
        if s.bounce_before_wall:
            s.notes.append("bounced before the wall: no speed")
            return
        t_pop_v = self._video_time(s.t_pop)
        t_thud_v = ev.t if ev is not None else self._video_time(s.t_thud)
        spose = self._scene_at(t_thud_v)
        if spose is None:
            s.notes.append("no camera pose")
            return
        if spose.rot_rate_dps > 15.0:
            s.notes.append("camera moving too fast")
            return
        dt = s.t_thud - s.t_pop
        player_xz = self._player_xz(t_pop_v)
        p_wall = None
        if s.wall_xy is not None:
            p_wall = np.array([s.wall_xy[0], s.wall_xy[1], 0.0])
        # --- primary: timed trajectory fit on the outbound leg
        if p_wall is not None:
            obs_all = [b for b in self.balls if t_pop_v + 0.01 <= b.t <= t_thud_v - 0.005]
            # keep the last contiguous leg before the wall (drop earlier legs)
            if obs_all:
                last_leg = obs_all[-1].leg_id
                obs = [b for b in obs_all if b.leg_id == last_leg]
            else:
                obs = []
            if len(obs) >= self.min_fit_points:
                v_wall, rms, cond, _, t_hit = fit_velocity([(b.t, b.x, b.y) for b in obs], spose, p_wall, t_thud_v)
                if rms <= self.max_fit_rms and cond < 5e3:
                    # flight time on the video clock: refined impact time minus the true contact time
                    # (the pop reached the mic r_p / c after contact)
                    launch, avg, D, p0 = speeds_from_fit(v_wall, max(0.03, t_hit - t_pop_v), p_wall)
                    r_p = float(np.linalg.norm(p0 - spose.C))
                    T = max(0.03, t_hit - (t_pop_v - r_p / SOUND))
                    launch, avg, D, p0 = speeds_from_fit(v_wall, T, p_wall)
                    s.launch_kmh, s.avg_kmh, s.distance_m, s.flight_s = ms_to_kmh(launch), ms_to_kmh(avg), D, T
                    s.speed_method, s.fit_rms_px = "fit", rms
                    s.speed_conf = float(max(0.2, 1.0 - rms / self.max_fit_rms))
                    return
                s.notes.append(f"fit rejected (rms {rms:.1f} px, cond {cond:.0f})")
            else:
                s.notes.append(f"only {len(obs)} outbound detections")
        # --- fallback: geometry from the player's position
        if p_wall is not None and player_xz is not None:
            launch, avg, D, p0 = geometric_speed(player_xz, p_wall, dt, self.player.reach_m)
            corr = propagation_correction(spose.C, p0, p_wall)
            T = max(0.03, dt - corr)
            launch, avg, D, p0 = geometric_speed(player_xz, p_wall, T, self.player.reach_m)
            s.launch_kmh, s.avg_kmh, s.distance_m, s.flight_s = ms_to_kmh(launch), ms_to_kmh(avg), D, T
            s.speed_method = "geometry"
            s.speed_conf = 0.4 if D > 2.5 else 0.25
            s.notes.append("speed from player position (low confidence)")
        elif player_xz is None:
            s.notes.append("player position unknown")

    # ---- outputs ----------------------------------------------------------------
    @property
    def confirmed(self) -> list[Stroke]:
        return [s for s in self.strokes if s.confirmed]

    def update_dashboard(self, st: DashboardState) -> None:
        strokes = self.confirmed
        st.strokes = len(strokes)
        st.unconfirmed = len(self.strokes) - len(strokes)
        st.forehands = sum(1 for s in strokes if s.kind == FOREHAND)
        st.backhands = sum(1 for s in strokes if s.kind == BACKHAND)
        st.unknown = sum(1 for s in strokes if s.kind == UNKNOWN)
        speeds = [(s, s.launch_kmh) for s in strokes if s.launch_kmh is not None]
        st.peak_kmh = max((v for _, v in speeds), default=None)
        fh = [v for s, v in speeds if s.kind == FOREHAND]
        bh = [v for s, v in speeds if s.kind == BACKHAND]
        st.fh_avg_kmh = float(np.mean(fh)) if fh else None
        st.bh_avg_kmh = float(np.mean(bh)) if bh else None
        st.speed_history = [(s.idx, s.launch_kmh, s.kind) for s in strokes]
        st.placements = [(s.idx, *s.placement_uv) for s in strokes if s.placement_uv is not None]
        if strokes:
            last = strokes[-1]
            st.latest_idx = last.idx
            st.speed_kmh = last.launch_kmh
            st.low_confidence = last.speed_method == "geometry"

    def to_json(self, path: str | Path, extra: dict | None = None) -> None:
        data = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "player": self.player.to_dict(),
            "wall": self.wall.to_dict(),
            "strokes": [s.to_dict() for s in self.strokes],
        }
        if extra:
            data.update(extra)
        Path(path).write_text(json.dumps(data, indent=2, default=float))
