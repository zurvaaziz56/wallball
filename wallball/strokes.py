"""Per-stroke analysis: forehand/backhand cues, speed from the timed trajectory fit, drag and
sound-propagation corrections, placement. Everything is expressed in the wall world frame so
the camera side, height and distance do not matter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import PlayerConfig, WallConfig
from .pose import L_ANKLE, L_HIP, L_SHOULDER, L_WRIST, R_ANKLE, R_HIP, R_SHOULDER, R_WRIST, midpoint, torso_scale
from .scene import ScenePose

G = 9.81
DRAG_K = 0.02  # per metre, tennis ball at 15-25 m/s
SOUND = 343.0

FOREHAND, BACKHAND, UNKNOWN = "forehand", "backhand", "unknown"


@dataclass
class Stroke:
    idx: int
    t_pop: float
    t_thud: float | None = None
    kind: str = UNKNOWN
    kind_conf: float = 0.0
    cues: dict[str, float] = field(default_factory=dict)
    wall_xy: tuple[float, float] | None = None
    placement_uv: tuple[float, float] | None = None
    bounce_before_wall: bool = False
    launch_kmh: float | None = None
    avg_kmh: float | None = None
    speed_method: str | None = None  # "fit" | "geometry"
    speed_conf: float = 0.0
    fit_rms_px: float | None = None
    distance_m: float | None = None
    flight_s: float | None = None
    confirmed: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["cues"] = {k: round(float(v), 3) for k, v in self.cues.items()}
        return d


# --------------------------------------------------------------------------- geometry helpers


def axis_directions_at(spose: ScenePose, p_world: np.ndarray, step: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Image directions (px per `step` metres) of world +X, +Y, +Z at a world point."""
    base = spose.project(p_world.reshape(1, 3))[0]
    dx = spose.project((p_world + np.array([step, 0, 0])).reshape(1, 3))[0] - base
    dy = spose.project((p_world + np.array([0, step, 0])).reshape(1, 3))[0] - base
    dz = spose.project((p_world + np.array([0, 0, step])).reshape(1, 3))[0] - base
    return dx, dy, dz


def player_world_point(kps: np.ndarray, spose: ScenePose, height_m: float = 1.75, thr: float = 0.4) -> np.ndarray:
    """Approximate 3D position of the player's hip midpoint (metres, wall frame)."""
    if kps[L_ANKLE, 2] >= thr and kps[R_ANKLE, 2] >= thr:
        ankle = midpoint(kps, L_ANKLE, R_ANKLE)
        xz = spose.image_to_floor(ankle.reshape(1, 2))[0]
        return np.array([xz[0], 0.55 * height_m, xz[1]])
    hip = midpoint(kps, L_HIP, R_HIP)
    return spose.point_at_plane_y(hip.reshape(1, 2), 0.55 * height_m)[0]


def player_floor_xz(kps: np.ndarray, spose: ScenePose, thr: float = 0.4) -> np.ndarray | None:
    if kps[L_ANKLE, 2] >= thr and kps[R_ANKLE, 2] >= thr:
        ankle = midpoint(kps, L_ANKLE, R_ANKLE)
        return spose.image_to_floor(ankle.reshape(1, 2))[0]
    return None


# --------------------------------------------------------------------------- forehand / backhand


@dataclass
class KpSample:
    t: float
    kps: np.ndarray
    spose: ScenePose


def classify_stroke(samples: list[KpSample], t_pop: float, player: PlayerConfig,
                    ball_px_at_pop: tuple[float, float] | None = None, thr: float = 0.4) -> tuple[str, float, dict[str, float]]:
    """Viewpoint-weighted cues on the keypoints around contact. Returns (kind, confidence, cues)."""
    sign = player.sign
    wrist_i = R_WRIST if player.handedness == "right" else L_WRIST
    other_wrist_i = L_WRIST if player.handedness == "right" else R_WRIST
    contact = [s for s in samples if t_pop - 0.15 <= s.t <= t_pop + 0.05]
    loaded = [s for s in samples if t_pop - 0.45 <= s.t <= t_pop - 0.15]
    cues: dict[str, float] = {}
    votes: list[tuple[float, float]] = []  # (vote in [-1, 1], weight)

    def axes(s: KpSample):
        P = player_world_point(s.kps, s.spose, player.height_m, thr)
        dx, _, dz = axis_directions_at(s.spose, P)
        nx, nz = np.linalg.norm(dx), np.linalg.norm(dz)
        wx = nx / (nx + nz + 1e-9)  # how visible the lateral axis is
        wz = nz / (nx + nz + 1e-9)  # how visible the depth axis is
        return dx / (nx + 1e-9), dz / (nz + 1e-9), wx, wz

    # --- contact-window cues: lateral position, ball side, wrist swing direction
    lat, ballside, wx_mean = [], [], []
    wrist_track = []
    for s in contact:
        k = s.kps
        if not (k[L_SHOULDER, 2] >= thr and k[R_SHOULDER, 2] >= thr and k[L_HIP, 2] >= thr and k[R_HIP, 2] >= thr):
            continue
        ux, uz, wx, wz = axes(s)
        ts = max(torso_scale(k), 1e-6)
        sh = midpoint(k, L_SHOULDER, R_SHOULDER)
        hip = midpoint(k, L_HIP, R_HIP)
        if k[wrist_i, 2] >= thr:
            lat.append(float(np.dot(k[wrist_i, :2] - sh, ux)) / ts * sign)
            wrist_track.append((s.t, float(np.dot(k[wrist_i, :2], ux)) / ts * sign))
        if ball_px_at_pop is not None and abs(s.t - t_pop) < 0.04:
            ballside.append(float(np.dot(np.array(ball_px_at_pop) - hip, ux)) / ts * sign)
        wx_mean.append(wx)
    wx_c = float(np.mean(wx_mean)) if wx_mean else 0.0
    if lat:
        v = float(np.median(lat))
        cues["lateral"] = v
        votes.append((math.tanh(v / 0.35), 0.8 * wx_c))
    if ballside:
        v = float(np.median(ballside))
        cues["ball_side"] = v
        votes.append((math.tanh(v / 0.35), 1.0 * wx_c))
    if len(wrist_track) >= 3:
        ts_, xs_ = zip(*wrist_track)
        slope = float(np.polyfit(ts_, xs_, 1)[0])  # torso units per second along +X (handedness-signed)
        cues["swing_dir"] = slope
        votes.append((math.tanh(-slope / 4.0), 0.6 * wx_c))  # forehand swings toward -X
    # --- loaded-position cue: which shoulder is nearer the wall (visible from the side)
    depth = []
    wz_mean = []
    for s in loaded:
        k = s.kps
        if not (k[L_SHOULDER, 2] >= thr and k[R_SHOULDER, 2] >= thr):
            continue
        ux, uz, wx, wz = axes(s)
        ts = max(torso_scale(k), 1e-6)
        # +Z is away from the wall: positive means the left shoulder is farther from the wall
        depth.append(float(np.dot(k[L_SHOULDER, :2] - k[R_SHOULDER, :2], uz)) / ts)
        wz_mean.append(wz)
    if depth:
        v = float(np.median(depth))
        cues["shoulder_depth"] = v
        # right-hander forehand: left shoulder toward the wall (v < 0) -> vote positive
        votes.append((math.tanh(-v * sign / 0.3), 0.9 * float(np.mean(wz_mean))))
    # --- two hands together at contact
    two = []
    for s in contact:
        k = s.kps
        if k[L_WRIST, 2] >= thr and k[R_WRIST, 2] >= thr:
            two.append(float(np.linalg.norm(k[L_WRIST, :2] - k[R_WRIST, :2])) / max(torso_scale(k), 1e-6))
    if two:
        d = float(np.median(two))
        cues["wrist_gap"] = d
        if d < 0.35:
            votes.append((-1.0, 1.0 if player.two_handed_backhand else 0.35))
    if not votes:
        return UNKNOWN, 0.0, cues
    wsum = sum(w for _, w in votes)
    if wsum < 0.2:
        return UNKNOWN, 0.0, cues
    score = sum(v * w for v, w in votes) / wsum
    cues["score"] = score
    if score > 0.2:
        return FOREHAND, min(1.0, abs(score)), cues
    if score < -0.2:
        return BACKHAND, min(1.0, abs(score)), cues
    return UNKNOWN, abs(score), cues


# --------------------------------------------------------------------------- speed


def _skew(d: np.ndarray) -> np.ndarray:
    return np.array([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])


def fit_velocity(obs: list[tuple[float, float, float]], spose: ScenePose, p_wall: np.ndarray, t_thud: float,
                 iters: int = 6, refine_time: bool = True, max_shift_s: float = 0.03
                 ) -> tuple[np.ndarray, float, float, np.ndarray, float]:
    """Fit the wall-arrival velocity of a ballistic flight ending at the known wall point.

    obs: (t, u, v) pixel observations of the outbound leg. The impact time is refined within
    +-max_shift_s of `t_thud` (a frame of timing error moves the whole path by many pixels).
    Returns (v_wall, rms_px, condition, points3d, t_thud_refined).
    """
    obs_a = np.asarray(obs, dtype=np.float64)
    t = obs_a[:, 0]
    px = obs_a[:, 1:3]
    C, D = spose.ray(px)

    def geometry(vv, shift):
        tau = t_thud + shift - t
        grav = np.stack([np.zeros_like(tau), 0.5 * G * tau ** 2, np.zeros_like(tau)], axis=1)
        return tau, grav

    # linear least squares on the ray constraints: [d]x (p_wall - grav - C) = tau [d]x v
    tau, grav = geometry(None, 0.0)
    A = np.vstack([tau[i] * _skew(D[i]) for i in range(len(tau))])
    b = np.concatenate([_skew(D[i]) @ (p_wall - grav[i] - C) for i in range(len(tau))])
    v, *_ = np.linalg.lstsq(A, b, rcond=None)
    shift = 0.0

    def points(vv, sh):
        tau_, grav_ = geometry(vv, sh)
        return p_wall[None, :] - vv[None, :] * tau_[:, None] - grav_

    def resid(vv, sh):
        return (spose.project(points(vv, sh)) - px).ravel()

    n_par = 4 if refine_time else 3
    J = np.zeros((2 * len(tau), n_par))
    for _ in range(iters):  # Gauss-Newton on pixel residuals
        r = resid(v, shift)
        for k in range(3):
            dv = np.zeros(3)
            dv[k] = 1e-3
            J[:, k] = (resid(v + dv, shift) - r) / 1e-3
        if refine_time:
            J[:, 3] = (resid(v, shift + 1e-4) - r) / 1e-4
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        v = v + step[:3]
        if refine_time:
            shift = float(np.clip(shift + step[3], -max_shift_s, max_shift_s))
    r = resid(v, shift).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum(r ** 2, axis=1))))
    sv = np.linalg.svd(J[:, :3], compute_uv=False)
    cond = float(sv[0] / max(sv[-1], 1e-9))
    return v, rms, cond, points(v, shift), t_thud + shift


def drag_correct(v_avg: float, distance_m: float, k: float = DRAG_K) -> float:
    """Launch speed from the average flight speed over `distance_m` with quadratic drag."""
    if distance_m <= 0:
        return v_avg
    return v_avg * (math.exp(k * distance_m) - 1.0) / (k * distance_m)


def speeds_from_fit(v_wall: np.ndarray, flight_s: float, p_wall: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """(launch_ms, avg_ms, distance_m, p_launch) from the fitted wall-arrival velocity."""
    T = flight_s
    p0 = p_wall - v_wall * T - np.array([0.0, 0.5 * G * T ** 2, 0.0])
    D = float(np.linalg.norm(p_wall - p0))
    avg = D / T
    return drag_correct(avg, D), avg, D, p0


def geometric_speed(player_xz: np.ndarray, p_wall: np.ndarray, flight_s: float, reach_m: float,
                    contact_height_m: float = 1.0) -> tuple[float, float, float, np.ndarray]:
    p0 = np.array([player_xz[0], contact_height_m, max(0.2, player_xz[1] - reach_m)])
    D = float(np.linalg.norm(p_wall - p0))
    avg = D / flight_s
    return drag_correct(avg, D), avg, D, p0


def propagation_correction(mic_pos: np.ndarray, p_launch: np.ndarray, p_wall: np.ndarray) -> float:
    """Seconds to subtract from the measured pop-to-thud interval (thud travels farther/nearer to the mic)."""
    return (float(np.linalg.norm(p_wall - mic_pos)) - float(np.linalg.norm(p_launch - mic_pos))) / SOUND


def ms_to_kmh(v: float) -> float:
    return v * 3.6
