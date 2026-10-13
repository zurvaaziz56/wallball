import numpy as np
import pytest

from tests.synth import project, synthetic_camera
from wallball.config import PlayerConfig, WallConfig
from wallball.pose import (L_ANKLE, L_EAR, L_ELBOW, L_EYE, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, NOSE, R_ANKLE, R_EAR,
                           R_ELBOW, R_EYE, R_HIP, R_KNEE, R_SHOULDER, R_WRIST)
from wallball.scene import SceneTracker, look_at
from wallball.strokes import (BACKHAND, FOREHAND, G, KpSample, classify_stroke, fit_velocity, geometric_speed,
                              propagation_correction, speeds_from_fit)

WALL = WallConfig.default()
CAM = synthetic_camera()


def pose_at(centre):
    """A ScenePose built from an exact synthetic camera (no rendering noise)."""
    from tests.synth import render_scene
    img, R, t = render_scene(WALL, CAM, centre)
    tracker = SceneTracker(WALL, CAM)
    p = tracker.update(img, 0.0)
    assert p is not None
    return p


VIEWS = {
    "behind": (1.0, 1.7, 6.5),
    "right_side": (5.5, 1.5, 4.0),
    "left_side": (-3.5, 1.4, 3.5),
    "ground": (0.4, 0.15, 5.5),
    "diagonal": (4.5, 1.6, 6.0),
}


# --------------------------------------------------------------------------- speed fit


@pytest.mark.parametrize("view", list(VIEWS))
def test_fit_recovers_launch_speed(view):
    spose = pose_at(VIEWS[view])
    rng = np.random.default_rng(7)
    p_wall = np.array([1.3, 1.4, 0.0])
    p0 = np.array([0.8, 1.0, 4.6])
    T = 0.26
    v_wall = (p_wall - p0 - np.array([0, 0.5 * G * T ** 2, 0])) / T
    t_thud = 10.0
    obs = []
    for i in range(int(T * 60)):
        t = t_thud - T + i / 60.0
        tau = t_thud - t
        p = p_wall - v_wall * tau - np.array([0, 0.5 * G * tau ** 2, 0])
        uv = spose.project(p.reshape(1, 3))[0] + rng.normal(0, 1.0, 2)
        obs.append((t, uv[0], uv[1]))
    v_fit, rms, cond, _, t_ref = fit_velocity(obs, spose, p_wall, t_thud + 0.012)  # start 12 ms off
    assert abs(t_ref - t_thud) < 0.004, t_ref - t_thud
    launch, avg, D, p0_fit = speeds_from_fit(v_fit, T, p_wall)
    true_D = np.linalg.norm(p_wall - p0)
    assert rms < 3.0, rms
    assert abs(avg - true_D / T) / (true_D / T) < 0.05, (view, avg, true_D / T, cond)
    assert np.linalg.norm(p0_fit - p0) < 0.25, (view, p0_fit)


def test_geometric_and_propagation():
    p_wall = np.array([1.0, 1.5, 0.0])
    launch, avg, D, p0 = geometric_speed(np.array([1.0, 4.7]), p_wall, 0.25, 0.7)
    assert abs(D - np.linalg.norm(p_wall - np.array([1.0, 1.0, 4.0]))) < 1e-9
    assert launch > avg > 0
    mic = np.array([1.0, 1.7, 6.5])
    corr = propagation_correction(mic, p0, p_wall)
    assert 0.005 < corr < 0.02  # wall farther from the mic than the racket


# --------------------------------------------------------------------------- forehand / backhand


def skeleton_world(z: float, kind: str, phase: str, sign: float = 1.0) -> np.ndarray:
    """3D keypoints (17x3) of a player at depth z facing the wall. `sign` mirrors for left-handers."""
    k = np.zeros((17, 3))
    hipw, shw = 0.16, 0.20
    yhip, ysh, yhead = 0.95, 1.42, 1.65
    turn = 0.0
    if phase == "loaded":
        turn = 0.18 if kind == FOREHAND else -0.18  # forehand: left shoulder toward the wall
    k[L_HIP] = [-hipw, yhip, z + 0.3 * turn]
    k[R_HIP] = [hipw, yhip, z - 0.3 * turn]
    k[L_SHOULDER] = [-shw, ysh, z - turn]
    k[R_SHOULDER] = [shw, ysh, z + turn]
    k[NOSE] = [0, yhead, z - 0.05]
    k[L_EYE] = [-0.03, yhead + 0.02, z - 0.05]
    k[R_EYE] = [0.03, yhead + 0.02, z - 0.05]
    k[L_EAR] = [-0.07, yhead, z]
    k[R_EAR] = [0.07, yhead, z]
    k[L_KNEE] = [-hipw, 0.5, z]
    k[R_KNEE] = [hipw, 0.5, z]
    k[L_ANKLE] = [-hipw, 0.05, z]
    k[R_ANKLE] = [hipw, 0.05, z]
    if phase == "contact":
        if kind == FOREHAND:
            k[R_WRIST] = [0.55, 1.0, z - 0.45]
            k[L_WRIST] = [-0.35, 1.1, z - 0.2]
        else:
            k[R_WRIST] = [-0.5, 1.0, z - 0.45]
            k[L_WRIST] = [-0.42, 1.02, z - 0.4]
    elif phase == "back":  # 120 ms before contact, racket still on its own side
        if kind == FOREHAND:
            k[R_WRIST] = [0.85, 0.95, z + 0.1]
            k[L_WRIST] = [-0.3, 1.1, z - 0.2]
        else:
            k[R_WRIST] = [-0.75, 0.95, z + 0.1]
            k[L_WRIST] = [-0.7, 0.97, z + 0.05]
    else:  # loaded
        if kind == FOREHAND:
            k[R_WRIST] = [0.9, 1.0, z + 0.4]
            k[L_WRIST] = [-0.2, 1.2, z - 0.3]
        else:
            k[R_WRIST] = [-0.8, 1.0, z + 0.4]
            k[L_WRIST] = [-0.75, 1.02, z + 0.35]
    k[L_ELBOW] = (k[L_SHOULDER] + k[L_WRIST]) / 2
    k[R_ELBOW] = (k[R_SHOULDER] + k[R_WRIST]) / 2
    k[:, 0] *= sign
    if sign < 0:  # a left-hander is the mirror image: swap left/right labels too
        pairs = [(L_EYE, R_EYE), (L_EAR, R_EAR), (L_SHOULDER, R_SHOULDER), (L_ELBOW, R_ELBOW), (L_WRIST, R_WRIST),
                 (L_HIP, R_HIP), (L_KNEE, R_KNEE), (L_ANKLE, R_ANKLE)]
        for a, b in pairs:
            k[[a, b]] = k[[b, a]]
    return k


def samples_for(spose, kind, z=4.5, sign=1.0, t_pop=5.0):
    out = []
    for t, phase in ((t_pop - 0.40, "loaded"), (t_pop - 0.30, "loaded"), (t_pop - 0.20, "loaded"),
                     (t_pop - 0.12, "back"), (t_pop - 0.06, "back"), (t_pop, "contact"), (t_pop + 0.03, "contact")):
        w = skeleton_world(z, kind, phase, sign)
        px = spose.project(w)
        kps = np.hstack([px, np.ones((17, 1))]).astype(np.float32)
        out.append(KpSample(t, kps, spose))
    ball_w = skeleton_world(z, kind, "contact", sign)[R_WRIST if sign > 0 else L_WRIST] + np.array([0.0, 0.0, -0.3])
    ball_px = tuple(spose.project(ball_w.reshape(1, 3))[0])
    return out, ball_px


@pytest.mark.parametrize("view", list(VIEWS))
@pytest.mark.parametrize("kind", [FOREHAND, BACKHAND])
def test_forehand_backhand_from_every_view(view, kind):
    spose = pose_at(VIEWS[view])
    samples, ball_px = samples_for(spose, kind)
    got, conf, cues = classify_stroke(samples, 5.0, PlayerConfig(handedness="right"), ball_px)
    assert got == kind, (view, kind, got, conf, cues)
    assert conf > 0.3


@pytest.mark.parametrize("kind", [FOREHAND, BACKHAND])
def test_left_hander_mirrored(kind):
    spose = pose_at(VIEWS["behind"])
    samples, ball_px = samples_for(spose, kind, sign=-1.0)
    got, conf, cues = classify_stroke(samples, 5.0, PlayerConfig(handedness="left"), ball_px)
    assert got == kind, (kind, got, conf, cues)


def test_low_confidence_gives_unknown():
    spose = pose_at(VIEWS["behind"])
    samples, ball_px = samples_for(spose, FOREHAND)
    for s in samples:
        s.kps[:, 2] = 0.1  # nothing reliable
    got, conf, cues = classify_stroke(samples, 5.0, PlayerConfig(), ball_px)
    assert got == "unknown"
