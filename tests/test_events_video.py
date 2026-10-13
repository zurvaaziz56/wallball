import numpy as np
import pytest

from tests.synth import render_scene, synthetic_camera
from wallball.audio import Onset
from wallball.ball import Detection, LegTracker
from wallball.config import WallConfig
from wallball.events import FLOOR_BOUNCE, WALL_HIT, VideoEventDetector
from wallball.scene import SceneTracker
from wallball.strokes import G

WALL = WallConfig.default()
CAM = synthetic_camera()
FPS = 60.0


def pose_at(centre):
    img, _, _ = render_scene(WALL, CAM, centre)
    p = SceneTracker(WALL, CAM).update(img, 0.0)
    assert p is not None
    return p


def flight(p0, p1, T, t0):
    """Ballistic 3D points from p0 to p1 in T seconds, sampled at 60 fps: list of (t, xyz)."""
    v0 = (np.array(p1) - np.array(p0) - np.array([0, -0.5 * G * T ** 2, 0])) / T
    out = []
    n = int(T * FPS)
    for i in range(n + 1):
        t = i / FPS
        p = np.array(p0) + v0 * t + np.array([0, -0.5 * G * t ** 2, 0])
        out.append((t0 + t, p))
    return out


def radius_px(spose, p, ball_r=0.0335):
    d = np.linalg.norm(p - spose.C)
    return spose.K[0, 0] * ball_r / d


def run_track(spose, pts, onsets=(), player_box=None):
    tracker = LegTracker()
    det = VideoEventDetector(WALL)
    events = []
    for i, (t, p) in enumerate(pts):
        uv = spose.project(p.reshape(1, 3))[0]
        r = radius_px(spose, p)
        d = Detection(t, i, uv[0], uv[1], r, 2 * r, np.pi * r * r, 1.0)
        tracker.update([d], t, i)
        ev = det.update(tracker, spose, list(onsets), player_box)
        if ev is not None:
            events.append(ev)
    return events


VIEWS = [(1.0, 1.7, 6.5), (5.5, 1.5, 4.0), (-3.5, 1.4, 3.5), (0.4, 0.15, 5.5), (4.5, 1.6, 6.0)]


@pytest.mark.parametrize("centre", VIEWS)
def test_wall_hit_detected_and_placed(centre):
    spose = pose_at(centre)
    p_wall = np.array([1.2, 1.3, 0.0])
    out = flight([0.8, 1.0, 4.5], p_wall, 0.25, 1.0)
    back = flight(p_wall, [0.6, 0.9, 4.0], 0.3, 1.25)[1:]
    events = run_track(spose, out + back)
    wall = [e for e in events if e.kind == WALL_HIT]
    assert len(wall) == 1, [(e.kind, e.cues) for e in events]
    e = wall[0]
    assert abs(e.t - 1.25) < 0.02
    assert abs(e.wall_xy[0] - 1.2) < 0.08 and abs(e.wall_xy[1] - 1.3) < 0.08, e.wall_xy


@pytest.mark.parametrize("centre", VIEWS[:3])
def test_floor_bounce_not_a_wall_hit(centre):
    spose = pose_at(centre)
    down = flight([0.9, 1.0, 4.5], [1.0, 0.03, 2.5], 0.22, 1.0)
    up = flight([1.0, 0.03, 2.5], [1.1, 1.2, 0.0], 0.25, 1.22)[1:]
    events = run_track(spose, down + up)
    kinds = [e.kind for e in events]
    assert WALL_HIT not in kinds[:1], events
    assert kinds and kinds[0] == FLOOR_BOUNCE, [(e.kind, e.cues) for e in events]


def test_audio_thud_breaks_a_tie():
    spose = pose_at((1.0, 1.7, 6.5))
    p_wall = np.array([1.2, 1.3, 0.0])
    out = flight([0.8, 1.0, 4.5], p_wall, 0.25, 1.0)
    back = flight(p_wall, [0.6, 0.9, 4.0], 0.3, 1.25)[1:]
    thud = Onset(1.252, 10.0, 600.0, (0.8, 0.1, 0.1), "thud", {"pop": 0.05, "thud": 0.9, "floor": 0.05})
    events = run_track(spose, out + back, onsets=[thud])
    assert events and events[0].kind == WALL_HIT and events[0].cues["thud"] > 0.8
