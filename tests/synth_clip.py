"""Synthetic session clip: rendered marker wall + flying ball + impact sounds, muxed into a .mov.

Used by the end-to-end test and handy for trying the pipeline without any real footage:
    uv run python -m tests.synth_clip recordings/synthetic.mov --view side_right
"""
from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.io import wavfile

from tests.synth import render_scene, synthetic_camera
from wallball.config import WallConfig
from wallball.scene import look_at
from wallball.strokes import G, SOUND

RATE = 48000
last_door_corners = None
last_proj = None  # world point -> (pixel, depth) of the last rendered clip
VIEWS = {  # camera positions that keep the markers clear of the player, as a real setup would
    "behind": (2.4, 1.7, 6.8),
    "side_right": (5.5, 1.5, 4.0),
    "side_left": (-3.5, 1.4, 3.5),
    "ground": (2.6, 0.15, 6.5),
    "diagonal": (4.5, 1.6, 6.0),
}


@dataclass
class SynthStroke:
    t_pop: float
    p0: np.ndarray
    p_wall: np.ndarray
    T: float
    T_back: float
    p_end: np.ndarray

    @property
    def t_thud(self) -> float:
        return self.t_pop + self.T

    @property
    def launch_speed(self) -> float:
        v0 = (self.p_wall - self.p0 - np.array([0, -0.5 * G * self.T ** 2, 0])) / self.T
        return float(np.linalg.norm(v0))

    @property
    def avg_speed(self) -> float:
        return float(np.linalg.norm(self.p_wall - self.p0) / self.T)


def default_strokes(n: int = 6, z_player: float = 4.6) -> list[SynthStroke]:
    rng = np.random.default_rng(11)
    out = []
    t = 1.2
    for i in range(n):
        p0 = np.array([0.9 + 0.3 * rng.uniform(-1, 1), 1.0, z_player - 0.4])
        p_wall = np.array([0.4 + 1.2 * rng.uniform(0, 1), 1.0 + 0.9 * rng.uniform(0, 1), 0.0])
        T = float(rng.uniform(0.22, 0.32))
        T_back = float(rng.uniform(0.35, 0.5))
        p_end = np.array([p0[0] + 0.3 * rng.uniform(-1, 1), 1.2, z_player - 0.6])
        out.append(SynthStroke(t, p0, p_wall, T, T_back, p_end))
        t += T + T_back + 0.7
    return out


def ball_position(strokes: list[SynthStroke], t: float) -> np.ndarray | None:
    for s in strokes:
        if s.t_pop <= t <= s.t_thud:
            tau = t - s.t_pop
            v0 = (s.p_wall - s.p0 - np.array([0, -0.5 * G * s.T ** 2, 0])) / s.T
            return s.p0 + v0 * tau + np.array([0, -0.5 * G * tau ** 2, 0])
        if s.t_thud < t <= s.t_thud + s.T_back:
            tau = t - s.t_thud
            v0 = (s.p_end - s.p_wall - np.array([0, -0.5 * G * s.T_back ** 2, 0])) / s.T_back
            return s.p_wall + v0 * tau + np.array([0, -0.5 * G * tau ** 2, 0])
    return None


def burst(kind: str, rng, dur_s: float = 0.08) -> np.ndarray:
    n = int(dur_s * RATE)
    noise = rng.standard_normal(n).astype(np.float32)
    spec = np.fft.rfft(noise)
    f = np.fft.rfftfreq(n, 1.0 / RATE)
    if kind == "pop":
        shape, tau = 1.0 / (1.0 + np.exp(-(f - 3500) / 500)), 0.012
    else:
        shape, tau = 1.0 / (1.0 + np.exp((f - 900) / 200)), 0.03
    sig = np.fft.irfft(spec * shape, n).astype(np.float32)
    sig *= np.exp(-np.arange(n) / (tau * RATE)).astype(np.float32)
    return sig / (np.abs(sig).max() + 1e-9)


DOOR = dict(width_m=0.9, height_m=2.05, bottom_m=0.0, x0_m=0.6)


def make_clip(out_path: str | Path, view: str = "behind", seconds: float | None = None, width: int = 1920,
              height: int = 1440, fps: float = 60.0, strokes: list[SynthStroke] | None = None,
              player_rect: bool = True, crf: int = 18, markers: bool = True, door: bool = False,
              texture: bool = True) -> list[SynthStroke]:
    """Render the session clip. `markers=False, door=True` draws a door-frame rectangle instead of ArUco
    markers (the no-printer setup); its projected corners are returned via `last_door_corners`."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wall = WallConfig.default()
    cam = synthetic_camera(width, height, f=1370.0 * width / 1920)
    centre = VIEWS[view]
    strokes = strokes or default_strokes()
    seconds = seconds or (strokes[-1].t_thud + strokes[-1].T_back + 1.0)
    bg, R, t = render_scene(wall, cam, centre, background=175, marker_ids=(None if markers else []))
    K = cam.K
    if texture:  # paint marks, dirt and skirting mapped onto the wall plane (X -1..4 m, Y 0..3 m)
        from tests.test_rect_scene import wall_texture

        tex = cv2.subtract(wall_texture(), 15)
        wc = np.array([[-1.0, 3.0, 0], [4.0, 3.0, 0], [4.0, 0.0, 0], [-1.0, 0.0, 0]])
        dst = np.array([[K[0, 0] * (R @ p + t)[0] / (R @ p + t)[2] + K[0, 2], K[1, 1] * (R @ p + t)[1] / (R @ p + t)[2] + K[1, 2]] for p in wc], np.float32)
        src = np.array([[0, 0], [tex.shape[1], 0], [tex.shape[1], tex.shape[0]], [0, tex.shape[0]]], np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(tex, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=175)
        keep = bg < 100  # marker pixels stay as rendered
        bg = np.where(keep, bg, warped).astype(np.uint8)
    bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    global last_door_corners, last_proj
    last_door_corners = None
    if door:
        from wallball.config import SceneRect

        rect = SceneRect(**DOOR)
        cw = rect.corners_world()
        px = np.array([[K[0, 0] * (R @ p + t)[0] / (R @ p + t)[2] + K[0, 2], K[1, 1] * (R @ p + t)[1] / (R @ p + t)[2] + K[1, 2]] for p in cw])
        cv2.polylines(bg, [np.round(px).astype(np.int32).reshape(-1, 1, 2)], True, (40, 40, 40), 9)
        last_door_corners = px

    def proj(p):
        c = R @ p + t
        return (K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2]), float(c[2])

    last_proj = proj

    if player_rect:  # a static dark block where the player stands (not a person: pose stays off in tests)
        z = strokes[0].p0[2] + 0.4
        a, _ = proj(np.array([0.75, 0.0, z]))
        b, _ = proj(np.array([1.25, 1.7, z]))
        cv2.rectangle(bg, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (70, 60, 60), -1)
    tmp_video = out_path.with_suffix(".tmp.mp4")
    enc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", str(crf), "-pix_fmt", "yuv420p", str(tmp_video)], stdin=subprocess.PIPE)
    assert enc.stdin is not None
    n_frames = int(seconds * fps)
    rng = np.random.default_rng(3)
    for i in range(n_frames):
        frame = bg.copy()
        p = ball_position(strokes, i / fps)
        if p is not None:
            (u, v), depth = proj(p)
            r = max(2, int(round(K[0, 0] * 0.0335 / depth)))
            cv2.circle(frame, (int(round(u)), int(round(v))), r, (30, 235, 245), -1)
        frame = cv2.add(frame, rng.integers(0, 4, frame.shape, dtype=np.uint8))
        enc.stdin.write(frame.tobytes())
    enc.stdin.close()
    enc.wait()
    # audio with propagation delays from the phone position
    C = np.array(centre)
    audio = rng.standard_normal(int(seconds * RATE)).astype(np.float32) * 0.002
    for s in strokes:
        for kind, tt, p in (("pop", s.t_pop, s.p0), ("thud", s.t_thud, s.p_wall)):
            delay = float(np.linalg.norm(C - p)) / SOUND
            i0 = int((tt + delay) * RATE)
            b = burst(kind, rng) * (0.8 if kind == "pop" else 0.4)
            b = b[: max(0, audio.size - i0)]
            audio[i0:i0 + b.size] += b
    tmp_wav = out_path.with_suffix(".tmp.wav")
    wavfile.write(tmp_wav, RATE, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(tmp_video), "-i", str(tmp_wav),
                    "-c:v", "copy", "-c:a", "pcm_s16le", "-shortest", str(out_path)], check=True)
    tmp_video.unlink(missing_ok=True)
    tmp_wav.unlink(missing_ok=True)
    return strokes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--view", default="behind", choices=list(VIEWS))
    args = ap.parse_args()
    st = make_clip(args.out, args.view)
    print(f"wrote {args.out}: {len(st)} strokes, launch speeds", [round(s.launch_speed * 3.6) for s in st])
