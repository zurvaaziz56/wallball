"""Streaming pipeline: frames -> scene pose -> player pose -> ball -> events -> strokes -> dashboard.

Threads (so the ball detector sees every frame at 60 fps):
  analysis  iterates the source: markers, audio, ball, events, strokes            (every frame)
  pose      runs YOLO11-pose on the newest frame it can get                       (about 30 Hz)
  main      composes the dashboard for the newest analysed frame, shows/writes it (about 30 Hz)
The same loop serves live capture and file replay.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .audio import LiveAudio, Onset, onsets_from_samples
from .ball import ClassicBallDetector, LegTracker
from .config import AppConfig, CameraConfig, PlayerConfig, SceneRect, WallConfig
from .events import AudioStroke, ImpactSequencer, VideoEventDetector
from .pose import KeypointSmoother, PoseEstimator
from .render import DashboardState, Renderer, draw_impacts, draw_scene, draw_skeleton, draw_trail
from .scene import MOVING, STATIC, RectSceneTracker, ScenePose, SceneTracker
from .session import StrokeStore
from .sources import extract_audio

BALL_RADIUS_M = 0.0335


@dataclass
class RenderItem:
    image: np.ndarray
    t: float
    spose: ScenePose | None
    kps: np.ndarray | None
    trail: list[tuple[float, float]]
    trail_r: float
    impacts: list[tuple[int, float, float]]
    latest_idx: int | None


class Pipeline:
    def __init__(self, app: AppConfig, wall: WallConfig, cam: CameraConfig | None, player: PlayerConfig, *,
                 root: str | Path = ".", enable_pose: bool = True, enable_ball: bool = True, display: bool = True,
                 out_video: str | Path | None = None, title: str = "wallball", stroke_delay_s: float = 0.15,
                 display_hz: float = 30.0, scene_rect: SceneRect | None = None):
        self.scene_rect = scene_rect
        if scene_rect is not None:
            wall = scene_rect.wall_config()  # the measured rectangle is the placement target, no markers
        self.app, self.wall, self.cam, self.player = app, wall, cam, player
        self.root = Path(root)
        self.display = display
        self.display_hz = display_hz
        self.out_video = Path(out_video) if out_video else None
        self.title = title
        self.renderer = Renderer(app.display_width)
        self.state = DashboardState()
        self.scene: SceneTracker | None = None
        self.pose_est: PoseEstimator | None = None
        if enable_pose:
            self.pose_est = PoseEstimator(self.root / app.pose_model, app.pose_fallback, imgsz=app.pose_imgsz)
        self.smoother = KeypointSmoother()
        self.last_kps: np.ndarray | None = None
        self.last_box: np.ndarray | None = None
        self.calibrated = cam is not None
        # audio
        self.sequencer = ImpactSequencer()
        self.live_audio: LiveAudio | None = None
        self.file_onsets: list[Onset] = []
        self._file_onset_i = 0
        self.onset_log: deque[Onset] = deque(maxlen=60)
        # ball + events + strokes
        self.enable_ball = enable_ball
        self.ball_det = ClassicBallDetector()
        self.tracker = LegTracker()
        self.video_events = VideoEventDetector(wall)
        self.store = StrokeStore(wall, player, self.sequencer)
        self.pending: list[AudioStroke] = []
        self.stroke_delay = stroke_delay_s
        self.prev_H: deque[np.ndarray] = deque(maxlen=2)
        self.n_frames = 0
        # threading
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pose_frame: tuple[np.ndarray, float, int] | None = None
        self._pose_event = threading.Event()
        self._render_item: RenderItem | None = None
        self._render_event = threading.Event()
        self.writer: cv2.VideoWriter | None = None
        self.writer_q: queue.Queue = queue.Queue(maxsize=90)
        self.writer_thread: threading.Thread | None = None
        self.error: BaseException | None = None
        self.proc_times: deque[float] = deque(maxlen=60)

    # ---- audio ------------------------------------------------------------------
    def attach_audio_file(self, path) -> int:
        samples = extract_audio(path, self.app.audio_rate)
        self.file_onsets = onsets_from_samples(samples, self.app.audio_rate) if samples.size else []
        self._file_onset_i = 0
        return len(self.file_onsets)

    def attach_live_audio(self) -> None:
        try:
            self.live_audio = LiveAudio(self.app.audio_device, self.app.audio_rate)
            self.live_audio.start()
            print(f"audio: listening on input device {self.live_audio.device}", flush=True)
        except Exception as e:  # no mic: keep running without audio
            print(f"audio disabled: {e}", flush=True)
            self.live_audio = None

    def _pending_onsets(self, t: float) -> list[Onset]:
        if self.live_audio is not None:
            return self.live_audio.poll()
        out = []
        while self._file_onset_i < len(self.file_onsets) and self.file_onsets[self._file_onset_i].t <= t:
            out.append(self.file_onsets[self._file_onset_i])
            self._file_onset_i += 1
        return out

    @property
    def audio_on(self) -> bool:
        return self.live_audio is not None or bool(self.file_onsets)

    # ---- scene ------------------------------------------------------------------
    def _ensure_scene(self, w: int, h: int) -> None:
        if self.scene is not None:
            return
        cam = self.cam.scaled(w, h) if self.cam is not None else CameraConfig.nominal(w, h)
        if self.cam is None:
            print("no camera calibration found: using a nominal 70 deg lens (run `wallball lens photo.jpg`)", flush=True)
        if self.scene_rect is not None:
            self.scene = RectSceneTracker(self.scene_rect, cam, self.scene_rect.corners_for(w, h))
            if self.scene.focal_refined:
                print(f"focal length refined from the rectangle: {self.scene.cam.fx:.0f} px (nominal {cam.fx:.0f})", flush=True)
                self.calibrated = True
        else:
            self.scene = SceneTracker(self.wall, cam)

    def _radius_range(self, spose) -> tuple[float, float]:
        f = float(spose.K[0, 0])
        far = max(1.0, spose.distance_to_wall_m + 0.5)
        return (max(1.5, f * BALL_RADIUS_M / far), min(80.0, f * BALL_RADIUS_M / 0.8))

    def _wall_roi(self, spose, shape) -> np.ndarray:
        rect = self.wall.practice_rect
        corners = np.array([[rect.x0_m - 1.0, 0.0, 0.0], [rect.x0_m - 1.0, 3.5, 0.0],
                            [rect.x1_m + 1.0, 3.5, 0.0], [rect.x1_m + 1.0, 0.0, 0.0]])
        px = spose.project(corners)
        mask = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(px).astype(np.int32)], 255)
        return mask

    # ---- strokes ----------------------------------------------------------------
    def _flush_strokes(self, t: float, force: bool = False) -> None:
        keep = []
        lo, hi = self.sequencer.pop_thud_window
        for a in self.pending:
            due = a.t_pop + max(hi, self.sequencer.max_open_s if a.t_thud is None else 0.0) + self.stroke_delay
            if force or t >= due:
                self.store.on_audio_stroke(a)
            else:
                keep.append(a)
        self.pending = keep
        self.store.update_dashboard(self.state)

    # ---- pose thread --------------------------------------------------------------
    def _pose_loop(self) -> None:
        assert self.pose_est is not None
        while not self._stop.is_set():
            if not self._pose_event.wait(timeout=0.2):
                continue
            with self._lock:
                item = self._pose_frame
                self._pose_frame = None
                self._pose_event.clear()
            if item is None:
                continue
            img, t, index = item
            try:
                pr = self.pose_est.infer(img, t, index)
            except Exception as e:  # keep the pipeline alive
                print(f"pose error: {e}", flush=True)
                continue
            if pr is not None:
                kps = self.smoother.update(pr.kps)
                with self._lock:
                    self.last_kps, self.last_box = kps, pr.box
                spose = self.scene.last if self.scene is not None else None
                self.store.add_pose(t, kps, spose if (spose is not None and t - spose.t < 0.5) else None)

    # ---- analysis (per frame) -------------------------------------------------------
    def process(self, frame) -> RenderItem:
        img = frame.image
        self.n_frames += 1
        self._ensure_scene(img.shape[1], img.shape[0])
        assert self.scene is not None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        spose = self.scene.update(gray, frame.t)
        if spose is not None:
            self.store.add_scene(frame.t, spose)
        # audio
        for onset in self._pending_onsets(frame.t):
            self.onset_log.append(onset)
            self.pending.extend(self.sequencer.push(onset))
        # pose: hand the newest frame to the pose thread (a copy, so overlays never leak into inference)
        if self.pose_est is not None and not self._pose_event.is_set():
            with self._lock:
                self._pose_frame = (img.copy(), frame.t, frame.index)
                self._pose_event.set()
        with self._lock:
            kps, box = self.last_kps, self.last_box
        # ball
        if self.enable_ball and spose is None:
            self.ball_det.prev.clear()
        if self.enable_ball and spose is not None:
            prev_override = None
            roi = None
            if spose.motion == MOVING and len(self.prev_H) == 2 and len(self.ball_det.prev) == 2:
                H_cur = spose.H_wall
                prev_override = [cv2.warpPerspective(g, H_cur @ np.linalg.inv(Hp), (img.shape[1], img.shape[0]))
                                 for g, Hp in zip(self.ball_det.prev, self.prev_H)]
                roi = self._wall_roi(spose, gray.shape)
            mask_boxes = [tuple(box)] if box is not None else None
            static = spose.motion == STATIC and spose.stale_s < 0.5
            dets = self.ball_det.detect(img, frame.t, frame.index, mask_boxes=mask_boxes,
                                        predicted=self.tracker.predict(frame.t), radius_range=self._radius_range(spose),
                                        prev_override=prev_override, roi_mask=roi, static=static)
            self.prev_H.append(spose.H_wall.copy())
            pt = self.tracker.update(dets, frame.t, frame.index)
            if pt is not None:
                self.store.add_ball(frame.t, pt.x, pt.y, pt.r, len(self.tracker.legs) - 1)
            ev = self.video_events.update(self.tracker, spose, list(self.onset_log), box)
            if ev is not None:
                self.store.add_event(ev)
            if len(self.tracker.legs) > 40:
                self.tracker.legs = self.tracker.legs[-20:]
        self._flush_strokes(frame.t)
        # dashboard text
        if spose is not None:
            self.state.badge = spose.badge + ("" if self.calibrated else "  [nominal lens]")
        else:
            self.state.badge = f"NO SCENE: need >= {self.wall.min_markers} markers spanning {self.wall.min_span_m:.0f} m"
        parts = [f"audio {len(self.onset_log)} onsets" if self.audio_on else "audio off",
                 f"ball legs {len(self.tracker.legs)}" if self.enable_ball else "ball off"]
        self.state.status = "  |  ".join(parts)
        impacts = [(s.idx, s.wall_xy[0], s.wall_xy[1]) for s in self.store.strokes[-12:] if s.wall_xy is not None]
        trail = self.tracker.trail_xy() if self.enable_ball else []
        trail_r = self.tracker.trail[-1].r * 2.5 if (self.enable_ball and self.tracker.trail) else 0.0
        return RenderItem(img, frame.t, spose, kps, trail, trail_r, impacts, self.state.latest_idx)

    def compose(self, item: RenderItem) -> np.ndarray:
        """Dashboard canvas for one analysed frame (overlays drawn on the downscaled video)."""

        def overlays(small: np.ndarray, scale: float) -> None:
            if item.spose is not None:
                draw_scene(small, item.spose, self.wall, scale)
                draw_impacts(small, item.spose, item.impacts, item.latest_idx, scale)
            if item.kps is not None:
                draw_skeleton(small, item.kps, scale=scale)
            if item.trail:
                draw_trail(small, item.trail, item.trail_r, scale)

        return self.renderer.compose(item.image, self.state, overlays)

    # ---- threads ----------------------------------------------------------------------
    def _analysis_loop(self, source, max_seconds: float | None, t_start_box: list) -> None:
        n = 0
        last_print = 0.0
        try:
            for frame in source:
                if self._stop.is_set():
                    break
                t0 = time.monotonic()
                if not t_start_box:
                    t_start_box.append(frame.t)
                self.state.clock_s = frame.t - t_start_box[0]
                if max_seconds is not None and self.state.clock_s > max_seconds:
                    break
                self.state.fps_capture = float(getattr(source, "measured_fps", 0.0) or source.fps)
                item = self.process(frame)
                self.proc_times.append(time.monotonic() - t0)
                self.state.fps_proc = 1.0 / max(1e-6, float(np.mean(self.proc_times)))
                n += 1
                with self._lock:
                    self._render_item = item
                self._render_event.set()
                if time.monotonic() - last_print > 2.0:
                    last_print = time.monotonic()
                    dropped = getattr(source, "dropped", 0)
                    print(f"[{self.state.clock_s:7.1f}s] frames {n}  cap {self.state.fps_capture:5.1f} fps  "
                          f"proc {self.state.fps_proc:5.1f} fps  dropped {dropped}  strokes {self.state.strokes}"
                          f"  | {self.state.badge}", flush=True)
        except BaseException as e:  # surfaced by run()
            self.error = e
        finally:
            self.n_processed = n
            self.last_t = self.state.clock_s + (t_start_box[0] if t_start_box else 0.0)
            self._stop.set()
            self._render_event.set()

    def _write_loop(self) -> None:
        assert self.writer is not None
        while True:
            item = self.writer_q.get()
            if item is None:
                return
            self.writer.write(item)

    def run(self, source, max_seconds: float | None = None, session_json: str | Path | None = None) -> DashboardState:
        t_start_box: list = []
        self.n_processed = 0
        self.last_t = 0.0
        threads = [threading.Thread(target=self._analysis_loop, args=(source, max_seconds, t_start_box), daemon=True)]
        if self.pose_est is not None:
            threads.append(threading.Thread(target=self._pose_loop, daemon=True))
        for th in threads:
            th.start()
        render_needed = self.display or self.out_video is not None
        min_period = 1.0 / self.display_hz
        last_render = 0.0
        try:
            while not self._stop.is_set() or self._render_event.is_set():
                if not self._render_event.wait(timeout=0.2):
                    continue
                with self._lock:
                    item = self._render_item
                    self._render_item = None
                    self._render_event.clear()
                if item is None:
                    continue
                if not render_needed:
                    continue
                now = time.monotonic()
                if now - last_render < min_period and not self._stop.is_set():
                    continue  # newer frames keep coming; show at most display_hz
                last_render = now
                canvas = self.compose(item)
                if self.out_video is not None:
                    if self.writer is None:
                        h, w = canvas.shape[:2]
                        fps = min(self.display_hz, float(source.fps))
                        self.writer = cv2.VideoWriter(str(self.out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        self.writer_thread = threading.Thread(target=self._write_loop, daemon=True)
                        self.writer_thread.start()
                    try:
                        self.writer_q.put_nowait(canvas)
                    except queue.Full:
                        pass
                if self.display:
                    cv2.imshow(self.title, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        self._stop.set()
                    elif key == ord("s"):
                        cv2.imwrite(f"screenshot_{int(time.time())}.png", canvas)
        finally:
            self._stop.set()
            source.close()
            for th in threads:
                th.join(timeout=5)
            self.pending.extend(self.sequencer.flush())
            self._flush_strokes(self.last_t, force=True)
            if self.live_audio is not None:
                self.live_audio.stop()
            if self.writer is not None:
                self.writer_q.put(None)
                if self.writer_thread is not None:
                    self.writer_thread.join(timeout=30)
                self.writer.release()
            if self.display:
                cv2.destroyAllWindows()
            if session_json is not None:
                self.store.to_json(session_json, {"frames": self.n_processed, "onsets": len(self.onset_log)})
                print("session written to", session_json, flush=True)
        if self.error is not None:
            raise self.error
        return self.state
