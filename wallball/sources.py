"""Frame sources: live camera (ffmpeg/avfoundation or OpenCV) and recorded files (PyAV).

Live frames carry monotonic-clock timestamps; file frames carry media time from the
container (iPhone clips are variable frame rate, so never derive time from frame index).
"""
from __future__ import annotations

import json
import queue
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


@dataclass
class Frame:
    image: np.ndarray  # BGR uint8, writable
    t: float  # seconds: monotonic clock (live) or media time (file)
    index: int


class FrameSource(Protocol):
    fps: float
    width: int
    height: int

    def __iter__(self) -> Iterator[Frame]: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- device discovery


def list_avfoundation_devices() -> tuple[list[str], list[str]]:
    """Names of AVFoundation video and audio devices, in ffmpeg index order."""
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True,
    )
    video: list[str] = []
    audio: list[str] = []
    section: list[str] | None = None
    for line in proc.stderr.splitlines():
        if "AVFoundation video devices" in line:
            section = video
            continue
        if "AVFoundation audio devices" in line:
            section = audio
            continue
        m = re.search(r"\]\s*\[(\d+)\]\s+(.*)$", line)
        if section is not None and m:
            section.append(m.group(2).strip())
    return video, audio


def supported_modes(device: str) -> list[str]:
    """Ask ffmpeg which (size@fps) modes a device offers by requesting an impossible frame rate."""
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-f", "avfoundation", "-framerate", "1234", "-i", f"{device}:none",
         "-t", "0.1", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    modes: list[str] = []
    for line in proc.stderr.splitlines():
        m = re.search(r"(\d+x\d+@\[[^\]]+\]fps)", line)
        if m and m.group(1) not in modes:
            modes.append(m.group(1))
    return modes


def resolve_device(name_or_index: str | int, devices: list[str]) -> str:
    """Case-insensitive substring match against device names; digits are taken as an index."""
    if isinstance(name_or_index, int) or str(name_or_index).isdigit():
        idx = int(name_or_index)
        if 0 <= idx < len(devices):
            return devices[idx]
        raise ValueError(f"device index {idx} out of range; found {devices}")
    needle = str(name_or_index).lower()
    matches = [d for d in devices if needle in d.lower()]
    # the iPhone exposes a "Desk View" camera too; only pick it when asked for explicitly
    preferred = [d for d in matches if "desk view" not in d.lower()] if "desk view" not in needle else matches
    if preferred:
        return preferred[0]
    if matches:
        return matches[0]
    raise ValueError(f"no device matching {name_or_index!r}; found {devices}")


# --------------------------------------------------------------------------- live: ffmpeg


class FFmpegLiveSource:
    """Reads raw BGR frames from an ffmpeg process capturing an AVFoundation device.

    Selecting the device by exact name and requesting size/fps explicitly is the most
    reliable way to get the iPhone's 60 fps mode on macOS; a wrong mode fails fast with
    ffmpeg's list of supported modes in the error message.
    """

    def __init__(self, device: str, width: int, height: int, fps: float, *, pixel_format: str | None = None,
                 queue_size: int = 4):
        self.device = device
        self.width, self.height, self.fps = width, height, float(fps)
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "avfoundation",
               "-framerate", f"{fps:g}", "-video_size", f"{width}x{height}"]
        if pixel_format:
            cmd += ["-pixel_format", pixel_format]
        cmd += ["-i", f"{device}:none", "-an", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.q: queue.Queue[Frame | None] = queue.Queue(maxsize=queue_size)
        self.dropped = 0
        self.received = 0
        self.measured_fps = 0.0
        self._stderr: list[str] = []
        self._stop = threading.Event()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        threading.Thread(target=self._reader, daemon=True).start()

    def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        for raw in self.proc.stderr:
            self._stderr.append(raw.decode(errors="replace").rstrip())

    def _read_exact(self, n: int) -> np.ndarray | None:
        assert self.proc.stdout is not None
        arr = np.empty(n, dtype=np.uint8)
        view = memoryview(arr)
        off = 0
        while off < n:
            got = self.proc.stdout.readinto(view[off:])
            if not got:
                return None
            off += got
        return arr

    def _reader(self) -> None:
        n = self.width * self.height * 3
        times: deque[float] = deque(maxlen=120)
        i = 0
        while not self._stop.is_set():
            buf = self._read_exact(n)
            if buf is None:
                break
            t = time.monotonic()
            times.append(t)
            if len(times) >= 2:
                self.measured_fps = (len(times) - 1) / (times[-1] - times[0])
            frame = Frame(buf.reshape(self.height, self.width, 3), t, i)
            i += 1
            self.received += 1
            try:
                self.q.put_nowait(frame)
            except queue.Full:
                try:
                    self.q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass
                self.q.put_nowait(frame)
        self._stop.set()
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass

    def error_text(self) -> str:
        return "\n".join(self._stderr[-20:])

    def __iter__(self) -> Iterator[Frame]:
        while True:
            try:
                frame = self.q.get(timeout=3.0)
            except queue.Empty:
                if self.proc.poll() is not None or self._stop.is_set():
                    raise RuntimeError(
                        "ffmpeg capture ended.\n" + self.error_text()
                        + "\nRun `wallball devices` to see the device names and supported modes."
                    )
                continue
            if frame is None:
                if self.proc.poll() not in (0, None):
                    raise RuntimeError("ffmpeg capture failed.\n" + self.error_text())
                return
            yield frame

    def close(self) -> None:
        self._stop.set()
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# --------------------------------------------------------------------------- live: OpenCV


class OpenCVLiveSource:
    """Fallback capture through OpenCV's AVFoundation backend (index based, less control over modes)."""

    def __init__(self, index: int, width: int, height: int, fps: float, *, queue_size: int = 4):
        self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV could not open camera index {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or fps)
        self.q: queue.Queue[Frame | None] = queue.Queue(maxsize=queue_size)
        self.dropped = 0
        self.measured_fps = 0.0
        self._stop = threading.Event()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        i = 0
        times: deque[float] = deque(maxlen=120)
        while not self._stop.is_set():
            ok, img = self.cap.read()
            if not ok:
                break
            t = time.monotonic()
            times.append(t)
            if len(times) >= 2:
                self.measured_fps = (len(times) - 1) / (times[-1] - times[0])
            frame = Frame(img, t, i)
            i += 1
            try:
                self.q.put_nowait(frame)
            except queue.Full:
                try:
                    self.q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass
                self.q.put_nowait(frame)
        self._stop.set()
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass

    def __iter__(self) -> Iterator[Frame]:
        while True:
            frame = self.q.get()
            if frame is None:
                return
            yield frame

    def close(self) -> None:
        self._stop.set()
        self.cap.release()


# --------------------------------------------------------------------------- files


def probe_rotation(path: str | Path) -> int:
    """Clockwise display rotation (0/90/180/270) stored in the container, via ffprobe."""
    try:
        proc = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream_side_data=rotation:stream_tags=rotate", "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        )
        info = json.loads(proc.stdout or "{}")
        streams = info.get("streams") or []
        if not streams:
            return 0
        s = streams[0]
        for sd in s.get("side_data_list", []) or []:
            if "rotation" in sd:
                return int(round(-float(sd["rotation"]))) % 360  # displaymatrix: counter-clockwise positive
        tag = (s.get("tags") or {}).get("rotate")
        if tag is not None:
            return int(round(float(tag))) % 360
    except Exception:
        return 0
    return 0


def probe_duration(path: str | Path) -> float:
    try:
        proc = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(json.loads(proc.stdout)["format"]["duration"])
    except Exception:
        return float("nan")


_ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


class FileSource:
    """Decodes a recorded clip with PyAV, yielding frames with their real presentation times."""

    def __init__(self, path: str | Path, *, realtime: bool = False, start_s: float = 0.0,
                 end_s: float | None = None, rotate: int | None = None):
        import av

        self.path = Path(path)
        self.container = av.open(str(self.path))
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        rate = self.stream.average_rate or self.stream.guessed_rate
        self.fps = float(rate) if rate else 30.0
        self.rotation = probe_rotation(self.path) if rotate is None else rotate % 360
        w, h = self.stream.codec_context.width, self.stream.codec_context.height
        if self.rotation in (90, 270):
            w, h = h, w
        self.width, self.height = w, h
        self.realtime = realtime
        self.start_s = start_s
        self.end_s = end_s
        self.duration_s = probe_duration(self.path)

    def __iter__(self) -> Iterator[Frame]:
        tb = float(self.stream.time_base)
        wall0: float | None = None
        t0: float | None = None
        for i, frame in enumerate(self.container.decode(self.stream)):
            t = frame.pts * tb if frame.pts is not None else i / self.fps
            if t < self.start_s:
                continue
            if self.end_s is not None and t > self.end_s:
                break
            img = frame.to_ndarray(format="bgr24")
            if self.rotation in _ROTATE:
                img = cv2.rotate(img, _ROTATE[self.rotation])
            if self.realtime:
                now = time.monotonic()
                if wall0 is None:
                    wall0, t0 = now, t
                else:
                    lag = (t - t0) - (now - wall0)
                    if lag > 0:
                        time.sleep(lag)
            yield Frame(np.ascontiguousarray(img), float(t), i)

    def close(self) -> None:
        self.container.close()


def extract_audio(path: str | Path, rate: int = 48000) -> np.ndarray:
    """Mono float32 samples of a clip's audio track at `rate` Hz (empty array if it has none)."""
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(rate),
         "-f", "f32le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


# --------------------------------------------------------------------------- raw session recording


def start_recording(video_device: str, audio_device: str | None, width: int, height: int, fps: float,
                    out_path: str | Path, *, bitrate_mbps: int = 20) -> subprocess.Popen:
    """Record video (+ audio) from the iPhone straight to disk with ffmpeg; returns the process.

    Timestamps stay exact inside the container, which is what replay needs. Send 'q' to stdin
    (or terminate) to stop.
    """
    inp = f"{video_device}:{audio_device}" if audio_device else f"{video_device}:none"
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "warning", "-f", "avfoundation",
           "-framerate", f"{fps:g}", "-video_size", f"{width}x{height}", "-i", inp,
           "-fps_mode", "passthrough", "-video_track_timescale", "60000",
           "-c:v", "h264_videotoolbox", "-b:v", f"{bitrate_mbps}M", "-pix_fmt", "yuv420p"]
    if audio_device:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    cmd += ["-movflags", "+faststart", "-y", str(out_path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def stop_recording(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            assert proc.stdin is not None
            proc.stdin.write(b"q")
            proc.stdin.flush()
            proc.wait(timeout=5)
        except Exception:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
