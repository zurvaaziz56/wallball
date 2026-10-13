"""Impact onsets from the microphone: racket "pop", wall "thud", floor bounce.

Per-band spectral flux at 48 kHz with a 512/128 window/hop (2.7 ms resolution), an adaptive
median threshold so reverb tails do not mask quieter impacts, and a spectral-centroid class
score per onset. Works identically on the live stream and on a clip's audio track.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

POP, THUD, FLOOR, UNKNOWN = "pop", "thud", "floor", "?"


@dataclass
class Onset:
    t: float  # seconds (monotonic for live, media time for files)
    strength: float
    centroid_hz: float
    bands: tuple[float, float, float]  # low / mid / high energy share right after the onset
    label: str = UNKNOWN
    scores: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Onset(t={self.t:.3f}, {self.label}, s={self.strength:.2f}, c={self.centroid_hz:.0f}Hz)"


def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


class OnsetDetector:
    """Streaming onset detector. Feed samples with `push`; confirmed onsets come back with a ~10 ms delay."""

    def __init__(self, rate: int = 48000, n_fft: int = 1024, hop: int = 128, *, median_s: float = 0.2,
                 adapt_s: float = 2.0, min_gap_s: float = 0.03, thr_k: float = 6.0, thr_floor: float = 9.0,
                 peak_half_window: int = 3, warmup_s: float = 0.25, mask_tau_s: float = 0.04,
                 bands_hz=((300, 600), (600, 1200), (1200, 2400), (2400, 4800), (4800, 9600), (9600, 19000)),
                 ):
        self.rate, self.n_fft, self.hop = rate, n_fft, hop
        self.window = _hann(n_fft).astype(np.float32)
        self.freqs = np.fft.rfftfreq(n_fft, 1.0 / rate)
        self.band_masks = [(self.freqs >= lo) & (self.freqs < hi) for lo, hi in bands_hz]
        self.band_centres = np.array([np.sqrt(lo * hi) for lo, hi in bands_hz])
        self.median_n = max(3, int(round(median_s * rate / hop)))
        self.adapt_n = max(self.median_n, int(round(adapt_s * rate / hop)))
        self.min_gap_frames = max(1, int(round(min_gap_s * rate / hop)))
        self.warmup_frames = int(round(warmup_s * rate / hop))
        self.mask_tau_frames = mask_tau_s * rate / hop
        self.thr_k, self.thr_floor = thr_k, thr_floor
        self.phw = peak_half_window
        self.buffer = np.zeros(0, dtype=np.float32)
        self.prev_log: np.ndarray | None = None
        self.prev_log2: np.ndarray | None = None
        self.strength: deque[float] = deque(maxlen=self.adapt_n)
        self.frames: deque[tuple[int, float, np.ndarray, np.ndarray]] = deque(maxlen=2 * self.phw + 8)
        self.n_frames = 0
        self.consumed = 0  # samples consumed from the start of the stream
        self.t0: float | None = None  # stream time of sample 0
        self.last_onset_frame = -10 ** 9
        self.last_onset_strength = 0.0
        self.total_onsets = 0

    # ---- timing ------------------------------------------------------------
    def frame_time(self, idx: int) -> float:
        t0 = self.t0 or 0.0
        return t0 + (idx * self.hop + self.n_fft / 2.0) / self.rate

    # ---- features ----------------------------------------------------------
    def _analyse(self, frame: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        power = np.abs(np.fft.rfft(frame * self.window)) ** 2
        energy = np.array([power[m].sum() for m in self.band_masks])
        log = np.log(energy + 1e-7)
        if self.prev_log is None or self.prev_log2 is None:
            self.prev_log2, self.prev_log = self.prev_log, log
            return 0.0, np.zeros(len(self.band_masks)), energy
        inc = np.maximum(0.0, log - self.prev_log2)  # rise over two hops (about 5 ms)
        self.prev_log2, self.prev_log = self.prev_log, log
        return float(inc.sum()), inc, energy

    def _classify(self, delta_e: np.ndarray) -> tuple[float, tuple[float, float, float], str, dict[str, float]]:
        """Class scores from where the energy arrived (linear energy increase per band), so a quiet
        dull thud inside the bright tail of a pop still reads as a thud."""
        total = float(delta_e.sum()) + 1e-12
        centroid = float((self.band_centres * delta_e).sum() / total)
        low = float(delta_e[self.band_centres < 1500].sum() / total)
        high = float(delta_e[self.band_centres >= 4000].sum() / total)
        mid = max(0.0, 1.0 - low - high)
        s_pop = 1.0 / (1.0 + np.exp(-(centroid - 4000.0) / 700.0))
        s_thud = 1.0 / (1.0 + np.exp((centroid - 1500.0) / 400.0))
        s_floor = max(0.0, 1.0 - s_pop - s_thud)
        scores = {POP: float(s_pop), THUD: float(s_thud), FLOOR: float(s_floor)}
        label = max(scores, key=scores.get)
        if scores[label] < 0.45:
            label = UNKNOWN
        return centroid, (low, mid, high), label, scores

    # ---- streaming ---------------------------------------------------------
    def push(self, samples: np.ndarray, t_first: float | None = None) -> list[Onset]:
        """Append samples (mono float32). `t_first` is the stream time of samples[0]; needed once."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self.t0 is None:
            self.t0 = (t_first if t_first is not None else 0.0) - self.consumed / self.rate
        self.buffer = np.concatenate([self.buffer, samples]) if self.buffer.size else samples.copy()
        out: list[Onset] = []
        while self.buffer.size >= self.n_fft:
            frame = self.buffer[: self.n_fft]
            flux, inc, energy = self._analyse(frame)
            self.buffer = self.buffer[self.hop:]
            self.consumed += self.hop
            idx = self.n_frames
            self.n_frames += 1
            # adaptive threshold from the recent history (median + k * MAD, with a floor)
            self.strength.append(flux)
            hist = np.fromiter(self.strength, dtype=np.float64)
            med = float(np.median(hist[-self.median_n:]))
            mad = float(np.median(np.abs(hist - np.median(hist)))) if hist.size > 8 else 0.0
            thr = max(self.thr_floor, med + self.thr_k * (mad + 0.05))
            self.frames.append((idx, flux, inc, energy))
            # decide the frame `phw` frames back, now that its right-hand neighbours exist
            if len(self.frames) >= 2 * self.phw + 1:
                cand = self.frames[-(self.phw + 1)]
                c_idx, c_flux, _, _ = cand
                neighbours = [f[1] for f in list(self.frames)[-(2 * self.phw + 1):]]
                mask = self.last_onset_strength * float(np.exp(-(c_idx - self.last_onset_frame) / self.mask_tau_frames))
                if (c_idx >= self.warmup_frames and c_flux >= thr and c_flux >= max(neighbours)
                        and c_idx - self.last_onset_frame >= self.min_gap_frames and c_flux > mask):
                    # where did the energy arrive: rise of each band over its pre-onset level
                    fr = list(self.frames)
                    k = len(fr) - (self.phw + 1)
                    before = np.mean([f[3] for f in fr[max(0, k - 3):k]], axis=0) if k > 0 else 0.0
                    after = np.max([f[3] for f in fr[k:k + 3]], axis=0)
                    delta_e = np.maximum(0.0, after - before)
                    centroid_o, bands_o, label, scores = self._classify(delta_e)
                    out.append(Onset(self.frame_time(c_idx), float(c_flux), centroid_o, bands_o, label, scores))
                    self.last_onset_frame = c_idx
                    self.last_onset_strength = float(c_flux)
                    self.total_onsets += 1
        return out

    def flush(self) -> list[Onset]:
        """Pad with silence so the last frames get decided (files only)."""
        return self.push(np.zeros(self.n_fft * 2, dtype=np.float32))


# --------------------------------------------------------------------------- live input


def find_input_device(substring: str) -> int:
    import sounddevice as sd

    devices = sd.query_devices()
    needle = substring.lower()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0 and needle in d["name"].lower():
            return i
    names = [d["name"] for d in devices if d["max_input_channels"] > 0]
    raise ValueError(f"no input device matching {substring!r}; inputs: {names}")


class LiveAudio:
    """Microphone stream feeding an OnsetDetector; onsets carry monotonic-clock timestamps."""

    def __init__(self, device: str | int, rate: int = 48000, blocksize: int = 1024, detector: OnsetDetector | None = None):
        import sounddevice as sd

        self.device = find_input_device(device) if isinstance(device, str) else device
        self.rate = rate
        self.detector = detector or OnsetDetector(rate)
        self.q: queue.Queue[Onset] = queue.Queue()
        self.level = 0.0
        self.offset_hist: deque[float] = deque(maxlen=50)
        self.stream = sd.InputStream(device=self.device, channels=1, samplerate=rate, blocksize=blocksize,
                                     dtype="float32", callback=self._callback)
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        now_mono = time.monotonic()
        # map the ADC time of the first sample onto the monotonic clock
        try:
            offset = now_mono - float(time_info.currentTime)
            t_first = float(time_info.inputBufferAdcTime) + offset
        except Exception:
            t_first = now_mono - frames / self.rate
        mono = indata[:, 0].copy()
        self.level = float(np.sqrt(np.mean(mono ** 2)))
        with self._lock:
            # re-anchor the detector clock every block to follow the stream clock
            self.detector.t0 = t_first - self.detector.consumed / self.rate + 0.0
            self.detector.t0 -= (self.detector.buffer.size) / self.rate  # buffered samples precede this block
            for o in self.detector.push(mono, None):
                self.q.put(o)

    def start(self) -> None:
        self.stream.start()

    def stop(self) -> None:
        self.stream.stop()
        self.stream.close()

    def poll(self) -> list[Onset]:
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out


def onsets_from_samples(samples: np.ndarray, rate: int = 48000, t0: float = 0.0, **kw) -> list[Onset]:
    """Run the detector over a whole array (file replay)."""
    det = OnsetDetector(rate, **kw)
    out: list[Onset] = []
    chunk = rate  # 1 s
    first = True
    for i in range(0, samples.size, chunk):
        out.extend(det.push(samples[i:i + chunk], t0 if first else None))
        first = False
    out.extend(det.flush())
    return out
