import numpy as np

from wallball.audio import FLOOR, POP, THUD, OnsetDetector, onsets_from_samples

RATE = 48000


def burst(kind: str, rng: np.random.Generator, dur_s: float = 0.08) -> np.ndarray:
    """Exponentially decaying noise burst coloured like a racket pop, a wall thud or a floor bounce."""
    n = int(dur_s * RATE)
    noise = rng.standard_normal(n).astype(np.float32)
    spec = np.fft.rfft(noise)
    f = np.fft.rfftfreq(n, 1.0 / RATE)
    if kind == POP:
        shape = 1.0 / (1.0 + np.exp(-(f - 3500) / 500))  # bright
        tau = 0.012
    elif kind == THUD:
        shape = 1.0 / (1.0 + np.exp((f - 900) / 200))  # dull
        tau = 0.03
    else:
        shape = np.exp(-((f - 2200) / 900) ** 2)  # mid
        tau = 0.02
    sig = np.fft.irfft(spec * shape, n).astype(np.float32)
    env = np.exp(-np.arange(n) / (tau * RATE)).astype(np.float32)
    sig *= env
    return sig / (np.abs(sig).max() + 1e-9)


def synth(events, seconds: float, rng, noise_level=0.003, reverb=True) -> np.ndarray:
    x = rng.standard_normal(int(seconds * RATE)).astype(np.float32) * noise_level
    for t, kind, amp in events:
        b = burst(kind, rng) * amp
        i = int(t * RATE)
        b = b[: max(0, x.size - i)]
        x[i:i + b.size] += b
    if reverb:
        # early reflections (-14 / -18 / -22 dB) plus a diffuse tail with RT60 about 0.5 s
        y = x.copy()
        for d_ms, g in ((23, 0.2), (41, 0.125), (67, 0.08)):
            d = int(d_ms * RATE / 1000)
            y[d:] += g * x[:-d]
        tail_n = int(0.4 * RATE)
        tail = rng.standard_normal(tail_n).astype(np.float32) * np.exp(-np.arange(tail_n) / (0.08 * RATE)).astype(np.float32)
        tail *= 0.03 / (np.abs(tail).max() + 1e-9)
        y = y + np.convolve(x, tail)[: y.size].astype(np.float32)
        x = y
    return x


def _check(events, onsets, tol=0.012):
    assert len(onsets) == len(events), [(round(o.t, 3), o.label) for o in onsets]
    for (t, kind, _), o in zip(events, onsets):
        assert abs(o.t - t) < tol, (t, o)
        assert o.label == kind, (t, kind, o)
    # relative timing between consecutive impacts is what the speed uses: a few ms at most
    for k in range(1, len(events)):
        assert abs((onsets[k].t - onsets[k - 1].t) - (events[k][0] - events[k - 1][0])) < 0.006


def test_onsets_timing_and_labels():
    rng = np.random.default_rng(1)
    events = [(0.5, POP, 0.8), (0.78, THUD, 0.35), (1.6, POP, 0.9), (1.9, THUD, 0.3), (2.05, FLOOR, 0.25),
              (3.0, POP, 0.7), (3.2, THUD, 0.4)]
    x = synth(events, 4.0, rng)
    onsets = onsets_from_samples(x, RATE)
    _check(events, onsets)


def test_volley_rhythm_100ms():
    rng = np.random.default_rng(2)
    events = []
    t = 0.5
    for _ in range(8):
        events.append((t, POP, 0.8))
        events.append((t + 0.1, THUD, 0.4))
        t += 0.2
    x = synth(events, 3.0, rng)
    onsets = onsets_from_samples(x, RATE)
    _check(events, onsets)


def test_streaming_matches_batch():
    rng = np.random.default_rng(3)
    events = [(0.4, POP, 0.8), (0.7, THUD, 0.4), (1.5, POP, 0.8), (1.8, THUD, 0.4)]
    x = synth(events, 2.5, rng)
    batch = onsets_from_samples(x, RATE)
    det = OnsetDetector(RATE)
    stream = []
    for i in range(0, x.size, 1024):  # live-like blocks
        stream.extend(det.push(x[i:i + 1024], 0.0 if i == 0 else None))
    stream.extend(det.flush())
    assert [round(o.t, 3) for o in stream] == [round(o.t, 3) for o in batch]


def test_quiet_thud_after_loud_pop_with_reverb():
    rng = np.random.default_rng(4)
    events = [(1.0, POP, 1.0), (1.12, THUD, 0.12)]  # thud 18 dB quieter, 120 ms later, inside the reverb tail
    x = synth(events, 2.0, rng)
    onsets = onsets_from_samples(x, RATE)
    _check(events, onsets)
