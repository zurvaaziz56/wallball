import numpy as np
import cv2

from wallball.ball import ClassicBallDetector, LegTracker

W, H, FPS = 960, 720, 60.0


def frame_with_ball(rng, cx, cy, r=7.0, streak=None, player=True):
    img = np.full((H, W), 128, np.uint8)
    img += rng.integers(0, 6, (H, W), dtype=np.uint8)  # sensor noise
    if player:  # a static dark "player" rectangle (masked by the caller)
        cv2.rectangle(img, (700, 200), (820, 600), 60, -1)
    if streak is not None:  # motion blur: draw a thick line along the motion
        (x0, y0), (x1, y1) = streak
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), 230, int(2 * r))
    cv2.circle(img, (int(round(cx)), int(round(cy))), int(r), 235, -1)
    return img


def test_detects_moving_ball_and_ignores_player():
    rng = np.random.default_rng(0)
    det = ClassicBallDetector()
    hits = 0
    for i in range(30):
        cx, cy = 100 + i * 12.0, 300 + 2.0 * i
        img = frame_with_ball(rng, cx, cy)
        found = det.detect(img, i / FPS, i, mask_boxes=[(700, 200, 820, 600)])
        if i >= 2:
            assert found, f"no detection at frame {i}"
            d = found[0]
            assert abs(d.x - cx) < 2.5 and abs(d.y - cy) < 2.5, (d, cx, cy)
            assert 5 < d.radius < 10
            hits += 1
    assert hits == 28


def test_streaked_ball_still_found():
    rng = np.random.default_rng(1)
    det = ClassicBallDetector()
    for i in range(6):
        cx, cy = 200 + i * 40.0, 300.0
        img = frame_with_ball(rng, cx, cy, streak=((cx - 30, cy), (cx, cy)))
        found = det.detect(img, i / FPS, i, mask_boxes=[(700, 200, 820, 600)])
        if i >= 2:
            assert found
            assert found[0].elongation > 2.0
            assert abs(found[0].y - cy) < 3


def test_tracker_single_leg_and_reversal():
    tracker = LegTracker()
    det = ClassicBallDetector()
    rng = np.random.default_rng(2)
    pts = [(80 + 14.0 * i, 300.0) for i in range(25)] + [(80 + 14.0 * 24 - 14.0 * i, 300.0 + 3.0 * i) for i in range(1, 20)]
    for i, (cx, cy) in enumerate(pts):
        img = frame_with_ball(rng, cx, cy, player=False)
        found = det.detect(img, i / FPS, i, predicted=tracker.predict(i / FPS))
        tracker.update(found, i / FPS, i)
    assert len(tracker.legs) == 2, [l.n for l in tracker.legs]
    assert tracker.legs[0].reason_ended == "reversal"
    assert tracker.last_reversal is not None
    a, b = tracker.last_reversal
    assert abs(a.x - (80 + 14.0 * 24)) < 4 and b.index == 25


def test_tracker_gap_ends_leg():
    tracker = LegTracker(max_gap_frames=2)
    from wallball.ball import Detection
    for i in range(10):
        tracker.update([Detection(i / FPS, i, 100 + 10 * i, 200, 6, 12, 100, 1.0)], i / FPS, i)
    for i in range(10, 16):  # nothing seen
        tracker.update([], i / FPS, i)
    assert tracker.legs[0].reason_ended == "gap"
    tracker.update([Detection(16 / FPS, 16, 400, 200, 6, 12, 100, 1.0)], 16 / FPS, 16)
    assert len(tracker.legs) == 2
