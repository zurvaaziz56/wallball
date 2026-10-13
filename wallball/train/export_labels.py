"""Bootstrap YOLO labels for the ball detector from the classic detector on recorded clips.

    uv run python -m wallball.train.export_labels recordings/clip1.mov recordings/clip2.mov --out data/ball

Writes data/ball/images/*.jpg and data/ball/labels/*.txt (class 0 = ball) for frames where the
tracker had a confident, moving ball. Review and correct the boxes in any YOLO-format labeller
(Roboflow, CVAT, Label Studio) before training; a few hundred corrected frames across your
viewpoints are enough for yolo11n.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from ..ball import ClassicBallDetector, LegTracker
from ..config import load_all
from ..scene import STATIC, SceneTracker
from ..sources import FileSource


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--out", default="data/ball")
    ap.add_argument("--every", type=int, default=3, help="keep every Nth tracked frame")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    app, wall, cam, _ = load_all(args.root)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    for clip in args.clips:
        src = FileSource(clip)
        scene = SceneTracker(wall, (cam or __import__("wallball.config", fromlist=["CameraConfig"]).CameraConfig.nominal(src.width, src.height)).scaled(src.width, src.height))
        det = ClassicBallDetector()
        tracker = LegTracker()
        stem = Path(clip).stem
        for frame in src:
            gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
            spose = scene.update(gray, frame.t)
            static = spose is not None and spose.motion == STATIC
            dets = det.detect(frame.image, frame.t, frame.index, predicted=tracker.predict(frame.t), static=static)
            pt = tracker.update(dets, frame.t, frame.index)
            leg = tracker.current
            if pt is None or leg is None or leg.n < 4 or frame.index % args.every:
                continue
            h, w = frame.image.shape[:2]
            r = max(pt.r, 3.0) * 1.3
            name = f"{stem}_{frame.index:06d}"
            cv2.imwrite(str(out / "images" / f"{name}.jpg"), frame.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out / "labels" / f"{name}.txt").write_text(f"0 {pt.x / w:.6f} {pt.y / h:.6f} {2 * r / w:.6f} {2 * r / h:.6f}\n")
            n += 1
        src.close()
    (out / "ball.yaml").write_text(f"path: {out.resolve()}\ntrain: images\nval: images\nnames:\n  0: ball\n")
    print(f"wrote {n} labelled frames to {out} (split train/val yourself before training)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
