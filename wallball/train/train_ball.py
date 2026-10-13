"""Fine-tune yolo11n on the labelled ball frames and export to CoreML.

    uv run python -m wallball.train.train_ball data/ball/ball.yaml --epochs 60 --imgsz 1280
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args(argv)
    from ultralytics import YOLO

    from ..export import export_detector

    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, device="mps", batch=8, project="runs", name="ball",
                exist_ok=True, mosaic=1.0, degrees=5.0, scale=0.3, fliplr=0.5)
    best = Path("runs/ball/weights/best.pt")
    out = export_detector(best, args.models_dir, imgsz=args.imgsz)
    print("trained and exported:", out)
    print("set ball_model in configs/app.yaml to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
