"""Model download and CoreML export (pose now, the fine-tuned ball detector later)."""
from __future__ import annotations

import shutil
from pathlib import Path


def export_pose(models_dir: str | Path = "models", weights: str = "yolo11n-pose.pt", imgsz: int = 640,
                half: bool = True) -> Path:
    """Download the pose weights if needed and export them to a CoreML package in `models_dir`."""
    from ultralytics import YOLO

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    pt_path = models_dir / weights
    model = YOLO(str(pt_path) if pt_path.exists() else weights)  # downloads to cwd when missing
    src = Path(model.ckpt_path) if getattr(model, "ckpt_path", None) else Path(weights)
    if src.exists() and src.resolve() != pt_path.resolve():
        shutil.copy(src, pt_path)
    out = model.export(format="coreml", imgsz=imgsz, half=half, nms=False)
    out_path = Path(out)
    target = models_dir / (Path(weights).stem + ".mlpackage")
    if out_path.resolve() != target.resolve():
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(out_path), str(target))
    return target


def export_detector(weights: str | Path, models_dir: str | Path = "models", imgsz: int = 1280,
                    half: bool = True) -> Path:
    """Export a fine-tuned ball detector (.pt) to CoreML."""
    from ultralytics import YOLO

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    out = Path(model.export(format="coreml", imgsz=imgsz, half=half, nms=True))
    target = models_dir / (Path(weights).stem + f"_{imgsz}.mlpackage")
    if out.resolve() != target.resolve():
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(out), str(target))
    return target
