"""Pose estimation wrapper (Ultralytics YOLO11-pose, CoreML or PyTorch) and keypoint helpers.

Keypoints follow the COCO-17 layout. All helpers keep the (17, 3) [x, y, confidence] array.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16

SKELETON = [
    (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP), (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
    (NOSE, L_EYE), (NOSE, R_EYE), (L_EYE, L_EAR), (R_EYE, R_EAR),
]


@dataclass
class PoseResult:
    kps: np.ndarray  # (17, 3) x, y, conf in frame pixels
    box: np.ndarray  # (4,) x0, y0, x1, y1
    score: float
    t: float
    index: int


def valid(kps: np.ndarray, idx: int, thr: float = 0.5) -> bool:
    return bool(kps[idx, 2] >= thr)


def midpoint(kps: np.ndarray, a: int, b: int) -> np.ndarray:
    return (kps[a, :2] + kps[b, :2]) / 2.0


def torso_scale(kps: np.ndarray) -> float:
    """Shoulder-midpoint to hip-midpoint distance in pixels (normalisation unit)."""
    return float(np.linalg.norm(midpoint(kps, L_SHOULDER, R_SHOULDER) - midpoint(kps, L_HIP, R_HIP)))


def hitting_wrist(handedness: str) -> int:
    return R_WRIST if handedness == "right" else L_WRIST


def hitting_shoulder(handedness: str) -> int:
    return R_SHOULDER if handedness == "right" else L_SHOULDER


def normalise(kps: np.ndarray) -> np.ndarray:
    """Translate to the hip midpoint and divide by the torso scale; confidences untouched."""
    out = kps.copy()
    hip = midpoint(kps, L_HIP, R_HIP)
    s = max(torso_scale(kps), 1e-6)
    out[:, :2] = (kps[:, :2] - hip) / s
    return out


class KeypointSmoother:
    """Per-keypoint exponential smoothing that ignores low-confidence updates."""

    def __init__(self, alpha: float = 0.6, thr: float = 0.3):
        self.alpha = alpha
        self.thr = thr
        self.state: np.ndarray | None = None

    def update(self, kps: np.ndarray) -> np.ndarray:
        if self.state is None:
            self.state = kps.copy()
            return self.state.copy()
        out = self.state.copy()
        good = kps[:, 2] >= self.thr
        out[good, :2] = self.alpha * kps[good, :2] + (1 - self.alpha) * self.state[good, :2]
        out[:, 2] = kps[:, 2]
        self.state = out
        return out.copy()


class PoseEstimator:
    """YOLO11-pose through Ultralytics. Loads a CoreML package when present, else the .pt on MPS."""

    def __init__(self, model_path: str | Path, fallback: str = "yolo11n-pose.pt", *, imgsz: int = 640,
                 conf: float = 0.25, device: str | None = None):
        from ultralytics import YOLO

        path = Path(model_path)
        self.path = path if path.exists() else Path(fallback)
        self.model = YOLO(str(self.path), task="pose")
        self.imgsz = imgsz
        self.conf = conf
        self.is_coreml = self.path.suffix in (".mlpackage", ".mlmodel")
        if device is None and not self.is_coreml:
            try:
                import torch

                device = "mps" if torch.backends.mps.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = None if self.is_coreml else device
        self.prev_centre: np.ndarray | None = None

    def warmup(self, width: int = 1920, height: int = 1440) -> None:
        self.infer(np.zeros((height, width, 3), dtype=np.uint8), 0.0, -1)

    def infer(self, frame_bgr: np.ndarray, t: float, index: int) -> PoseResult | None:
        kwargs = dict(imgsz=self.imgsz, conf=self.conf, verbose=False)
        if self.device:
            kwargs["device"] = self.device
        res = self.model.predict(frame_bgr, **kwargs)[0]
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return None
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        kps_all = res.keypoints.data.cpu().numpy()  # (n, 17, 3)
        if kps_all.shape[1] != 17:
            return None
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        centres = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
        score = areas * scores
        if self.prev_centre is not None:
            d = np.linalg.norm(centres - self.prev_centre, axis=1)
            score = score * np.exp(-d / (0.25 * frame_bgr.shape[1]))
        i = int(np.argmax(score))
        self.prev_centre = centres[i]
        return PoseResult(kps=kps_all[i].astype(np.float32), box=boxes[i], score=float(scores[i]), t=t, index=index)
