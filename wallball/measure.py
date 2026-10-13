"""Metric scale without a tape measure.

A rectangle on the wall (door, window, poster, wall) has its four corners in a photo, plus a
reference object of known length lying in the wall plane (a 68.6 cm racket, the 6.7 cm ball, the
player's stature). With the camera intrinsics, the rectangle's true aspect ratio follows from
its perspective, and the reference fixes the absolute scale. The corners can then be carried
over to any clip of the same wall by feature matching.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

RACKET_M = 0.686  # standard adult racket length (27 in)
BALL_M = 0.067


def _homography(quad: np.ndarray, aspect: float) -> np.ndarray:
    """World (x right, y up, rectangle = aspect x 1) -> pixels."""
    world = np.array([[0, 1], [aspect, 1], [aspect, 0], [0, 0]], dtype=np.float32)
    return cv2.getPerspectiveTransform(world, np.asarray(quad, dtype=np.float32).reshape(4, 2)).astype(np.float64)


def _orthonormality_error(H: np.ndarray, K: np.ndarray) -> float:
    """How far the first two columns of K^-1 H are from an orthonormal pair (0 = perfect rectangle)."""
    M = np.linalg.inv(K) @ H
    r1, r2 = M[:, 0], M[:, 1]
    n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 1e9
    cos = float(np.dot(r1, r2) / (n1 * n2))
    return abs(cos) + abs(np.log(n1 / n2))


def aspect_from_perspective(quad: np.ndarray, K: np.ndarray, lo: float = 0.15, hi: float = 8.0) -> tuple[float, float]:
    """True width/height ratio of the rectangle whose corners are `quad` (TL, TR, BR, BL), given K.

    Returns (aspect, residual). A frontal view gives a flat residual curve; the pixel aspect
    corrected for foreshortening is then all we can know, and that is what the minimum yields.
    """
    ratios = np.geomspace(lo, hi, 400)
    errs = np.array([_orthonormality_error(_homography(quad, r), K) for r in ratios])
    i = int(np.argmin(errs))
    # refine around the minimum
    a, b = ratios[max(0, i - 2)], ratios[min(len(ratios) - 1, i + 2)]
    fine = np.linspace(a, b, 200)
    ferrs = np.array([_orthonormality_error(_homography(quad, r), K) for r in fine])
    j = int(np.argmin(ferrs))
    return float(fine[j]), float(ferrs[j])


def pixel_aspect(quad: np.ndarray) -> float:
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    w = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
    h = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
    return float(w / h)


@dataclass
class RectMeasure:
    width_m: float
    height_m: float
    bottom_m: float
    aspect: float
    aspect_residual: float
    scale_m_per_unit: float


def measure_rectangle(quad: np.ndarray, K: np.ndarray, ref_px: np.ndarray, ref_len_m: float,
                      floor_px: np.ndarray | None = None, aspect: float | None = None) -> RectMeasure:
    """Metres of the rectangle from its corners, the intrinsics and a reference segment in the wall plane.

    ref_px: 2x2 pixel endpoints of the reference object (in the wall plane). floor_px: a point on
    the wall-floor line, to get the rectangle's height above the floor.
    """
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    if aspect is None:
        aspect, resid = aspect_from_perspective(quad, K)
    else:
        resid = 0.0
    H = _homography(quad, aspect)
    Hinv = np.linalg.inv(H)

    def to_units(p):
        v = Hinv @ np.array([p[0], p[1], 1.0])
        return v[:2] / v[2]

    a, b = to_units(ref_px[0]), to_units(ref_px[1])
    units = float(np.linalg.norm(a - b))
    scale = ref_len_m / max(units, 1e-9)  # metres per unit (the rectangle's height is 1 unit)
    height_m = scale
    width_m = aspect * scale
    bottom_m = 0.0
    if floor_px is not None:
        fy = to_units(floor_px)[1]  # floor line is below the bottom edge (y = 0), so fy < 0
        bottom_m = float(max(0.0, -fy * scale))
    return RectMeasure(width_m, height_m, bottom_m, aspect, resid, scale)


# --------------------------------------------------------------------------- corner transfer


def transfer_corners(ref_img: np.ndarray, ref_quad: np.ndarray, img: np.ndarray, *, min_matches: int = 12
                     ) -> tuple[np.ndarray | None, int, float]:
    """Find the reference rectangle's corners in another image of the same wall by matching SIFT
    features on the wall plane. Returns (quad or None, inliers, mean reprojection consistency px)."""
    sift = cv2.SIFT_create(nfeatures=6000)
    g0 = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if ref_img.ndim == 3 else ref_img
    g1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    k0, d0 = sift.detectAndCompute(g0, None)
    k1, d1 = sift.detectAndCompute(g1, None)
    if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
        return None, 0, 0.0
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(d0, d1, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < min_matches:
        return None, len(good), 0.0
    p0 = np.float32([k0[m.queryIdx].pt for m in good])
    p1 = np.float32([k1[m.trainIdx].pt for m in good])
    # prefer matches near the wall plane: weight by distance to the reference quad (RANSAC handles the rest)
    H, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 4.0)
    if H is None:
        return None, 0, 0.0
    inl = int(mask.sum())
    if inl < min_matches:
        return None, inl, 0.0
    quad = cv2.perspectiveTransform(np.asarray(ref_quad, np.float32).reshape(-1, 1, 2), H).reshape(4, 2)
    proj = cv2.perspectiveTransform(p0.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = float(np.mean(np.linalg.norm(proj[mask.flatten() == 1] - p1[mask.flatten() == 1], axis=1)))
    return quad.astype(np.float64), inl, err


def refine_corners(gray: np.ndarray, quad: np.ndarray, win: int = 9) -> np.ndarray:
    """Snap corners to the nearest strong corner response (helps after a feature-based transfer)."""
    pts = np.asarray(quad, np.float32).reshape(-1, 1, 2)
    try:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        out = cv2.cornerSubPix(gray, pts.copy(), (win, win), (-1, -1), crit).reshape(4, 2)
        moved = np.linalg.norm(out - pts.reshape(4, 2), axis=1)
        out[moved > 2 * win] = pts.reshape(4, 2)[moved > 2 * win]  # do not let a corner run away
        return out.astype(np.float64)
    except cv2.error:
        return np.asarray(quad, np.float64).reshape(4, 2)
