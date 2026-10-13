"""Printable markers, a ChArUco board, and the one-time camera intrinsic calibration."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import CameraConfig, WallConfig
from .scene import ARUCO_DICTS

PAPERS_CM = {"A4": (21.0, 29.7), "A3": (29.7, 42.0), "A2": (42.0, 59.4), "LETTER": (21.59, 27.94)}


def _px(cm: float, dpi: int) -> int:
    return int(round(cm / 2.54 * dpi))


def _font(size_px: int):
    try:
        return ImageFont.load_default(size=size_px)
    except TypeError:  # older Pillow
        return ImageFont.load_default()


def marker_canvas(dictionary: str, marker_id: int, size_cm: float, dpi: int, quiet_cm: float = 1.0) -> Image.Image:
    """The marker's black square at exact physical size plus a white quiet zone."""
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dictionary])
    side = _px(size_cm, dpi)
    bitmap = cv2.aruco.generateImageMarker(d, marker_id, side, borderBits=1)
    q = _px(quiet_cm, dpi)
    canvas = np.full((side + 2 * q, side + 2 * q), 255, dtype=np.uint8)
    canvas[q:q + side, q:q + side] = bitmap
    return Image.fromarray(canvas, mode="L")


def tile_pages(canvas: Image.Image, paper: str, dpi: int, label: str, margin_cm: float = 1.0) -> list[Image.Image]:
    """Split a canvas across as many pages as needed; each page gets crop marks and a label."""
    pw_cm, ph_cm = PAPERS_CM[paper.upper()]
    landscape = canvas.width > canvas.height and canvas.width <= _px(ph_cm - 2 * margin_cm, dpi)
    if landscape:
        pw_cm, ph_cm = ph_cm, pw_cm
    pw, ph = _px(pw_cm, dpi), _px(ph_cm, dpi)
    m = _px(margin_cm, dpi)
    tw, th = pw - 2 * m, ph - 2 * m
    cols = max(1, math.ceil(canvas.width / tw))
    rows = max(1, math.ceil(canvas.height / th))
    # equal tiles so seams are easy to align
    tile_w = math.ceil(canvas.width / cols)
    tile_h = math.ceil(canvas.height / rows)
    pages = []
    font = _font(max(12, _px(0.35, dpi)))
    for r in range(rows):
        for c in range(cols):
            page = Image.new("L", (pw, ph), 255)
            box = (c * tile_w, r * tile_h, min(canvas.width, (c + 1) * tile_w), min(canvas.height, (r + 1) * tile_h))
            tile = canvas.crop(box)
            page.paste(tile, (m, m))
            draw = ImageDraw.Draw(page)
            x1, y1 = m + tile.width, m + tile.height
            for (xa, ya, xb, yb) in [(m - 40, m, m, m), (m, m - 40, m, m), (x1, m, x1 + 40, m), (x1, m - 40, x1, m),
                                     (m - 40, y1, m, y1), (m, y1, m, y1 + 40), (x1, y1, x1 + 40, y1), (x1, y1, x1, y1 + 40)]:
                draw.line([(xa, ya), (xb, yb)], fill=0, width=2)
            text = label if rows * cols == 1 else f"{label}  |  tile row {r + 1}/{rows}, col {c + 1}/{cols}: cut on the marks, butt the tiles together"
            draw.text((m, ph - m + 10), text, fill=0, font=font)
            pages.append(page)
    return pages


def _save_pdf(pages: list[Image.Image], path: Path, dpi: int) -> None:
    pages[0].save(path, "PDF", resolution=dpi, save_all=True, append_images=pages[1:])


def generate_marker_pdfs(wall: WallConfig, out_dir: str | Path, paper: str = "A4", dpi: int = 200,
                         quiet_cm: float = 1.0) -> list[Path]:
    """One PDF per marker at its configured physical size (tiled over pages when larger than the paper)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for m in wall.markers:
        size_cm = m.size_m * 100.0
        canvas = marker_canvas(wall.dictionary, m.id, size_cm, dpi, quiet_cm)
        label = f"wallball marker id {m.id}  ({wall.dictionary})  black square = {size_cm:.0f} cm  centre at X={m.x_m:+.2f} m, height {m.y_m:.2f} m"
        pages = tile_pages(canvas, paper, dpi, label)
        path = out_dir / f"marker_{m.id:02d}_{size_cm:.0f}cm.pdf"
        _save_pdf(pages, path, dpi)
        paths.append(path)
    readme = out_dir / "README.md"
    readme.write_text(_readme(wall, paper))
    return paths


def _readme(wall: WallConfig, paper: str) -> str:
    rows = "\n".join(
        f"| {m.id} | {m.size_m * 100:.0f} cm | {m.x_m:+.2f} m | {m.y_m:.2f} m |" for m in wall.markers
    )
    return f"""# Wall markers

Print every `marker_*.pdf` at **100 % scale** (no "fit to page"). Large markers come as tiles:
cut on the crop marks and butt the tiles together on the wall (a millimetre gap is fine).

Tape each marker so its centre sits at the position below. X runs along the wall and is
**positive toward the player's right when facing the wall**; the height is measured from the
floor to the marker centre. The practice rectangle spans X {wall.practice_rect.x0_m:.2f} to
{wall.practice_rect.x1_m:.2f} m, height {wall.practice_rect.y0_m:.2f} to {wall.practice_rect.y1_m:.2f} m.

| id | black square | centre X | centre height |
| --- | --- | --- | --- |
{rows}

After taping, **measure the real centre positions and the real black-square sizes** with a tape
measure (printers scale slightly) and write them into `configs/wall.yaml`. Accuracy of about
1 cm is enough. Keep at least a couple of centimetres of plain wall around each marker.

Paper assumed: {paper}. Any size works as long as the yaml matches what is on the wall.
"""


def generate_charuco_pdf(out_path: str | Path, squares_x: int = 5, squares_y: int = 7, square_cm: float = 3.8,
                         marker_cm: float = 2.8, dictionary: str = "DICT_5X5_100", paper: str = "A4",
                         dpi: int = 200) -> Path:
    """ChArUco board for the one-time intrinsic calibration (default fits A4 portrait)."""
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dictionary])
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_cm / 100.0, marker_cm / 100.0, d)
    w, h = _px(squares_x * square_cm, dpi), _px(squares_y * square_cm, dpi)
    img = board.generateImage((w, h), marginSize=0, borderBits=1)
    canvas = Image.fromarray(img, mode="L")
    label = (f"wallball ChArUco {squares_x}x{squares_y}, square {square_cm} cm, marker {marker_cm} cm, {dictionary}. "
             f"Print at 100 %, glue flat on card, then measure one square and pass --square-cm")
    pages = tile_pages(canvas, paper, dpi, label)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_pdf(pages, out_path, dpi)
    return out_path


class CharucoCalibrator:
    """Collects ChArUco views and solves the intrinsics."""

    def __init__(self, squares_x: int = 5, squares_y: int = 7, square_m: float = 0.038, marker_m: float = 0.028,
                 dictionary: str = "DICT_5X5_100", min_corners: int = 8, min_move_px: float = 60.0):
        d = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dictionary])
        self.board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_m, marker_m, d)
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.min_corners = min_corners
        self.min_move_px = min_move_px
        self.obj_points: list[np.ndarray] = []
        self.img_points: list[np.ndarray] = []
        self.last_centroid: np.ndarray | None = None
        self.coverage = np.zeros((4, 6), dtype=bool)

    def detect(self, gray: np.ndarray):
        ch_corners, ch_ids, _, _ = self.detector.detectBoard(gray)
        return ch_corners, ch_ids

    def maybe_add(self, gray: np.ndarray, ch_corners, ch_ids) -> bool:
        if ch_ids is None or ch_corners is None or len(ch_ids) < self.min_corners:
            return False
        centroid = ch_corners.reshape(-1, 2).mean(axis=0)
        if self.last_centroid is not None and np.linalg.norm(centroid - self.last_centroid) < self.min_move_px:
            return False
        obj, img = self.board.matchImagePoints(ch_corners, ch_ids)
        if obj is None or len(obj) < self.min_corners:
            return False
        self.obj_points.append(obj.reshape(-1, 1, 3).astype(np.float32))
        self.img_points.append(img.reshape(-1, 1, 2).astype(np.float32))
        self.last_centroid = centroid
        h, w = gray.shape[:2]
        for (x, y) in ch_corners.reshape(-1, 2):
            self.coverage[min(3, int(y / h * 4)), min(5, int(x / w * 6))] = True
        return True

    @property
    def n_views(self) -> int:
        return len(self.obj_points)

    def calibrate(self, width: int, height: int, name: str = "camera") -> CameraConfig:
        rms, K, dist, _, _ = cv2.calibrateCamera(self.obj_points, self.img_points, (width, height), None, None)
        dist = np.asarray(dist).flatten()[:5].tolist()
        return CameraConfig(width, height, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), dist,
                            name=name, rms_px=float(rms))


def run_camera_calibration(source, out_path: str | Path, *, target_views: int = 60, name: str = "camera",
                           display: bool = True, **board_kwargs) -> CameraConfig | None:
    """Interactive loop: move the board around the frame (all corners, tilted, near and far)."""
    calib = CharucoCalibrator(**board_kwargs)
    width = height = None
    result = None
    try:
        for frame in source:
            gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2]
            ch_corners, ch_ids = calib.detect(gray)
            added = calib.maybe_add(gray, ch_corners, ch_ids)
            if display:
                view = frame.image
                if ch_ids is not None and ch_corners is not None:
                    cv2.aruco.drawDetectedCornersCharuco(view, ch_corners, ch_ids, (0, 255, 0))
                h, w = view.shape[:2]
                for r in range(4):
                    for c in range(6):
                        col = (0, 200, 0) if calib.coverage[r, c] else (0, 0, 220)
                        cv2.rectangle(view, (int(c * w / 6), int(r * h / 4)), (int((c + 1) * w / 6) - 1, int((r + 1) * h / 4) - 1), col, 2)
                msg = f"views {calib.n_views}/{target_views}   c = calibrate now   q = quit"
                cv2.putText(view, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.imshow("wallball calib-camera", cv2.resize(view, None, fx=0.5, fy=0.5))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c") and calib.n_views >= 10:
                    break
            elif added:
                print(f"view {calib.n_views}/{target_views}")
            if calib.n_views >= target_views:
                break
    finally:
        source.close()
        if display:
            cv2.destroyAllWindows()
    if width is None or calib.n_views < 10:
        print(f"not enough views ({calib.n_views}); need at least 10, ideally {target_views}")
        return None
    result = calib.calibrate(width, height, name=name)
    result.save(out_path)
    print(f"calibrated from {calib.n_views} views: fx={result.fx:.1f} fy={result.fy:.1f} "
          f"cx={result.cx:.1f} cy={result.cy:.1f} rms={result.rms_px:.3f} px -> {out_path}")
    return result
