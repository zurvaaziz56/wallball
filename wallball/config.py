"""Configuration: wall marker layout, camera intrinsics, player and app settings.

World frame (anchored to the wall):
  X  along the wall, +X is the player's RIGHT when facing the wall
  Y  up, Y = 0 at the floor
  Z  out of the wall toward the player, Z = 0 on the wall plane
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml


# --------------------------------------------------------------------------- wall


@dataclass
class Marker:
    """One printed ArUco marker taped on the wall."""

    id: int
    size_m: float  # side of the black square, metres
    x_m: float  # centre X on the wall, metres (+X = player's right when facing the wall)
    y_m: float  # centre height above the floor, metres

    def corners_world(self) -> np.ndarray:
        """4x3 world corners in OpenCV ArUco order (TL, TR, BR, BL as seen facing the wall)."""
        h = self.size_m / 2.0
        return np.array(
            [
                [self.x_m - h, self.y_m + h, 0.0],
                [self.x_m + h, self.y_m + h, 0.0],
                [self.x_m + h, self.y_m - h, 0.0],
                [self.x_m - h, self.y_m - h, 0.0],
            ],
            dtype=np.float64,
        )


@dataclass
class PracticeRect:
    """Target rectangle on the wall used for the placement panel (metres, wall frame)."""

    x0_m: float = 0.0
    y0_m: float = 0.8
    x1_m: float = 2.0
    y1_m: float = 2.3

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x0_m + self.x1_m) / 2.0, (self.y0_m + self.y1_m) / 2.0)

    @property
    def width_m(self) -> float:
        return self.x1_m - self.x0_m

    @property
    def height_m(self) -> float:
        return self.y1_m - self.y0_m

    def corners_world(self) -> np.ndarray:
        return np.array(
            [
                [self.x0_m, self.y1_m, 0.0],
                [self.x1_m, self.y1_m, 0.0],
                [self.x1_m, self.y0_m, 0.0],
                [self.x0_m, self.y0_m, 0.0],
            ],
            dtype=np.float64,
        )

    def normalise(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Wall point -> (u, v) in [0, 1]^2 relative to the rectangle (v grows upward)."""
        u = (x_m - self.x0_m) / self.width_m
        v = (y_m - self.y0_m) / self.height_m
        return u, v


@dataclass
class WallConfig:
    dictionary: str = "DICT_4X4_50"
    markers: list[Marker] = field(default_factory=list)
    practice_rect: PracticeRect = field(default_factory=PracticeRect)
    min_markers: int = 3
    min_span_m: float = 2.0
    max_reproj_px: float = 1.5

    @classmethod
    def default(cls) -> "WallConfig":
        """Suggested layout: two 40 cm markers high in the corners, four 20 cm down the sides."""
        return cls(
            markers=[
                Marker(0, 0.40, -0.30, 2.70),
                Marker(1, 0.40, 2.30, 2.70),
                Marker(2, 0.20, -0.30, 1.90),
                Marker(3, 0.20, 2.30, 1.90),
                Marker(4, 0.20, -0.30, 1.10),
                Marker(5, 0.20, 2.30, 1.10),
            ],
            practice_rect=PracticeRect(0.0, 0.8, 2.0, 2.3),
        )

    def marker_by_id(self, marker_id: int) -> Marker | None:
        for m in self.markers:
            if m.id == marker_id:
                return m
        return None

    @property
    def ids(self) -> list[int]:
        return [m.id for m in self.markers]

    def to_dict(self) -> dict:
        return {
            "dictionary": self.dictionary,
            "min_markers": self.min_markers,
            "min_span_m": self.min_span_m,
            "max_reproj_px": self.max_reproj_px,
            "practice_rect": asdict(self.practice_rect),
            "markers": [asdict(m) for m in self.markers],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WallConfig":
        return cls(
            dictionary=d.get("dictionary", "DICT_4X4_50"),
            markers=[Marker(**m) for m in d.get("markers", [])],
            practice_rect=PracticeRect(**d.get("practice_rect", {})),
            min_markers=int(d.get("min_markers", 3)),
            min_span_m=float(d.get("min_span_m", 2.0)),
            max_reproj_px=float(d.get("max_reproj_px", 1.5)),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def load(cls, path: str | Path) -> "WallConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})


# --------------------------------------------------------------------------- camera


@dataclass
class CameraConfig:
    """Pinhole intrinsics plus OpenCV distortion (k1, k2, p1, p2, k3)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    name: str = "nominal"
    rms_px: float = float("nan")

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def dist_coeffs(self) -> np.ndarray:
        return np.array(self.dist, dtype=np.float64).reshape(-1, 1)

    @classmethod
    def nominal(cls, width: int, height: int, hfov_deg: float = 70.0, name: str = "nominal") -> "CameraConfig":
        """Uncalibrated guess from the horizontal field of view (iPhone main lens is about 70 deg)."""
        f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(width, height, f, f, width / 2.0, height / 2.0, name=name)

    def scaled(self, width: int, height: int) -> "CameraConfig":
        """Intrinsics for a frame of another size (same lens, same aspect ratio assumed)."""
        sx, sy = width / self.width, height / self.height
        return CameraConfig(width, height, self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, list(self.dist), self.name, self.rms_px)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def load(cls, path: str | Path) -> "CameraConfig":
        d = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**d)


# --------------------------------------------------------------------------- player


@dataclass
class PlayerConfig:
    handedness: str = "right"  # "right" | "left"
    height_m: float = 1.75
    reach_m: float = 0.7  # fallback geometry only
    two_handed_backhand: bool = False

    @property
    def sign(self) -> float:
        """+1 for right-handers, -1 for left-handers (mirrors every lateral rule)."""
        return 1.0 if self.handedness == "right" else -1.0

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def load(cls, path: str | Path) -> "PlayerConfig":
        d = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**d)


# --------------------------------------------------------------------------- app


@dataclass
class AppConfig:
    """Runtime settings (camera selection, model paths, rates)."""

    camera_name: str = "iPhone"  # substring of the avfoundation device name
    camera_index: int | None = None  # OpenCV index fallback
    width: int = 1920
    height: int = 1440
    fps: float = 60.0
    video_backend: str = "ffmpeg"  # "ffmpeg" | "opencv"
    audio_device: str = "iPhone"  # substring of the sounddevice input name
    audio_rate: int = 48000
    pose_model: str = "models/yolo11n-pose.mlpackage"
    pose_fallback: str = "yolo11n-pose.pt"
    pose_every: int = 2
    pose_imgsz: int = 640
    ball_model: str | None = None  # CoreML/PT path once trained; None = classic detector
    wall_config: str = "configs/wall.yaml"
    camera_config: str = "configs/camera.yaml"
    player_config: str = "configs/player.yaml"
    display_width: int = 1280

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        d = yaml.safe_load(p.read_text()) or {}
        return cls(**d)


def load_all(root: str | Path = ".") -> tuple[AppConfig, WallConfig, CameraConfig | None, PlayerConfig]:
    """Load app/wall/camera/player configs from a project root, with defaults where files are missing."""
    root = Path(root)
    app = AppConfig.load(root / "configs" / "app.yaml")
    wall_path = root / app.wall_config
    wall = WallConfig.load(wall_path) if wall_path.exists() else WallConfig.default()
    cam_path = root / app.camera_config
    cam = CameraConfig.load(cam_path) if cam_path.exists() else None
    player_path = root / app.player_config
    player = PlayerConfig.load(player_path) if player_path.exists() else PlayerConfig()
    return app, wall, cam, player


# --------------------------------------------------------------------------- measured rectangle (no markers)


@dataclass
class SceneRect:
    """A measured rectangle on the wall (door frame, window, poster, the wall itself, or four sticky
    notes) whose four corners were clicked once in a frame. Replaces the printed markers."""

    width_m: float
    height_m: float
    bottom_m: float = 0.0  # height of the bottom edge above the floor
    x0_m: float = 0.0  # X of the left edge (player's right is +X); 0 unless you care about absolute X
    corners: list[list[float]] = field(default_factory=list)  # 4 x [u, v] pixels: TL, TR, BR, BL as seen in the frame
    frame_width: int = 0
    frame_height: int = 0
    source: str = ""
    f_px: float | None = None  # optional focal length override for frame_width
    ref_image: str = ""  # image the corners were marked on (enables corner transfer to clips)
    measured_by: str = ""  # how the metres were obtained (e.g. "racket 68.6 cm")
    aspect: float | None = None

    def corners_world(self) -> np.ndarray:
        x0, x1 = self.x0_m, self.x0_m + self.width_m
        y0, y1 = self.bottom_m, self.bottom_m + self.height_m
        return np.array([[x0, y1, 0.0], [x1, y1, 0.0], [x1, y0, 0.0], [x0, y0, 0.0]], dtype=np.float64)

    def practice_rect(self) -> PracticeRect:
        return PracticeRect(self.x0_m, self.bottom_m, self.x0_m + self.width_m, self.bottom_m + self.height_m)

    def wall_config(self) -> WallConfig:
        return WallConfig(markers=[], practice_rect=self.practice_rect(), min_markers=1, min_span_m=0.0)

    def corners_for(self, width: int, height: int) -> np.ndarray:
        """Clicked corners scaled to another frame size."""
        c = np.asarray(self.corners, dtype=np.float64).reshape(4, 2)
        if self.frame_width and self.frame_height:
            c = c * np.array([width / self.frame_width, height / self.frame_height])
        return c

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def load(cls, path: str | Path) -> "SceneRect":
        d = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**d)


def scene_file_for(clip: str | Path) -> Path:
    """Where the clicked-corner scene file for a clip lives: recordings/clip.mov -> recordings/clip.scene.yaml"""
    p = Path(clip)
    return p.with_name(p.stem + ".scene.yaml")
