"""Command line entry point: `wallball <command>`."""
from __future__ import annotations

import argparse
import sys

import numpy as np
import time
from pathlib import Path

from .config import CameraConfig, SceneRect, WallConfig, load_all, scene_file_for

ROOT = Path(__file__).resolve().parent.parent


def _root(args) -> Path:
    return Path(args.root).resolve() if args.root else ROOT


def cmd_devices(args) -> int:
    from .sources import list_avfoundation_devices, resolve_device, supported_modes

    video, audio = list_avfoundation_devices()
    print("video devices:")
    for i, v in enumerate(video):
        print(f"  [{i}] {v}")
    print("audio devices:")
    for i, a in enumerate(audio):
        print(f"  [{i}] {a}")
    app, _, _, _ = load_all(_root(args))
    try:
        dev = resolve_device(args.camera or app.camera_name, video)
    except ValueError as e:
        print(e)
        return 1
    print(f"\nsupported modes for {dev!r}:")
    for m in supported_modes(dev):
        print("  ", m)
    return 0


def cmd_markers(args) -> int:
    from .calib import generate_charuco_pdf, generate_marker_pdfs

    root = _root(args)
    wall_path = root / "configs" / "wall.yaml"
    if wall_path.exists():
        wall = WallConfig.load(wall_path)
    else:
        wall = WallConfig.default()
        wall.save(wall_path)
        print(f"wrote default layout to {wall_path} (edit it after measuring the taped markers)")
    out = root / "markers"
    paths = generate_marker_pdfs(wall, out, paper=args.paper, dpi=args.dpi)
    board = generate_charuco_pdf(out / "charuco_board.pdf", paper=args.paper, dpi=args.dpi)
    for p in paths:
        print("wrote", p)
    print("wrote", board)
    print("wrote", out / "README.md")
    return 0


def cmd_calib_camera(args) -> int:
    from .calib import run_camera_calibration

    root = _root(args)
    app, _, _, _ = load_all(root)
    source = _open_source(args, app)
    out = root / (args.out or app.camera_config)
    res = run_camera_calibration(source, out, target_views=args.views, name=args.name, display=not args.no_display,
                                 squares_x=args.squares_x, squares_y=args.squares_y,
                                 square_m=args.square_cm / 100.0, marker_m=args.marker_cm / 100.0)
    return 0 if res is not None else 1


def cmd_export_models(args) -> int:
    from .export import export_pose

    root = _root(args)
    path = export_pose(root / "models", imgsz=args.imgsz)
    print("exported", path)
    return 0


def _open_source(args, app):
    from .sources import FFmpegLiveSource, FileSource, OpenCVLiveSource, list_avfoundation_devices, resolve_device

    file = getattr(args, "file", None)
    if file:
        return FileSource(file, realtime=getattr(args, "realtime", False), start_s=getattr(args, "start", 0.0),
                          end_s=getattr(args, "end", None))
    backend = getattr(args, "backend", None) or app.video_backend
    if backend == "opencv":
        return OpenCVLiveSource(app.camera_index or 0, app.width, app.height, app.fps)
    video, _ = list_avfoundation_devices()
    dev = resolve_device(getattr(args, "camera", None) or app.camera_name, video)
    print(f"opening {dev!r} at {app.width}x{app.height} @ {app.fps:g} fps")
    return FFmpegLiveSource(dev, app.width, app.height, app.fps)


def _scene_rect_for(args, replay: bool) -> SceneRect | None:
    path = getattr(args, "scene", None)
    if path is None and replay:
        cand = scene_file_for(args.file)
        path = str(cand) if cand.exists() else None
    if path is None:
        return None
    rect = SceneRect.load(path)
    print(f"scene: measured rectangle {rect.width_m:.2f} x {rect.height_m:.2f} m from {path}")
    return rect


def _run_pipeline(args, replay: bool) -> int:
    from .app import Pipeline

    root = _root(args)
    app, wall, cam, player = load_all(root)
    pipe = Pipeline(app, wall, cam, player, root=root, enable_pose=not args.no_pose, enable_ball=not args.no_ball,
                    display=not args.no_display, out_video=args.out_video, scene_rect=_scene_rect_for(args, replay))
    source = _open_source(args, app)
    if not args.no_audio:
        if replay:
            n = pipe.attach_audio_file(args.file)
            print(f"audio: {n} onsets found in the clip's track")
        else:
            pipe.attach_live_audio()
    session = args.session_json
    if session is None and replay:
        session = str(Path(args.file).with_suffix('')) + '_session.json'
    state = pipe.run(source, max_seconds=args.max_seconds, session_json=session)
    print(f"done: {state.strokes} strokes, {state.forehands} forehands, {state.backhands} backhands")
    return 0


def cmd_live(args) -> int:
    args.file = None
    return _run_pipeline(args, replay=False)


def cmd_replay(args) -> int:
    return _run_pipeline(args, replay=True)


def cmd_corners(args) -> int:
    """Click the four corners of a measured rectangle on a frame; saves a scene file."""
    import cv2
    import numpy as np

    root = _root(args)
    app, _, _, _ = load_all(root)
    if args.source == "live":
        args.file = None
        src = _open_source(args, app)
        frame = None
        for i, fr in enumerate(src):
            frame = fr
            if i >= 10:
                break
        src.close()
        out = Path(args.out or (root / "configs" / "scene_live.yaml"))
        source_name = "live"
    else:
        from .sources import FileSource

        src = FileSource(args.source, start_s=args.at)
        frame = next(iter(src))
        src.close()
        out = Path(args.out or scene_file_for(args.source))
        source_name = str(args.source)
    if frame is None:
        print("no frame available")
        return 1
    img = frame.image
    h, w = img.shape[:2]
    scale = min(1.0, 1400.0 / w)
    view = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1.0 else img.copy()
    pts: list[tuple[float, float]] = []
    names = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]

    def redraw():
        canvas = view.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(canvas, (int(x * scale), int(y * scale)), 6, (255, 0, 255), -1)
            cv2.putText(canvas, names[i], (int(x * scale) + 8, int(y * scale) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        if len(pts) == 4:
            poly = np.array([[int(x * scale), int(y * scale)] for x, y in pts], np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [poly], True, (255, 0, 255), 2)
        msg = f"click {names[len(pts)]}" if len(pts) < 4 else "ENTER = save   r = reset   q = quit"
        cv2.putText(canvas, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow("wallball corners", canvas)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x / scale, y / scale))
            redraw()

    cv2.namedWindow("wallball corners")
    cv2.setMouseCallback("wallball corners", on_mouse)
    redraw()
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:
            cv2.destroyAllWindows()
            return 1
        if key == ord("r"):
            pts.clear()
            redraw()
        if key in (13, 10, 32) and len(pts) == 4:
            break
    cv2.destroyAllWindows()
    width_m = args.width_m or float(input("rectangle width in metres: "))
    height_m = args.height_m or float(input("rectangle height in metres: "))
    bottom_m = args.bottom_m if args.bottom_m is not None else float(input("height of its bottom edge above the floor in metres (0 for a door or the wall): ") or 0)
    rect = SceneRect(width_m, height_m, bottom_m, 0.0, [[float(x), float(y)] for x, y in pts], w, h, source_name)
    rect.save(out)
    print(f"saved {out}")
    return 0


def cmd_lens(args) -> int:
    """Focal length from a photo's metadata (35 mm equivalent) -> configs/camera.yaml (no board needed)."""
    import math
    import subprocess

    root = _root(args)
    app, _, _, _ = load_all(root)
    f35 = None
    try:
        out = subprocess.run(["mdls", "-name", "kMDItemFocalLength35mm", "-raw", args.photo], capture_output=True, text=True).stdout.strip()
        if out and out != "(null)":
            f35 = float(out)
    except Exception:
        pass
    pw = ph = None
    try:
        from PIL import Image

        with Image.open(args.photo) as im:
            pw, ph = im.size
            if f35 is None:
                exif = im.getexif()
                f35 = float(exif.get(41989) or exif.get_ifd(0x8769).get(41989))
    except Exception:
        pass
    if f35 is None and args.hfov is None:
        print("could not read the 35 mm-equivalent focal length; pass --hfov <degrees> instead")
        return 1
    W, H = app.width, app.height
    if args.hfov is not None:
        f = (W / 2.0) / math.tan(math.radians(args.hfov) / 2.0)
        note = f"hfov {args.hfov} deg"
    else:
        # 35 mm equivalence is defined on the image diagonal (43.27 mm)
        if pw and ph:
            diag_px = math.hypot(pw, ph)
            f_photo = f35 / 43.27 * diag_px
            f = f_photo * W / pw
        else:
            f = f35 / 36.0 * W
        note = f"35mm-equiv {f35:.1f} from {Path(args.photo).name}"
    cam = CameraConfig(W, H, f, f, W / 2.0, H / 2.0, name=f"iphone ({note})")
    out = root / (args.out or app.camera_config)
    cam.save(out)
    print(f"wrote {out}: f = {f:.0f} px for {W}x{H} ({note}); hfov {math.degrees(2 * math.atan(W / (2 * f))):.1f} deg")
    return 0


def _parse_pts(items):
    import numpy as np

    return np.array([[float(v) for v in it.split(",")] for it in items], dtype=np.float64)


def _load_image(path: str, at: float = 0.5):
    """An image file, or a frame of a clip at `at` seconds. Returns (bgr, is_clip)."""
    import cv2

    p = Path(path)
    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".bmp", ".webp"):
        img = cv2.imread(str(p))
        if img is None:  # HEIC via pillow-heif if installed, else via sips (macOS)
            import subprocess
            import tempfile

            tmp = Path(tempfile.mkdtemp()) / "conv.jpg"
            subprocess.run(["sips", "-s", "format", "jpeg", str(p), "--out", str(tmp)], capture_output=True)
            img = cv2.imread(str(tmp))
        if img is None:
            raise SystemExit(f"cannot read image {path}")
        return img, False
    from .sources import FileSource

    src = FileSource(path, start_s=at)
    frame = next(iter(src))
    src.close()
    return frame.image, True


def cmd_measure(args) -> int:
    """Metres of a wall rectangle from pixel coordinates (no GUI): corners + a reference of known length."""
    from .measure import BALL_M, RACKET_M, measure_rectangle

    root = _root(args)
    app, _, cam, _ = load_all(root)
    img, _ = _load_image(args.image, args.at)
    h, w = img.shape[:2]
    quad = _parse_pts(args.corners)
    ref = _parse_pts(args.ref)
    ref_m = args.ref_m if args.ref_m else (BALL_M if args.ball else RACKET_M)
    floor = _parse_pts(args.floor)[0] if args.floor else None
    K = (cam.scaled(w, h) if cam is not None else CameraConfig.nominal(w, h)).K
    m = measure_rectangle(quad, K, ref, ref_m, floor_px=floor, aspect=args.aspect)
    rect = SceneRect(round(m.width_m, 3), round(m.height_m, 3), round(m.bottom_m, 3), 0.0, quad.tolist(), w, h,
                     str(args.image), ref_image=str(args.image),
                     measured_by=f"reference {ref_m:.3f} m; aspect {m.aspect:.3f} (residual {m.aspect_residual:.3f})",
                     aspect=round(m.aspect, 4))
    out = Path(args.out or scene_file_for(args.image))
    rect.save(out)
    print(f"rectangle: {m.width_m:.3f} x {m.height_m:.3f} m, bottom edge {m.bottom_m:.3f} m above the floor "
          f"(aspect {m.aspect:.3f}, residual {m.aspect_residual:.3f}, lens {'calibrated' if cam else 'nominal'})")
    print("saved", out)
    return 0


def cmd_transfer(args) -> int:
    """Carry a measured rectangle from its reference image into a clip by matching wall features."""
    import cv2

    from .measure import refine_corners, transfer_corners

    ref = SceneRect.load(args.scene)
    if not ref.ref_image:
        print("the scene file has no ref_image; measure from an image first")
        return 1
    ref_img, _ = _load_image(ref.ref_image)
    img, _ = _load_image(args.clip, args.at)
    h, w = img.shape[:2]
    quad, inl, err = transfer_corners(ref_img, ref.corners_for(*ref_img.shape[1::-1]), img)
    if quad is None:
        print(f"transfer failed ({inl} matches): click the corners with `wallball corners {args.clip}` instead")
        return 1
    if args.snap:  # optional: pull corners onto the strongest nearby corner response
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        quad = refine_corners(gray, quad)
    out_rect = SceneRect(ref.width_m, ref.height_m, ref.bottom_m, ref.x0_m, quad.tolist(), w, h, str(args.clip),
                         f_px=ref.f_px, ref_image=ref.ref_image, measured_by=ref.measured_by + f"; transferred ({inl} inliers, {err:.1f} px)",
                         aspect=ref.aspect)
    out = Path(args.out or scene_file_for(args.clip))
    out_rect.save(out)
    print(f"transferred with {inl} inliers ({err:.1f} px): corners {np.round(quad).astype(int).tolist()} -> {out}")
    if args.check:
        view = img.copy()
        cv2.polylines(view, [np.round(quad).astype(np.int32).reshape(-1, 1, 2)], True, (255, 0, 255), 3)
        cv2.imwrite(args.check, view)
        print("check image", args.check)
    return 0


def cmd_mark(args) -> int:
    """Draw numbered points on an image (optionally a zoomed crop) to verify pixel coordinates by eye."""
    import cv2

    img, _ = _load_image(args.image, args.at)
    pts = _parse_pts(args.points) if args.points else []
    view = img.copy()
    for i, (x, y) in enumerate(pts):
        cv2.drawMarker(view, (int(round(x)), int(round(y))), (255, 0, 255), cv2.MARKER_CROSS, 40, 2)
        cv2.putText(view, str(i), (int(x) + 12, int(y) - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
    if args.crop:
        x0, y0, x1, y1 = [int(v) for v in args.crop.split(",")]
        view = view[max(0, y0):y1, max(0, x0):x1]
        if args.zoom != 1.0:
            view = cv2.resize(view, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_CUBIC)
    elif args.max_width and view.shape[1] > args.max_width:
        sc = args.max_width / view.shape[1]
        view = cv2.resize(view, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    cv2.imwrite(args.out, view)
    print("wrote", args.out, view.shape[1], "x", view.shape[0])
    return 0


def cmd_record(args) -> int:
    from .sources import list_avfoundation_devices, resolve_device, start_recording, stop_recording

    root = _root(args)
    app, _, _, _ = load_all(root)
    video, audio = list_avfoundation_devices()
    vdev = resolve_device(args.camera or app.camera_name, video)
    adev = None
    if not args.no_audio:
        try:
            adev = resolve_device(args.mic or app.audio_device, audio)
        except ValueError as e:
            print(f"audio: {e}; recording video only")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"recording {vdev!r} + {adev!r} at {app.width}x{app.height} @ {app.fps:g} -> {out}")
    print("clap once in view at the start; press Ctrl-C to stop" if not args.duration else f"stopping after {args.duration:g} s")
    proc = start_recording(vdev, adev, app.width, app.height, app.fps, out)
    try:
        if args.duration:
            time.sleep(args.duration)
        else:
            while proc.poll() is None:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_recording(proc)
    print("saved", out)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="wallball", description="Live wall-tennis coach")
    p.add_argument("--root", help="project root holding configs/, models/, markers/ (default: the package's parent)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("devices", help="list cameras/mics and the supported modes of the configured camera")
    s.add_argument("--camera")
    s.set_defaults(fn=cmd_devices)

    s = sub.add_parser("markers", help="generate the wall marker PDFs, the ChArUco board and configs/wall.yaml")
    s.add_argument("--paper", default="A4", choices=["A4", "A3", "A2", "LETTER"])
    s.add_argument("--dpi", type=int, default=200)
    s.set_defaults(fn=cmd_markers)

    s = sub.add_parser("calib-camera", help="one-time intrinsic calibration with the ChArUco board")
    s.add_argument("--file", help="calibrate from a recorded clip instead of the live camera")
    s.add_argument("--camera")
    s.add_argument("--backend", choices=["ffmpeg", "opencv"])
    s.add_argument("--square-cm", type=float, default=3.8, help="measured square size on the printed board")
    s.add_argument("--marker-cm", type=float, default=2.8)
    s.add_argument("--squares-x", type=int, default=5)
    s.add_argument("--squares-y", type=int, default=7)
    s.add_argument("--views", type=int, default=60)
    s.add_argument("--name", default="iphone")
    s.add_argument("--out")
    s.add_argument("--no-display", action="store_true")
    s.set_defaults(fn=cmd_calib_camera)

    s = sub.add_parser("export-models", help="download yolo11n-pose and export it to CoreML")
    s.add_argument("--imgsz", type=int, default=640)
    s.set_defaults(fn=cmd_export_models)

    for name, fn in (("live", cmd_live), ("replay", cmd_replay)):
        s = sub.add_parser(name, help="run the live dashboard" if name == "live" else "run the pipeline on a recorded clip")
        if name == "replay":
            s.add_argument("file")
            s.add_argument("--realtime", action="store_true", help="pace playback at the clip's real speed")
            s.add_argument("--start", type=float, default=0.0)
            s.add_argument("--end", type=float)
        else:
            s.add_argument("--camera")
            s.add_argument("--backend", choices=["ffmpeg", "opencv"])
        s.add_argument("--no-pose", action="store_true")
        s.add_argument("--no-display", action="store_true")
        s.add_argument("--out-video", help="write the dashboard video to this mp4")
        s.add_argument("--max-seconds", type=float, help="stop after this many seconds of footage")
        s.add_argument("--no-audio", action="store_true")
        s.add_argument("--no-ball", action="store_true")
        s.add_argument("--scene", help="clicked-corner scene file (replay default: <clip>.scene.yaml when present)")
        s.add_argument("--session-json", help="write the analysed strokes here (replay default: next to the clip)")
        s.set_defaults(fn=fn)

    s = sub.add_parser("corners", help="click the 4 corners of a measured rectangle on the wall (no markers needed)")
    s.add_argument("source", help="a clip path, or 'live'")
    s.add_argument("--at", type=float, default=0.5, help="clip time (s) of the frame to click on")
    s.add_argument("--width-m", type=float)
    s.add_argument("--height-m", type=float)
    s.add_argument("--bottom-m", type=float)
    s.add_argument("--camera")
    s.add_argument("--backend", choices=["ffmpeg", "opencv"])
    s.add_argument("--out")
    s.set_defaults(fn=cmd_corners)

    s = sub.add_parser("lens", help="focal length from one photo's metadata (no calibration board needed)")
    s.add_argument("photo")
    s.add_argument("--hfov", type=float, help="use this horizontal field of view (degrees) instead")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_lens)

    s = sub.add_parser("measure", help="metres of a wall rectangle from pixel coordinates and a reference object (no GUI)")
    s.add_argument("image", help="photo, or a clip (use --at)")
    s.add_argument("--at", type=float, default=0.5)
    s.add_argument("--corners", nargs=4, required=True, metavar="x,y", help="TL TR BR BL pixel coordinates")
    s.add_argument("--ref", nargs=2, required=True, metavar="x,y", help="two endpoints of the reference object")
    s.add_argument("--ref-m", type=float, help="reference length in metres (default: racket 0.686)")
    s.add_argument("--ball", action="store_true", help="the reference is the ball (0.067 m)")
    s.add_argument("--floor", nargs=1, metavar="x,y", help="a point on the wall-floor line (for the bottom height)")
    s.add_argument("--aspect", type=float, help="force the rectangle's width/height ratio")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_measure)

    s = sub.add_parser("transfer", help="carry a measured rectangle into a clip by matching wall features")
    s.add_argument("scene", help="scene yaml produced by `measure` (or `corners`) on a photo")
    s.add_argument("clip")
    s.add_argument("--at", type=float, default=0.5)
    s.add_argument("--out")
    s.add_argument("--check", help="write an image with the transferred rectangle drawn")
    s.add_argument("--snap", action="store_true", help="snap the transferred corners to nearby image corners")
    s.set_defaults(fn=cmd_transfer)

    s = sub.add_parser("mark", help="draw points on an image / clip frame to verify coordinates")
    s.add_argument("image")
    s.add_argument("--at", type=float, default=0.5)
    s.add_argument("--points", nargs="*", metavar="x,y")
    s.add_argument("--crop", help="x0,y0,x1,y1")
    s.add_argument("--zoom", type=float, default=1.0)
    s.add_argument("--max-width", type=int, default=1600)
    s.add_argument("--out", default="marked.png")
    s.set_defaults(fn=cmd_mark)

    s = sub.add_parser("record", help="record a raw session (video + iPhone mic) straight to disk with ffmpeg")
    s.add_argument("out")
    s.add_argument("--camera")
    s.add_argument("--mic")
    s.add_argument("--no-audio", action="store_true")
    s.add_argument("--duration", type=float)
    s.set_defaults(fn=cmd_record)

    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
