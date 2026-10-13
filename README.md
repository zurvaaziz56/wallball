# wallball

A live practice coach for hitting a tennis ball against a wall. A phone camera films the player;
a laptop shows, in real time, a skeleton on the player, the ball's trail, numbered impact marks
pinned to the wall, and a dashboard with stroke count, forehand / backhand split, launch speed
(current, peak and per-stroke-type average), a per-stroke speed chart and a shot-placement map.

Built with Claude Code.

## The problem it solves

Wall practice is the cheapest way to groove a stroke, but it gives no feedback: no speed, no
record of where the ball landed on the wall, no count of forehands versus backhands. Radar guns
and court-tracking systems are expensive and assume a full court. wallball turns a phone and a
laptop into that feedback loop, using only the wall itself as the measuring frame.

The camera can sit anywhere the wall anchor is visible: behind the player, on either side, on a
tripod, on the floor, or handheld. Distance to the wall is worked out per shot, so no fixed
camera position is required.

All processing runs locally on your own devices. No server, no upload.

## How it works

1. **Scene anchor.** Every frame, the camera pose is solved in a wall-anchored world frame
   (metres along the wall, height above the floor, distance out from the wall). The anchor is
   either printed ArUco markers or one measured rectangle on the wall (see Calibration).
2. **Audio events.** The microphone hears the racket "pop" and the wall "thud" with
   millisecond timing. Pops and thuds are paired into strokes, with timing windows derived from
   the player's distance to the wall.
3. **Ball tracking.** The ball is found by frame differencing (a background model when the
   camera is still, motion-compensated differencing when it moves) and tracked leg by leg.
   Reversals in the track are classified as wall hits or floor bounces.
4. **Pose.** Ultralytics YOLO11-pose gives a 17-point skeleton for the player.
5. **Strokes.** Forehand or backhand is decided from viewpoint-weighted cues on the skeleton.
   Launch speed comes from fitting the outbound flight to the known wall impact point and time
   (with drag and sound-delay corrections). Numbers that cannot be trusted are shown as a dash
   rather than a guess.

## Stack

- Python 3.12, managed with [uv](https://docs.astral.sh/uv/)
- OpenCV (ArUco markers, optical flow, rendering), NumPy, SciPy
- Ultralytics YOLO11-pose (PyTorch, optionally exported to CoreML for the Apple Neural Engine)
- PyAV and ffmpeg (video decode, live capture, audio extraction), sounddevice (live microphone)
- Pillow, matplotlib, PyYAML, coremltools
- pytest

## Requirements

- macOS for live capture and recording. Live capture uses ffmpeg's AVFoundation input and
  `record` uses Apple's hardware H.264 encoder. Replay of recorded clips and the test suite only
  need Python and ffmpeg. Development and testing happened on macOS on Apple Silicon.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (it installs Python 3.12 for
  you).
- `ffmpeg` and `ffprobe` on your `PATH` (for example `brew install ffmpeg`).
- A camera. An iPhone through Continuity Camera works out of the box. Any phone webcam app, or
  any other camera that shows up as a macOS video device, works too: pick it with `--camera`
  (see Camera setup).

## Install

```bash
git clone <this repo> wallball
cd wallball
uv sync                         # creates .venv with every dependency (pytest included)
uv run wallball --help          # lists all commands
```

### Model weights (not included)

The pose model is third-party and is not shipped in this repository.

- The code looks for **`models/yolo11n-pose.mlpackage`** (a CoreML export) first, and falls back
  to **`yolo11n-pose.pt`** (the Ultralytics YOLO11 nano pose checkpoint) in the current working
  directory. If the `.pt` file is missing, the Ultralytics library downloads it automatically
  from the Ultralytics GitHub releases on first use.
- To create the CoreML package (recommended on a Mac; it runs on the Neural Engine):

  ```bash
  uv run wallball export-models   # downloads yolo11n-pose.pt, writes models/yolo11n-pose.mlpackage
  ```

- On other systems, skip the export: the `.pt` file is used on Apple's MPS backend when present,
  otherwise on the CPU.
- **License: the Ultralytics library and the YOLO11 models are licensed under AGPL-3.0.** Using
  or distributing software built on them brings AGPL-3.0 obligations unless you hold an
  Ultralytics Enterprise License. Check the terms at
  <https://www.ultralytics.com/license> before redistributing.
- The Ultralytics library collects anonymous usage statistics by default. Turn this off with
  `uv run yolo settings sync=False`.
- Pose is optional: every pipeline command accepts `--no-pose`, which skips the model entirely.

## Calibration

The app needs two things: the lens (focal length) and the wall anchor.

### Lens

Pick one:

```bash
uv run wallball lens wall.jpg            # focal length from a photo's 35 mm-equivalent metadata
uv run wallball lens wall.jpg --hfov 70  # or give the horizontal field of view in degrees
uv run wallball calib-camera --square-cm 3.8   # or a full ChArUco-board calibration
```

All three write `configs/camera.yaml`. `lens` takes a photo from the same phone and camera lens
you will film with. `calib-camera` needs the printed `markers/charuco_board.pdf` glued flat on
card: measure one square, run the command and move the board around the whole frame until 60
views are collected (`c` finishes early, `q` quits). Without any lens file the app assumes a
nominal 70-degree lens and says so.

### Wall anchor, option A: printed ArUco markers

```bash
uv run wallball markers          # writes markers/*.pdf, markers/charuco_board.pdf, markers/README.md
                                 # and configs/wall.yaml (default layout) if it does not exist
```

The `markers/` folder is generated output and is not committed; this command recreates it
(`--paper A4|A3|A2|LETTER`, `--dpi`).

1. Print every marker at **100 % scale**. Large markers come as tiles: cut on the crop marks and
   butt them together.
2. Tape them on the wall. The default layout (`configs/wall.yaml`) puts two 40 cm markers high in
   the corners and four 20 cm markers down the sides, around a 2.0 m x 1.5 m practice rectangle.
3. Measure each marker's real centre (X along the wall, height above the floor) and its real
   black-square size, and write them into `configs/wall.yaml`. Positive X is the player's right
   when facing the wall. About 1 cm accuracy is enough.

At least three markers spanning 2 m must be visible for a pose (both configurable in the yaml).

### Wall anchor, option B: a measured rectangle (no printer)

Any rectangle of known size on the wall works: a door frame, a window, a poster, the whole wall,
or four sticky notes at the corners of a rectangle you measure. Bigger is better: a door seen
from 7 m gives roughly 20 cm of camera-position uncertainty, a whole wall a few centimetres.

```bash
uv run wallball corners clip.mov          # click TL, TR, BR, BL, then type width, height, bottom edge
uv run wallball replay clip.mov           # picks up clip.scene.yaml automatically

uv run wallball corners live --out configs/scene_live.yaml
uv run wallball live --scene configs/scene_live.yaml
```

The clicked corners are tracked with optical flow, so a static camera never drifts and a slowly
moving one keeps working while the corners stay in view. Redo the clicks whenever the camera
moves to a new spot. The rectangle doubles as the placement map, and the app can refine the
focal length from it.

**No tape measure either?** Take one photo of the wall with a racket held flat against it (an
adult racket is 68.6 cm) or the ball (6.7 cm) in the wall plane. From that photo:

```bash
uv run wallball measure wall.jpg --corners TLx,y TRx,y BRx,y BLx,y --ref x1,y1 x2,y2 --floor x,y
uv run wallball mark wall.jpg --points 812,404 1630,410        # draw points to check pixel coordinates
uv run wallball transfer wall.scene.yaml clip.mov --check check.png
```

`measure` takes pixel coordinates of the rectangle corners and of the reference object's two
ends (`--ball` for the ball, `--ref-m` for any other length; `--floor` marks a point on the
wall-floor line to get the rectangle's height above the floor). `transfer` matches the wall's
features between the photo and a frame of the clip, so the corners do not need clicking on every
clip. If a clip has too little texture to match, fall back to `wallball corners`.

## Camera setup

- `uv run wallball devices` lists cameras and microphones and the modes the selected camera
  offers. The default camera is the first device whose name contains "iPhone" (iPhones offer
  1920x1440 at 60 fps). Choose another with `--camera <name or index>`, or set `camera_name`,
  `width`, `height` and `fps` in an optional `configs/app.yaml`.
- `--backend opencv` reads the camera through OpenCV instead of ffmpeg (device index from
  `camera_index` in `configs/app.yaml`, default 0).
- Continuity Camera: phone locked, landscape, stable, cable preferred. Turn off **Center Stage,
  Portrait and Studio Light** (Control Center on the Mac while the camera is in use). Center
  Stage re-crops the image and breaks calibration.
- Bright, non-flickering light matters more than resolution. Fresh yellow balls help.
- `configs/player.yaml` (optional): `handedness: right|left`, `height_m`, `two_handed_backhand`.

## Run

```bash
uv run wallball live                           # live dashboard (q quits, s saves a screenshot)
uv run wallball record recordings/s1.mov       # raw video + microphone to disk (Ctrl-C stops)
uv run wallball replay recordings/s1.mov       # same pipeline on a clip; writes s1_session.json
uv run wallball replay clip.mov --realtime --out-video out.mp4
```

Common flags for `live` and `replay`: `--no-pose`, `--no-ball`, `--no-audio`, `--no-display`,
`--max-seconds N`, `--out-video FILE`, `--scene FILE`, `--session-json FILE`. `replay` also
takes `--start` and `--end` (seconds). `--root DIR` (before the command) points at a different
folder holding `configs/`, `models/` and `markers/`.

At the start of a session, clap once in view (audio/video sync check) and hit a few shots while
watching the badge in the top-left corner. It should read something like
`BEHIND-RIGHT 1.6m high, 4.8m to wall, STATIC`, and the practice rectangle should stay pinned
to the wall.

### All commands

| Command | What it does |
| --- | --- |
| `devices` | list cameras, microphones and supported camera modes |
| `markers` | generate marker PDFs, the ChArUco board and `configs/wall.yaml` |
| `calib-camera` | ChArUco lens calibration (live or `--file clip`) |
| `lens` | focal length from a photo's metadata, or `--hfov` |
| `export-models` | download YOLO11n-pose and export it to CoreML |
| `corners` | click a measured rectangle's corners on a clip or the live view |
| `measure` | measure a wall rectangle from a photo plus a reference object |
| `transfer` | carry a measured rectangle into a clip by feature matching |
| `mark` | draw points on an image or clip frame to check coordinates |
| `live` | live dashboard |
| `replay` | run the pipeline on a recorded clip |
| `record` | record raw video and audio to disk |

## Try it without a camera

```bash
uv run python -m tests.synth_clip recordings/synthetic.mov --view side_right
uv run wallball replay recordings/synthetic.mov --no-pose
```

The first command renders a synthetic session (marker wall, flying ball, impact sounds) with
known ground truth. Views: `behind`, `side_right`, `side_left`, `ground`, `diagonal`.

## Tests

```bash
uv run pytest -q
```

74 tests: unit tests for audio onsets, ball tracking, event pairing, scene pose, the measured
rectangle and stroke analysis, plus end-to-end runs on synthetic clips from three viewpoints and
in measured-rectangle mode. The suite needs `ffmpeg` on the `PATH` but no model weights and no
real footage (pose is disabled in the end-to-end tests).

## Status and known limits

Working prototype (version 0.1.0).

- **Live capture on macOS: launch from Terminal.app** (or another app that holds camera and
  microphone permission). macOS grants camera access per app; the first run asks for camera and
  microphone access. Shells embedded in some other apps silently receive no camera frames.
- Live capture, `devices` and `record` are macOS-only (AVFoundation through ffmpeg). The OpenCV
  backend is the likely route on other systems but is untested there.
- Strokes are counted from the audio: each one starts at a racket pop heard by the microphone
  (a wall hit seen in the video can stand in for a missed thud). With `--no-audio`, no
  microphone, or a clip without a sound track, the ball trail and wall anchor still show but no
  strokes or speeds are counted. A noisy room can cause missed strokes.
- The ball detector is classical (frame differencing). Helpers to bootstrap labels and fine-tune a
  learned detector exist in `wallball/train/`, but a learned detector is not yet wired into the
  pipeline.
- Accuracy is verified against synthetic clips with known ground truth. There is no published
  benchmark on real footage yet.
- Measured-rectangle mode is only as accurate as the rectangle's size and the corner clicks.

## Project structure

```
wallball/
├── wallball/            the package
│   ├── cli.py           command-line entry point (`wallball <command>`)
│   ├── app.py           threaded pipeline: frames → scene → pose → ball → events → dashboard
│   ├── scene.py         camera pose from ArUco markers or a tracked measured rectangle
│   ├── measure.py       rectangle measurement from a photo and corner transfer to clips
│   ├── calib.py         marker / ChArUco PDF generation and lens calibration
│   ├── audio.py         racket-pop / wall-thud / floor-bounce onset detection
│   ├── events.py        pairs audio onsets into strokes; classifies ball-track reversals
│   ├── ball.py          ball detection and per-leg tracking
│   ├── pose.py          YOLO11-pose wrapper and keypoint helpers
│   ├── strokes.py       forehand/backhand cues, trajectory fit, speed and placement
│   ├── session.py       stroke store, dashboard statistics, session JSON export
│   ├── render.py        dashboard and wall-anchored overlays (OpenCV)
│   ├── sources.py       live camera (ffmpeg / OpenCV) and file sources, recording
│   ├── config.py        wall, camera, player, app and scene-rectangle configs
│   ├── export.py        model download and CoreML export
│   └── train/           label bootstrapping and ball-detector fine-tuning
├── tests/               pytest suite and the synthetic-clip generator
├── configs/wall.yaml    default marker layout (edit after measuring your wall)
├── pyproject.toml
└── uv.lock
```

Generated at run time and kept out of git: `markers/`, `models/`, `recordings/`,
`configs/camera.yaml`, `configs/player.yaml`, scene files (`*.scene.yaml`) and session JSON
files (`*_session.json`).

## License

No license file is included yet. The Ultralytics dependency is AGPL-3.0 (see Model weights), which
affects how software built on it can be distributed.
