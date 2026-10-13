import numpy as np
import pytest

from tests.synth import synthetic_camera
from tests.synth_clip import VIEWS, make_clip
from wallball.app import Pipeline
from wallball.config import AppConfig, PlayerConfig, WallConfig
from wallball.sources import FileSource


@pytest.mark.parametrize("view", ["behind", "side_right", "ground"])
def test_pipeline_on_synthetic_clip(tmp_path, view):
    clip = tmp_path / f"synthetic_{view}.mov"
    truth = make_clip(clip, view)
    cam = synthetic_camera()
    pipe = Pipeline(AppConfig(display_width=960), WallConfig.default(), cam, PlayerConfig(), enable_pose=False,
                    display=False)
    n_onsets = pipe.attach_audio_file(clip)
    assert n_onsets == 2 * len(truth), n_onsets
    pipe.run(FileSource(clip))
    strokes = pipe.store.strokes
    assert len(strokes) == len(truth), [(s.idx, s.t_pop, s.notes) for s in strokes]
    placed = [s for s in strokes if s.wall_xy is not None]
    assert len(placed) >= len(truth) - 1, [(s.idx, s.notes) for s in strokes]
    for s, tr in zip(strokes, truth):
        assert abs(s.t_pop - tr.t_pop) < 0.03
        if s.wall_xy is not None:
            assert abs(s.wall_xy[0] - tr.p_wall[0]) < 0.12 and abs(s.wall_xy[1] - tr.p_wall[1]) < 0.12, (s.wall_xy, tr.p_wall)
        if s.avg_kmh is not None and s.speed_method == "fit":
            assert abs(s.avg_kmh - tr.avg_speed * 3.6) / (tr.avg_speed * 3.6) < 0.05, (s.avg_kmh, tr.avg_speed * 3.6, s.notes)
            assert abs(s.launch_kmh - tr.launch_speed * 3.6) / (tr.launch_speed * 3.6) < 0.10  # drag correction adds ~4 %
    fitted = [s for s in strokes if s.speed_method == "fit"]
    assert len(fitted) >= len(truth) - 1, [(s.idx, s.speed_method, s.notes) for s in strokes]
    assert pipe.state.strokes == len(truth)
    assert pipe.state.peak_kmh is not None


def test_pipeline_with_measured_rectangle_instead_of_markers(tmp_path):
    """The no-printer path: a door frame with clicked corners anchors the scene."""
    import tests.synth_clip as sc
    from wallball.config import SceneRect

    clip = tmp_path / "synthetic_door.mov"
    truth = make_clip(clip, "side_right", markers=False, door=True)
    corners = sc.last_door_corners + np.random.default_rng(5).normal(0, 1.0, (4, 2))
    rect = SceneRect(**sc.DOOR, corners=corners.tolist(), frame_width=1920, frame_height=1440)
    cam = synthetic_camera()
    nominal = type(cam)(cam.width, cam.height, cam.fx * 1.15, cam.fy * 1.15, cam.cx, cam.cy, name="nominal")  # 15 % off
    pipe = Pipeline(AppConfig(display_width=960), WallConfig.default(), nominal, PlayerConfig(), enable_pose=False,
                    display=False, scene_rect=rect)
    pipe.attach_audio_file(clip)
    pipe.run(FileSource(clip))
    strokes = pipe.store.strokes
    assert len(strokes) == len(truth)
    placed = [s for s in strokes if s.wall_xy is not None]
    assert len(placed) >= len(truth) - 1, [(s.idx, s.notes) for s in strokes]
    for s, tr in zip(strokes, truth):
        if s.wall_xy is not None:
            assert abs(s.wall_xy[0] - tr.p_wall[0]) < 0.15 and abs(s.wall_xy[1] - tr.p_wall[1]) < 0.15, (s.wall_xy, tr.p_wall)
        if s.avg_kmh is not None and s.speed_method == "fit":
            assert abs(s.avg_kmh - tr.avg_speed * 3.6) / (tr.avg_speed * 3.6) < 0.08, (s.avg_kmh, tr.avg_speed * 3.6)
    fitted = [s for s in strokes if s.speed_method == "fit"]
    assert len(fitted) >= len(truth) - 2, [(s.idx, s.speed_method, s.notes) for s in strokes]
