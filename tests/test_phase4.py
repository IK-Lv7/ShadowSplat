import numpy as np
from PIL import Image

from workstation.render.render_views import event_frame_range


def test_event_frame_range_maps_and_clips():
    info = {"scene_start_us": 0, "scene_end_us": 20_000_000, "event_start_us": 5_000_000, "event_end_us": 15_000_000}
    assert event_frame_range(info, 201) == (50, 150)
    info["event_start_us"], info["event_end_us"] = -1_000_000, 99_000_000  # margins outside the scene are clipped
    assert event_frame_range(info, 201) == (0, 200)


def test_miou_ignores_teacher_ignore_pixels(tmp_path):
    from workstation.eval.ood_eval import miou

    t = np.zeros((4, 4), np.uint8)
    t[:, 2:] = 1
    t[0, 0] = 255  # ignored pixel
    Image.fromarray(t).save(tmp_path / "1.png")
    p = t.copy()
    p[0, 0] = 7  # predictions on ignored pixels do not matter
    r = miou([p], tmp_path, [{"timestamp": 1}])
    assert r["pixel_acc_vs_teacher"] == 1.0 and r["miou_vs_teacher"] == 1.0
    p[1, 1] = 1  # one wrong pixel
    assert miou([p], tmp_path, [{"timestamp": 1}])["pixel_acc_vs_teacher"] < 1.0
