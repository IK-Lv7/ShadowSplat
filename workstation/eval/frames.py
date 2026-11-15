"""Selection of the real frames used for evaluation (shared by ood_eval and teacher_real)."""
from __future__ import annotations

import json
from pathlib import Path

from workstation.extract.extract import scene_frames


def eval_frames(job: Path, cfg: dict, nuscenes_root: Path) -> dict[str, list[dict]]:
    info = json.loads((job / "extract.json").read_text())
    meta = nuscenes_root / "v1.0-mini"
    t0, t1 = info["scene_start_us"], info["scene_end_us"]
    t_eval = t0 + cfg["finetune"]["train_frame_fraction"] * (t1 - t0) + cfg["eval"]["heldout_gap_sec"] * 1e6
    return {
        "heldout": [f for f in scene_frames(meta, info["sceneId"], "CAM_FRONT") if f["timestamp"] >= t_eval],
        "forgetting": scene_frames(meta, cfg["eval"]["forgetting_scene"], "CAM_FRONT")[::cfg["eval"]["forgetting_stride"]],
    }
