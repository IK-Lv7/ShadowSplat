"""Renders novel views (laterally shifted front camera) of the event interval from a reconstructed scene.

Run in the drivestudio environment (DRIVESTUDIO_PYTHON). drivestudio is not modified; its public
DrivingDataset / trainer / prepare_novel_view_render_data are used as is.

  python workstation/render/render_views.py --job_dir <job_dir> --run_dir <training_run_dir>

Output: <job_dir>/render/offset_<m>/frame_<NNNN>.png and render/manifest.json
Existing images are skipped (re-runnable).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "third_party" / "drivestudio"
log = logging.getLogger("render")


def event_frame_range(info: dict, n_frames: int, margin_note: bool = True) -> tuple[int, int]:
    """Maps the event interval (us) linearly to 10 Hz reconstruction frame indices."""
    t0, t1 = info["scene_start_us"], info["scene_end_us"]
    f = lambda t: int(round((t - t0) / (t1 - t0) * (n_frames - 1)))
    a, b = f(max(info["event_start_us"], t0)), f(min(info["event_end_us"], t1))
    return max(0, a), min(n_frames - 1, b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job_dir", type=Path, required=True)
    ap.add_argument("--run_dir", type=Path, required=True, help="training run with checkpoint_final.pth and config.yaml")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/phase4_render.yaml")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg_r = yaml.safe_load(a.config.read_text())
    job_dir, run_dir = a.job_dir.resolve(), a.run_dir.resolve()
    info = json.loads((job_dir / "extract.json").read_text())
    out_root = job_dir / cfg_r["out_subdir"]

    sys.path.insert(0, str(DS))
    os.chdir(DS)
    import torch
    from omegaconf import OmegaConf
    from PIL import Image

    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str

    cfg = OmegaConf.load(run_dir / "config.yaml")
    device = torch.device("cuda")
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer, num_timesteps=dataset.num_img_timesteps, model_config=cfg.model,
        num_train_images=len(dataset.train_image_set), num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps, scene_aabb=dataset.get_aabb().reshape(2, 3), device=device)
    trainer.resume_from_checkpoint(ckpt_path=str(run_dir / "checkpoint_final.pth"), load_only_model=True)
    trainer.set_eval()

    cam = dataset.pixel_source.camera_data[cfg_r["camera_id"]]
    n = dataset.pixel_source.num_frames
    c2w_all = cam.cam_to_worlds  # (n, 4, 4)
    lo, hi = event_frame_range(info, n)
    frames = list(range(lo, hi + 1, cfg_r["frame_stride"]))
    log.info("scene frames=%d event frames=[%d,%d] -> render %d frames x %d offsets",
             n, lo, hi, len(frames), len(cfg_r["lateral_offsets_m"]))

    manifest = {"sceneId": info["sceneId"], "scene_idx": info["scene_idx"], "frames": frames,
                "offsets_m": cfg_r["lateral_offsets_m"], "images": []}
    for off in cfg_r["lateral_offsets_m"]:
        d = out_root / f"offset_{off:+.1f}"
        d.mkdir(parents=True, exist_ok=True)
        todo = [i for i in frames if not (d / f"frame_{i:04d}.png").exists()]
        manifest["images"] += [f"offset_{off:+.1f}/frame_{i:04d}.png" for i in frames]
        if not todo:
            log.info("offset %+.1f: done, skip", off)
            continue
        traj = c2w_all.clone()
        traj[:, :3, 3] += traj[:, :3, 0] * off  # shift along the camera x axis (right)
        with torch.no_grad():
            for i in todo:
                # prepare one frame at a time (do not hold the rays of all frames in memory).
                # prepare_novel_view_render_data derives frame_idx etc. from the trajectory length, so fix them to the real frame.
                fd = dataset.prepare_novel_view_render_data(traj[i:i + 1])[0]
                fd["image_infos"]["frame_idx"].fill_(i)
                fd["image_infos"]["img_idx"].fill_(i)
                fd["image_infos"]["normed_time"].fill_(i / (n - 1))
                for k, v in fd["cam_infos"].items():
                    fd["cam_infos"][k] = v.to(device)
                for k, v in fd["image_infos"].items():
                    fd["image_infos"][k] = v.to(device)
                out = trainer(image_infos=fd["image_infos"], camera_infos=fd["cam_infos"], novel_view=True)
                rgb = (out["rgb"].cpu().numpy().clip(1e-6, 1 - 1e-6) * 255).astype(np.uint8)
                Image.fromarray(rgb).save(d / f"frame_{i:04d}.png")
        log.info("offset %+.1f: rendered %d", off, len(todo))
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
