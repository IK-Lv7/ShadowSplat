"""Creates SegFormer-B5 (teacher) predictions for the real evaluation frames (run in the segformer env = SEGFORMER_PYTHON).

  python workstation/eval/teacher_real.py --job_dir <job_dir>
Output: <eval.out_dir>/teacher/<job>/<set>/<timestamp>.png (uint8, ignore=255)
These are teacher predictions, not ground truth; used as a proxy for how close the fine-tuned B0 is to the teacher.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from workstation.eval.frames import eval_frames  # noqa: E402

MEAN, STD = np.array([123.675, 116.28, 103.53], np.float32), np.array([58.395, 57.12, 57.375], np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job_dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/phase4_finetune.yaml")
    ap.add_argument("--nuscenes-root", type=Path, default=os.environ.get("NUSCENES_RAW"))
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    pl = cfg["pseudo_label"]
    job = a.job_dir.resolve()
    root = Path(a.nuscenes_root)

    import torch
    import torch.nn.functional as F
    from PIL import Image

    sf = Path(os.environ["SEGFORMER_REPO"])
    sys.path.insert(0, str(sf))
    from mmseg.apis import init_segmentor

    model = init_segmentor(str(sf / "local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py"),
                           os.environ["SEGFORMER_CKPT"], device="cuda:0").eval()
    th, tw = pl["teacher_input"]
    lh, lw = pl["label_size"]
    for name, frames in eval_frames(job, cfg, root).items():
        out = ROOT / cfg["eval"]["out_dir"] / "teacher" / job.name / name
        out.mkdir(parents=True, exist_ok=True)
        for f in frames:
            dst = out / f"{f['timestamp']}.png"
            if dst.exists():
                continue
            im = Image.open(root / f["filename"]).convert("RGB").resize((tw, th), Image.BILINEAR)
            x = torch.from_numpy(((np.asarray(im, np.float32) - MEAN) / STD).transpose(2, 0, 1))[None].cuda()
            with torch.no_grad():
                meta = [dict(img_shape=(th, tw, 3), ori_shape=(th, tw, 3), pad_shape=(th, tw, 3),
                             scale_factor=1.0, flip=False)]
                prob = F.softmax(F.interpolate(model.encode_decode(x, meta), size=(lh, lw), mode="bilinear",
                                               align_corners=False), dim=1)
                conf, lab = prob.max(dim=1)
            lab = lab[0].cpu().numpy().astype(np.uint8)
            lab[(conf[0] < pl["min_confidence"]).cpu().numpy()] = 255
            Image.fromarray(lab).save(dst)
        print(name, len(frames))


if __name__ == "__main__":
    main()
