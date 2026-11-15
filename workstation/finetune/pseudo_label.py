"""Creates pseudo labels for rendered images from SegFormer-B5 (teacher) predictions.

Run in the segformer environment (SEGFORMER_PYTHON).
  python workstation/finetune/pseudo_label.py --job_dir <job_dir>

Output: <job_dir>/pseudo/<offset>/frame_NNNN.png (uint8, 19 Cityscapes classes, ignore=255) and pseudo/stats.json
Low-confidence teacher pixels are ignored. Confidently wrong teacher pixels cannot be removed (known limitation).
Existing labels are skipped (re-runnable).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("pseudo_label")
MEAN, STD = np.array([123.675, 116.28, 103.53], np.float32), np.array([58.395, 57.12, 57.375], np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job_dir", type=Path, help="job directory (renders listed in render/manifest.json)")
    ap.add_argument("--src_dir", type=Path, help="alternative to --job_dir: label every *.png in this directory")
    ap.add_argument("--dst_dir", type=Path, help="output directory for --src_dir")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/phase4_finetune.yaml")
    a = ap.parse_args()
    if not a.job_dir and not (a.src_dir and a.dst_dir):
        ap.error("give --job_dir, or --src_dir and --dst_dir")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(a.config.read_text())["pseudo_label"]
    if a.src_dir:
        items = [(p, a.dst_dir / p.name) for p in sorted(a.src_dir.glob("*.png"))]
    else:
        job = a.job_dir.resolve()
        manifest = json.loads((job / "render/manifest.json").read_text())
        items = [(job / "render" / rel, job / cfg["out_subdir"] / rel) for rel in manifest["images"]]

    import sys
    import torch
    import torch.nn.functional as F
    from PIL import Image

    sf = Path(os.environ["SEGFORMER_REPO"])
    sys.path.insert(0, str(sf))
    from mmseg.apis import init_segmentor

    model = init_segmentor(str(sf / "local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py"),
                           os.environ["SEGFORMER_CKPT"], device="cuda:0").eval()
    th, tw = cfg["teacher_input"]
    lh, lw = cfg["label_size"]
    stats: dict[str, dict] = {}
    for src, dst in items:
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(src).convert("RGB").resize((tw, th), Image.BILINEAR)
        x = torch.from_numpy(((np.asarray(im, np.float32) - MEAN) / STD).transpose(2, 0, 1))[None].cuda()
        with torch.no_grad():
            logits = model.encode_decode(x, [dict(img_shape=(th, tw, 3), ori_shape=(th, tw, 3), pad_shape=(th, tw, 3),
                                                  scale_factor=1.0, flip=False)])
            prob = F.softmax(F.interpolate(logits, size=(lh, lw), mode="bilinear", align_corners=False), dim=1)
            conf, lab = prob.max(dim=1)
        lab = lab[0].cpu().numpy().astype(np.uint8)
        lab[(conf[0] < cfg["min_confidence"]).cpu().numpy()] = 255
        Image.fromarray(lab).save(dst)
        s = stats.setdefault(dst.parent.name, {"n": 0, "ignored": 0.0})
        s["n"] += 1
        s["ignored"] += float((lab == 255).mean())
    for k, v in stats.items():
        v["ignored_ratio"] = round(v.pop("ignored") / v["n"], 4)
        log.info("%s: %s", k, v)
    if not a.src_dir:
        (job / cfg["out_subdir"] / "stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
