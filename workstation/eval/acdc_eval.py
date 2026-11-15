"""Evaluates the model before/after fine-tuning on ACDC (real night/fog/rain/snow images with 19-class labels).

  python -m workstation.eval.acdc_eval

1. Segmentation: mIoU against the real labels (labelTrainIds, ignore=255). Predictions are upsampled to label resolution.
2. OOD detection: AUROC with adverse images (val) as positives and same-place normal images (val_ref) as negatives,
   plus detection / false-positive rates at the per-model CARLA threshold.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from models.segformer import ROOT, build, load_cfg
from models.ood_score import OodConfig, frame_score, pixel_energy
from workstation.eval.carla_threshold import auroc

log = logging.getLogger("acdc_eval")
N_CLS = 19
CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation",
           "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]


def confusion(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    m = gt != 255
    return np.bincount(gt[m].astype(np.int64) * N_CLS + pred[m], minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)


def miou_from_conf(conf: np.ndarray) -> tuple[float, list[float | None]]:
    """Average over the classes present in the ground truth (absent classes are excluded)."""
    inter = np.diag(conf).astype(float)
    union = conf.sum(0) + conf.sum(1) - inter
    present = conf.sum(1) > 0
    iou = np.where(present, inter / np.maximum(union, 1), np.nan)
    return float(np.nanmean(iou)), [None if np.isnan(v) else round(float(v), 4) for v in iou]


@torch.no_grad()
def run_split(net, files: list[Path], size, ood: OodConfig, gts: list[Path] | None):
    h, w = size
    scores, conf = [], np.zeros((N_CLS, N_CLS), np.int64)
    for f, g in zip(files, gts or [None] * len(files)):
        im = Image.open(f).convert("RGB")
        x = torch.from_numpy(np.asarray(im.resize((w, h), Image.BILINEAR), np.float32)).permute(2, 0, 1)[None].cuda()
        lg = net(x).float()
        scores.append(frame_score(pixel_energy(lg[0].cpu().numpy(), ood.temperature), ood.top_k_percent))
        if g is not None:
            gt = np.asarray(Image.open(g))
            pred = F.interpolate(lg, size=gt.shape, mode="bilinear", align_corners=False).argmax(1)[0]
            conf += confusion(pred.cpu().numpy(), gt)
    return np.array(scores), conf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/phase5_acdc.yaml")
    ap.add_argument("--acdc-root", type=Path, default=os.environ.get("ACDC_ROOT"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(a.config.read_text())
    root, sp = Path(a.acdc_root), cfg["split"]
    mcfg = load_cfg(ROOT / "configs/phase1_model.yaml")
    size = (mcfg["model"]["input_height"], mcfg["model"]["input_width"])
    ood = OodConfig.load()
    out_dir = ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for mname, ckpt in cfg["models"].items():
        net = build(mcfg).cuda().eval()
        if ckpt:
            net.model.load_state_dict(torch.load(ROOT / ckpt))
        carla = json.loads((ROOT / cfg["carla_eval_dir"] / f"{mname}.json").read_text())
        thr = carla["by_k"][str(float(ood.top_k_percent))]["threshold"]
        res: dict = {"model": mname, "k": ood.top_k_percent, "carla_threshold": thr, "conditions": {}}
        total = np.zeros((N_CLS, N_CLS), np.int64)
        for c in cfg["conditions"]:
            imgs = sorted((root / "rgb_anon" / c / sp).glob("*/*_rgb_anon.png"))
            gts = [root / "gt" / c / sp / p.parent.name / p.name.replace("_rgb_anon.png", "_gt_labelTrainIds.png")
                   for p in imgs]
            refs = sorted((root / "rgb_anon" / c / f"{sp}_ref").glob("*/*_rgb_ref_anon.png"))
            s_adv, conf = run_split(net, imgs, size, ood, gts)
            s_ref, _ = run_split(net, refs, size, ood, None)
            total += conf
            mi, per = miou_from_conf(conf)
            res["conditions"][c] = {
                "n": len(imgs), "n_ref": len(refs), "miou": round(mi, 4), "iou_per_class": dict(zip(CLASSES, per)),
                "ood_auroc_vs_ref": round(auroc(s_ref, s_adv), 4),
                "score_mean": {"adverse": round(float(s_adv.mean()), 4), "ref": round(float(s_ref.mean()), 4)},
                "tpr@carla_thr": round(float((s_adv > thr).mean()), 4),
                "fpr_ref@carla_thr": round(float((s_ref > thr).mean()), 4)}
            log.info("%s %s: %s", mname, c, {k: v for k, v in res["conditions"][c].items() if k != "iou_per_class"})
        res["miou_all"] = round(miou_from_conf(total)[0], 4)
        (out_dir / f"acdc_{mname}.json").write_text(json.dumps(res, indent=1))
        log.info("%s mIoU(all)=%.4f", mname, res["miou_all"])


if __name__ == "__main__":
    main()
