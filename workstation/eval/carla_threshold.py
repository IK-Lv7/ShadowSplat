"""Measures OOD detection (AUROC, FPR@95TPR) on per-condition CARLA data and picks the threshold.

  python -m workstation.eval.carla_threshold

ID = daytime conditions (negatives), OOD = night, dense fog, heavy rain, low sun (positives). Frame level.
- AUROC: separability of OOD vs ID by the score.
- FPR@95TPR: false-positive rate on ID frames at the threshold that detects 95% of OOD frames.
- Threshold: smallest value with an ID false-positive rate of at most target_fpr (false alarms are the operational cost).
Updating the model shifts the score scale, so the threshold is picked per model.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from scipy.stats import rankdata

from models.segformer import ROOT, build, load_cfg
from models.ood_score import OodConfig, frame_score, pixel_energy

log = logging.getLogger("carla_threshold")


def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    """Probability that a positive scores higher than a negative (Mann-Whitney U)."""
    r = rankdata(np.concatenate([neg, pos]))
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def fpr_at_95tpr(neg: np.ndarray, pos: np.ndarray) -> float:
    th = np.percentile(pos, 5)
    return float((neg >= th).mean())


def pick_threshold(neg: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold whose ID false-positive rate (score > thr) is at most target_fpr."""
    return float(np.quantile(neg, 1 - target_fpr))


@torch.no_grad()
def score_all(net, files: list[Path], size, ood: OodConfig, ks: list[float], bs: int = 8) -> dict[float, np.ndarray]:
    h, w = size
    out: dict[float, list[float]] = {k: [] for k in ks}
    for i in range(0, len(files), bs):
        x = np.stack([np.asarray(Image.open(f).convert("RGB").resize((w, h), Image.BILINEAR), np.float32)
                      for f in files[i:i + bs]])
        logits = net(torch.from_numpy(x).permute(0, 3, 1, 2).cuda()).float().cpu().numpy()
        for l in logits:
            e = pixel_energy(l, ood.temperature)
            for k in ks:
                out[k].append(frame_score(e, k))
    return {k: np.array(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/phase5_carla.yaml")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(a.config.read_text())
    ev, root = cfg["eval"], ROOT / cfg["out_dir"]
    mcfg = load_cfg(ROOT / "configs/phase1_model.yaml")
    size = (mcfg["model"]["input_height"], mcfg["model"]["input_width"])
    ood = OodConfig.load()
    ks = sorted({float(k) for k in [*ev["k_sensitivity"], ood.top_k_percent]})

    files: dict[str, list[Path]] = {c: sorted((root / c).glob("route*/frame_*.jpg")) for c in cfg["conditions"]
                                    if (root / c / "meta.json").exists()}
    role = {c: cfg["conditions"][c]["role"] for c in files}
    log.info("frames: %s", {c: len(f) for c, f in files.items()})
    out_dir = ROOT / ev["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for mname, ckpt in ev["models"].items():
        net = build(mcfg).cuda().eval()
        if ckpt:
            net.model.load_state_dict(torch.load(ROOT / ckpt))
        sc = {c: score_all(net, f, size, ood, ks) for c, f in files.items()}
        res: dict = {"model": mname, "T": ood.temperature, "primary_k": ood.top_k_percent,
                     "target_fpr": ev["target_fpr"], "frames": {c: len(f) for c, f in files.items()}, "by_k": {}}
        for k in ks:
            neg = np.concatenate([sc[c][k] for c in sc if role[c] == "id"])
            r = {"id_mean": round(float(neg.mean()), 4), "threshold": round(pick_threshold(neg, ev["target_fpr"]), 4),
                 "per_condition": {}}
            pos_all = np.concatenate([sc[c][k] for c in sc if role[c] == "ood"])
            r["pooled"] = {"auroc": round(auroc(neg, pos_all), 4), "fpr@95tpr": round(fpr_at_95tpr(neg, pos_all), 4),
                           "tpr@thr": round(float((pos_all > r["threshold"]).mean()), 4),
                           "fpr@thr": round(float((neg > r["threshold"]).mean()), 4)}
            for c in sc:
                if role[c] == "ood":
                    p = sc[c][k]
                    r["per_condition"][c] = {"auroc": round(auroc(neg, p), 4), "fpr@95tpr": round(fpr_at_95tpr(neg, p), 4),
                                             "mean": round(float(p.mean()), 4),
                                             "tpr@thr": round(float((p > r["threshold"]).mean()), 4)}
            res["by_k"][str(k)] = r
        (out_dir / f"{mname}.json").write_text(json.dumps(res, indent=1))
        log.info("%s k=%s pooled=%s thr=%s", mname, ood.top_k_percent, res["by_k"][str(ood.top_k_percent)]["pooled"],
                 res["by_k"][str(ood.top_k_percent)]["threshold"])


if __name__ == "__main__":
    main()
