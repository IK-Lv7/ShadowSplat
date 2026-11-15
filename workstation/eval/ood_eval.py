"""Compares the OOD score (energy) and the predictions on real images before/after fine-tuning.

  python -m workstation.eval.ood_eval --job_dir <job_dir>

Evaluation sets:
  heldout : real frames from the later part of the target scene (not used for training); change in frame score.
  forgetting : real frames of a daytime scene; pixel agreement with the original B0 shows whether behaviour is kept.
Note: pseudo labels are teacher (B5) predictions, not ground truth; teacher_agreement measures agreement with the teacher.
Create the teacher predictions first with workstation/eval/teacher_real.py (segformer environment).
A lower energy only means higher confidence, not more correct predictions.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from models.segformer import ROOT, build, load_cfg
from models.ood_score import OodConfig, frame_score, pixel_energy
from workstation.eval.frames import eval_frames

log = logging.getLogger("ood_eval")


def load_batch(root: Path, frames: list[dict], size: tuple[int, int]) -> torch.Tensor:
    h, w = size
    x = np.stack([np.asarray(Image.open(root / f["filename"]).convert("RGB").resize((w, h), Image.BILINEAR),
                             np.float32) for f in frames])
    return torch.from_numpy(x).permute(0, 3, 1, 2)


@torch.no_grad()
def run(net, root: Path, frames: list[dict], size, ood: OodConfig, bs: int = 8):
    scores, preds = [], []
    for i in range(0, len(frames), bs):
        lg = net(load_batch(root, frames[i:i + bs], size).cuda()).float()
        logits = lg.cpu().numpy()
        scores += [frame_score(pixel_energy(l, ood.temperature), ood.top_k_percent) for l in logits]
        # upsample to input resolution before argmax (compare at the teacher label resolution)
        up = torch.nn.functional.interpolate(lg, size=size, mode="bilinear", align_corners=False)
        preds += [p.astype(np.uint8) for p in up.argmax(1).cpu().numpy()]
    return np.array(scores), preds


def miou(pred: list[np.ndarray], tdir: Path, frames: list[dict], n_cls: int = 19) -> dict:
    """Agreement with the teacher (B5) predictions, excluding teacher-ignored (255) pixels. Not ground truth."""
    inter, union = np.zeros(n_cls), np.zeros(n_cls)
    acc = []
    for p, f in zip(pred, frames):
        t = np.asarray(Image.open(tdir / f"{f['timestamp']}.png"))
        m = t != 255
        pm, tm = p[m], t[m]
        acc.append((pm == tm).mean())
        for c in range(n_cls):
            inter[c] += ((pm == c) & (tm == c)).sum()
            union[c] += ((pm == c) | (tm == c)).sum()
    ok = union > 0
    return {"miou_vs_teacher": round(float((inter[ok] / union[ok]).mean()), 4),
            "pixel_acc_vs_teacher": round(float(np.mean(acc)), 4)}


def summary(s: np.ndarray) -> dict:
    return {"n": int(len(s)), "mean": round(float(s.mean()), 4), "p90": round(float(np.percentile(s, 90)), 4),
            "max": round(float(s.max()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job_dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/phase4_finetune.yaml")
    ap.add_argument("--nuscenes-root", type=Path, default=os.environ.get("NUSCENES_RAW"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(a.config.read_text())
    job = a.job_dir.resolve()
    info = json.loads((job / "extract.json").read_text())
    root = Path(a.nuscenes_root)
    mcfg = load_cfg(ROOT / "configs/phase1_model.yaml")
    size = (mcfg["model"]["input_height"], mcfg["model"]["input_width"])
    ood = OodConfig.load()

    sets = eval_frames(job, cfg, root)
    held, day = sets["heldout"], sets["forgetting"]
    log.info("heldout frames=%d (%s), forgetting frames=%d (%s)", len(held), info["sceneId"], len(day),
             cfg["eval"]["forgetting_scene"])

    base = build(mcfg).cuda().eval()
    ft = build(mcfg).cuda().eval()
    ft.model.load_state_dict(torch.load(ROOT / cfg["finetune"]["out_dir"] / job.name / "b0_finetuned.pt"))

    res: dict = {"job": job.name, "config": str(a.config.name), "threshold": ood.threshold}
    for name, frames in (("heldout", held), ("forgetting", day)):
        sb, pb = run(base, root, frames, size, ood)
        sf, pf = run(ft, root, frames, size, ood)
        r = {"base": summary(sb), "finetuned": summary(sf),
             "frames_over_threshold": {"base": int((sb > ood.threshold).sum()), "finetuned": int((sf > ood.threshold).sum())},
             "mean_score_delta": round(float((sf - sb).mean()), 4),
             "pixel_agreement_with_base": round(float(np.mean([(a_ == b_).mean() for a_, b_ in zip(pb, pf)])), 4)}
        tdir = ROOT / cfg["eval"]["out_dir"] / "teacher" / job.name / name
        r["teacher_agreement"] = {"base": miou(pb, tdir, frames), "finetuned": miou(pf, tdir, frames)}
        res[name] = r
        log.info("%s: %s", name, r)
    out = ROOT / cfg["eval"]["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{job.name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
