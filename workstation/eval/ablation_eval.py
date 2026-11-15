"""Evaluates the ablation runs (and the original B0) on ACDC val with real labels (mIoU only).

  python -m workstation.eval.ablation_eval

Writes <out_dir>/summary.json with per-run, per-condition mIoU, and prints mean +- std over seeds for every arm.
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

from models.ood_score import OodConfig
from models.segformer import ROOT, build, load_cfg
from workstation.eval.acdc_eval import N_CLS, miou_from_conf, run_split

log = logging.getLogger("ablation_eval")


def eval_model(net, root: Path, conditions: list[str], size, ood) -> dict:
    out, total = {}, np.zeros((N_CLS, N_CLS), np.int64)
    for c in conditions:
        imgs = sorted((root / "rgb_anon" / c / "val").glob("*/*_rgb_anon.png"))
        gts = [root / "gt" / c / "val" / p.parent.name / p.name.replace("_rgb_anon.png", "_gt_labelTrainIds.png")
               for p in imgs]
        _, conf = run_split(net, imgs, size, ood, gts)
        out[c] = round(miou_from_conf(conf)[0], 4)
        total += conf
    out["all"] = round(miou_from_conf(total)[0], 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/phase4_ablation.yaml")
    ap.add_argument("--acdc-root", type=Path, default=os.environ.get("ACDC_ROOT"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(a.config.read_text())
    conds = yaml.safe_load((ROOT / "configs/phase5_acdc.yaml").read_text())["conditions"]
    mcfg = load_cfg(ROOT / "configs/phase1_model.yaml")
    size = (mcfg["model"]["input_height"], mcfg["model"]["input_width"])
    ood = OodConfig.load()
    out_dir = ROOT / cfg["out_dir"]
    sp = out_dir / "summary.json"
    res: dict = json.loads(sp.read_text()) if sp.exists() else {}
    runs = {"original": None} | {f"{arm}_s{s}": out_dir / f"{arm}_s{s}" / "b0_finetuned.pt"
                                 for arm in cfg["arms"] for s in cfg["seeds"]}
    for name, ckpt in runs.items():
        if name in res:
            continue
        net = build(mcfg).cuda().eval()
        if ckpt:
            net.model.load_state_dict(torch.load(ckpt))
        res[name] = eval_model(net, Path(a.acdc_root), conds, size, ood)
        sp.write_text(json.dumps(res, indent=1))
        log.info("%s %s", name, res[name])
    print(f"{'arm':12s}" + "".join(f"{c:>10s}" for c in conds + ["all"]))
    print(f"{'original':12s}" + "".join(f"{res['original'][c]:>10.4f}" for c in conds + ["all"]))
    for arm in cfg["arms"]:
        rows = np.array([[res[f'{arm}_s{s}'][c] for c in conds + ["all"]] for s in cfg["seeds"]])
        print(f"{arm:12s}" + "".join(f"{m:>6.4f}±{sd:.4f}" if False else f"{m:>10.4f}" for m, sd in zip(rows.mean(0), rows.std(0))))
        print(f"{'  (std)':12s}" + "".join(f"{sd:>10.4f}" for sd in rows.std(0)))


if __name__ == "__main__":
    main()
