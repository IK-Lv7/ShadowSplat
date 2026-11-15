"""AUROC per top-k% setting on nuScenes mini day vs. night (proxy labels).

Checks whether the k chosen in CARLA also holds on real images. Night = scenes whose description contains "Night".
  python -m workstation.eval.nuscenes_k_check
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import yaml

from models.segformer import ROOT, build, load_cfg
from models.ood_score import OodConfig
from workstation.eval.carla_threshold import auroc, score_all
from workstation.extract.extract import scene_frames

log = logging.getLogger("nuscenes_k_check")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load((ROOT / "configs/phase5_carla.yaml").read_text())["eval"]
    root = Path(os.environ["NUSCENES_RAW"])
    meta = root / "v1.0-mini"
    mcfg = load_cfg(ROOT / "configs/phase1_model.yaml")
    size = (mcfg["model"]["input_height"], mcfg["model"]["input_width"])
    ood = OodConfig.load()
    ks = sorted({float(k) for k in [*cfg["k_sensitivity"], ood.top_k_percent]})
    scenes = json.loads((meta / "scene.json").read_text())
    net = build(mcfg).cuda().eval()
    sc: dict[bool, dict[float, list[np.ndarray]]] = {False: {k: [] for k in ks}, True: {k: [] for k in ks}}
    for s in scenes:
        night = "night" in s["description"].lower()
        files = [root / f["filename"] for f in scene_frames(meta, s["name"], "CAM_FRONT")]
        r = score_all(net, files, size, ood, ks)
        for k in ks:
            sc[night][k].append(r[k])
    res = {}
    for k in ks:
        neg, pos = np.concatenate(sc[False][k]), np.concatenate(sc[True][k])
        res[str(k)] = {"auroc_night_vs_day": round(auroc(neg, pos), 4), "n_day": len(neg), "n_night": len(pos)}
        thr_file = ROOT / cfg["out_dir"] / "base.json"
        if thr_file.exists():  # false-positive / detection rates of the CARLA threshold (original B0) on real images
            thr = json.loads(thr_file.read_text())["by_k"][str(k)]["threshold"]
            res[str(k)].update({"carla_threshold": thr, "fpr_day@carla_thr": round(float((neg > thr).mean()), 4),
                                "tpr_night@carla_thr": round(float((pos > thr).mean()), 4),
                                "mean_day": round(float(neg.mean()), 4), "mean_night": round(float(pos.mean()), 4)})
        log.info("k=%s %s", k, res[str(k)])
    out = ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "nuscenes_k_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
