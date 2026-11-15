"""Edge runtime. Replays the nuScenes front camera and runs
SegFormer-B0 -> energy score -> threshold -> event extraction.

Outputs (per scene):
  <out>/<scene>/scores.csv      timestamp_us, frame_score
  <out>/<scene>/events.json     detected events (OODEvent metadata + thumbnail)
  <out>/<scene>/thumb_*.jpg     downscaled JPEG of the peak frame of each event
Finished scenes are skipped (re-runnable).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from models.segformer import ROOT, build, load_cfg
from workstation.extract.extract import scene_frames
from models.ood_score import OodConfig, extract_events, frame_score, pixel_energy

log = logging.getLogger("edge_sim")


def run_scene(net, ood: OodConfig, cfg: dict, mcfg: dict, root: Path, name: str, out: Path) -> None:
    if (out / "events.json").exists():
        log.info("skip %s (done)", name)
        return
    out.mkdir(parents=True, exist_ok=True)
    frames = scene_frames(root / "v1.0-mini", name, cfg["camera"])
    h, w = mcfg["input_height"], mcfg["input_width"]
    scores: list[float] = []
    ts = [f["timestamp"] for f in frames]
    if (out / "scores.csv").exists():  # reuse cached inference, so only the threshold step is recomputed
        scores = [float(r["frame_score"]) for r in csv.DictReader(open(out / "scores.csv"))]
    else:
        bs = cfg["batch_size"]
        for i in range(0, len(frames), bs):
            chunk = frames[i:i + bs]
            x = np.stack([np.asarray(Image.open(root / f["filename"]).convert("RGB").resize((w, h), Image.BILINEAR),
                                     dtype=np.float32) for f in chunk])
            with torch.no_grad():
                logits = net(torch.from_numpy(x).permute(0, 3, 1, 2)).numpy()
            scores += [frame_score(pixel_energy(l, ood.temperature), ood.top_k_percent) for l in logits]
    with open(out / "scores.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["timestamp_us", "frame_score"])
        wr.writerows(zip(ts, scores))
    events = []
    for k, ev in enumerate(extract_events(ts, scores, ood)):
        inside = [j for j, t in enumerate(ts) if ev.start_us <= t <= ev.end_us]
        peak = max(inside, key=lambda j: scores[j])
        im = Image.open(root / frames[peak]["filename"]).convert("RGB")
        im.thumbnail((cfg["thumbnail_width"], cfg["thumbnail_width"]))
        thumb = f"thumb_{k:02d}.jpg"
        im.save(out / thumb, quality=cfg["thumbnail_quality"])
        events.append({"sceneId": name, "startTimestamp": ev.start_us, "endTimestamp": ev.end_us,
                       "oodScore": ev.score, "thumbnail": thumb, "status": "detected"})
    (out / "events.json").write_text(json.dumps(events, indent=1))
    log.info("%s frames=%d score[min/mean/max]=%.2f/%.2f/%.2f events=%d", name, len(scores),
             min(scores), float(np.mean(scores)), max(scores), len(events))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/phase1_edge_sim.yaml")
    ap.add_argument("--scenes", nargs="*", help="scene names (default: all scenes)")
    ap.add_argument("--nuscenes-root", type=Path, default=os.environ.get("NUSCENES_RAW"))
    ap.add_argument("--threshold", type=float, help="override the threshold in phase1_ood.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    out_root = ROOT / cfg["out_dir"]
    out_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_root / "edge_sim.log")])
    if not args.nuscenes_root:
        ap.error("NUSCENES_RAW (.env) or --nuscenes-root is required")
    root = Path(args.nuscenes_root)
    ood = OodConfig.load()
    if args.threshold is not None:
        ood = OodConfig(**{**vars(ood), "threshold": args.threshold})
    mcfg = load_cfg(ROOT / "configs/phase1_model.yaml")
    net = build(mcfg)
    names = args.scenes or [s["name"] for s in json.loads((root / "v1.0-mini/scene.json").read_text())]
    for n in names:
        run_scene(net, ood, cfg, mcfg["model"], root, n, out_root / n)


if __name__ == "__main__":
    main()
