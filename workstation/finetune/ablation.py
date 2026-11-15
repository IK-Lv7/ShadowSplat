"""Ablation: is data from the 3DGS reconstruction useful for fine-tuning the edge model?

  python -m workstation.finetune.ablation prepare          # build the real-image training sets
  # (then label them with the teacher: see the commands printed by `prepare`)
  python -m workstation.finetune.ablation train            # train every arm x seed

Arms (same teacher, same recipe, same number of optimizer steps; only the images differ):
  real        real camera frames at the same 71 time steps as the renders
  real_lr     the same frames downsampled to the render resolution (isolates the effect of render sharpness)
  render_0    renders at the original camera pose
  render_all  renders at 0, +-1, +-2 m lateral offsets (the setting used in phase 4)
Runs that already exist are skipped (re-runnable).
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models.segformer import ROOT, build, load_cfg

log = logging.getLogger("ablation")


def job_dir(cfg: dict) -> Path:
    return Path(sorted(glob.glob(str(ROOT / cfg["job_glob"])))[0])


def train_frames(job: Path, cfg: dict) -> list[int]:
    man = json.loads((job / "render/manifest.json").read_text())
    last = int(cfg["train_frame_fraction"] * max(man["frames"]))
    return [f for f in man["frames"] if f <= last]


def arm_items(job: Path, cfg: dict, arm: str) -> list[tuple[Path, Path]]:
    ab = job / "ablation"
    pl = yaml.safe_load((ROOT / "configs/phase4_finetune.yaml").read_text())["pseudo_label"]["out_subdir"]
    fr = train_frames(job, cfg)
    if arm == "real":
        return [(ab / "real" / f"{f:04d}.png", ab / "pseudo_real" / f"{f:04d}.png") for f in fr]
    if arm == "real_lr":
        return [(ab / "real_lr" / f"{f:04d}.png", ab / "pseudo_real_lr" / f"{f:04d}.png") for f in fr]
    offs = ["offset_+0.0"] if arm == "render_0" else sorted(p.name for p in (job / "render").glob("offset_*"))
    return [(job / "render" / o / f"frame_{f:04d}.png", job / pl / o / f"frame_{f:04d}.png") for o in offs for f in fr]


def prepare(cfg: dict) -> None:
    job = job_dir(cfg)
    info = json.loads((job / "extract.json").read_text())
    src = ROOT / "data/phase0/processed_10Hz/mini" / f"{info['scene_idx']:03d}" / "images"
    ab = job / "ablation"
    (ab / "real").mkdir(parents=True, exist_ok=True)
    (ab / "real_lr").mkdir(parents=True, exist_ok=True)
    w, h = cfg["render_size"]
    for f in train_frames(job, cfg):
        im = Image.open(src / f"{f:03d}_0.jpg").convert("RGB")   # cam 0 = front camera, same frame as the render
        if not (ab / "real" / f"{f:04d}.png").exists():
            im.save(ab / "real" / f"{f:04d}.png")
        if not (ab / "real_lr" / f"{f:04d}.png").exists():
            im.resize((w, h), Image.BILINEAR).save(ab / "real_lr" / f"{f:04d}.png")
    n = len(train_frames(job, cfg))
    print(f"prepared {n} real frames in {ab}")
    print("label them with the teacher (segformer env, SEGFORMER_PYTHON):")
    for name in ("real", "real_lr"):
        print(f"  $SEGFORMER_PYTHON workstation/finetune/pseudo_label.py --src_dir {ab / name} --dst_dir {ab / ('pseudo_' + name)}")


class PairSet(Dataset):
    def __init__(self, items, size, jitter) -> None:
        self.items, self.size, self.jit = items, size, jitter

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        ip, lp = self.items[i]
        h, w = self.size
        img = np.asarray(Image.open(ip).convert("RGB").resize((w, h), Image.BILINEAR), np.float32)
        lab = np.asarray(Image.open(lp), np.int64)
        if random.random() < 0.5:
            img, lab = img[:, ::-1], lab[:, ::-1]
        img = np.clip(img * random.uniform(*self.jit), 0, 255)
        return torch.from_numpy(img.copy()).permute(2, 0, 1), torch.from_numpy(lab.copy())


def train_one(cfg: dict, items, seed: int, out: Path, size) -> None:
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    net = build(load_cfg(ROOT / "configs/phase1_model.yaml")).cuda()
    net.train()
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(PairSet(items, size, cfg["brightness_jitter"]), batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=8, drop_last=True, generator=g, persistent_workers=True,
                    worker_init_fn=lambda i: (random.seed(seed * 100 + i), np.random.seed(seed * 100 + i)))
    opt = torch.optim.AdamW(net.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    step, losses = 0, []
    while step < cfg["steps"]:
        for x, y in dl:
            x, y = x.cuda(), y.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(x)
            logits = F.interpolate(logits.float(), size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, y, ignore_index=255)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            step += 1
            if step >= cfg["steps"]:
                break
    torch.save(net.model.state_dict(), out / "b0_finetuned.pt")
    (out / "history.json").write_text(json.dumps({"steps": step, "first10_loss": float(np.mean(losses[:10])),
                                                  "last10_loss": float(np.mean(losses[-10:]))}, indent=1))
    log.info("saved %s (loss %.3f -> %.3f)", out, np.mean(losses[:10]), np.mean(losses[-10:]))


def train(cfg: dict) -> None:
    job = job_dir(cfg)
    mcfg = load_cfg(ROOT / "configs/phase1_model.yaml")
    size = (mcfg["model"]["input_height"], mcfg["model"]["input_width"])
    for arm in cfg["arms"]:
        items = arm_items(job, cfg, arm)
        missing = [p for _, p in items if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{arm}: {len(missing)} labels missing, e.g. {missing[0]} (run the teacher first)")
        for seed in cfg["seeds"]:
            out = ROOT / cfg["out_dir"] / f"{arm}_s{seed}"
            if (out / "b0_finetuned.pt").exists():
                log.info("skip %s", out.name)
                continue
            log.info("train %s seed=%d images=%d steps=%d", arm, seed, len(items), cfg["steps"])
            train_one(cfg, items, seed, out, size)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "train"])
    ap.add_argument("--config", type=Path, default=ROOT / "configs/phase4_ablation.yaml")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(a.config.read_text())
    {"prepare": prepare, "train": train}[a.cmd](cfg)


if __name__ == "__main__":
    main()
