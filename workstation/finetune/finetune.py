"""Fine-tunes SegFormer-B0 on rendered images + pseudo labels.

  python -m workstation.finetune.finetune --job_dir <job_dir>

Only renders from the first part of the scene (train_frame_fraction) are used; later real frames are kept for evaluation.
The original training data (Cityscapes) is not mixed in (forgetting is checked by the daytime agreement in eval).
"""
from __future__ import annotations

import argparse
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

log = logging.getLogger("finetune")


class RenderedSet(Dataset):
    def __init__(self, job: Path, cfg: dict, size: tuple[int, int], train: bool) -> None:
        pl = yaml.safe_load((ROOT / "configs/phase4_finetune.yaml").read_text())["pseudo_label"]
        man = json.loads((job / "render/manifest.json").read_text())
        last = int(cfg["train_frame_fraction"] * max(man["frames"]))
        self.items = [(job / "render" / r, job / pl["out_subdir"] / r) for r in man["images"]
                      if int(r.split("frame_")[1][:4]) <= last]
        self.size, self.train, self.jit = size, train, cfg["brightness_jitter"]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        ip, lp = self.items[i]
        h, w = self.size
        img = np.asarray(Image.open(ip).convert("RGB").resize((w, h), Image.BILINEAR), np.float32)
        lab = np.asarray(Image.open(lp), np.int64)
        if self.train:
            if random.random() < 0.5:
                img, lab = img[:, ::-1], lab[:, ::-1]
            img = np.clip(img * random.uniform(*self.jit), 0, 255)
        return torch.from_numpy(img.copy()).permute(2, 0, 1), torch.from_numpy(lab.copy())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job_dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/phase4_finetune.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())["finetune"]
    job = a.job_dir.resolve()
    out = ROOT / cfg["out_dir"] / job.name
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "train.log")])
    if (out / "b0_finetuned.pt").exists():
        log.info("exists, skip: %s", out / "b0_finetuned.pt")
        return
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    mcfg = load_cfg(ROOT / "configs/phase1_model.yaml")
    size = (mcfg["model"]["input_height"], mcfg["model"]["input_width"])
    net = build(mcfg).cuda()
    net.train()
    ds = RenderedSet(job, cfg, size, train=True)
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=8, drop_last=True)
    log.info("train images=%d (frames<=%.0f%% of scene) epochs=%d", len(ds), cfg["train_frame_fraction"] * 100,
             cfg["epochs"])
    opt = torch.optim.AdamW(net.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    hist = []
    for ep in range(cfg["epochs"]):
        tot, cnt = 0.0, 0
        for x, y in dl:
            x, y = x.cuda(), y.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(x)
            logits = F.interpolate(logits.float(), size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, y, ignore_index=255)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); cnt += 1
        hist.append({"epoch": ep + 1, "loss": tot / cnt})
        log.info("epoch %d loss %.4f", ep + 1, tot / cnt)
    torch.save(net.model.state_dict(), out / "b0_finetuned.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    log.info("saved %s", out / "b0_finetuned.pt")


if __name__ == "__main__":
    main()
