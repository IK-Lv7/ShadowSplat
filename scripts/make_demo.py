"""Builds the demo video of the pipeline (detect -> automated job -> reconstruct -> evaluate).

  python scripts/make_demo.py

- ACDC images are not shown (non-commercial research license); only numbers are.
- nuScenes images are CC BY-NC-SA 4.0; the credit is shown in the video.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.segformer import build, load_cfg  # noqa: E402
from models.ood_score import OodConfig, extract_events, frame_score, pixel_energy  # noqa: E402
from workstation.extract.extract import scene_frames  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BG, FG, ACC, OODC, MUTED = (18, 20, 26), (235, 235, 240), (90, 170, 255), (255, 90, 80), (150, 150, 160)
CREDIT = "Real-world data: nuScenes (CC BY-NC-SA 4.0). Simulation: CARLA."


def font(sz: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD if bold else FONT, sz)


def canvas(size) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", size, BG)
    return im, ImageDraw.Draw(im)


def header(d: ImageDraw.ImageDraw, step: str, title: str, size) -> None:
    d.text((30, 20), step, font=font(20, True), fill=ACC)
    d.text((30, 46), title, font=font(28, True), fill=FG)
    d.text((30, size[1] - 28), CREDIT, font=font(14), fill=MUTED)


def fit(im: Image.Image, w: int, h: int) -> Image.Image:
    return im.convert("RGB").resize((w, h), Image.BILINEAR)


def label(d, xy, text, color=FG, sz=18):
    x, y = xy
    tw = d.textlength(text, font=font(sz, True))
    d.rectangle([x, y, x + tw + 12, y + sz + 10], fill=(0, 0, 0))
    d.text((x + 6, y + 4), text, font=font(sz, True), fill=color)


class Writer:
    def __init__(self, path: Path, size, fps: int, exe: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.p = subprocess.Popen([exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                   "-pix_fmt", "yuv420p", "-crf", "20", str(path)], stdin=subprocess.PIPE)
        self.n = 0

    def add(self, im: Image.Image, repeat: int = 1) -> None:
        b = np.asarray(im.convert("RGB")).tobytes()
        for _ in range(repeat):
            self.p.stdin.write(b)
            self.n += 1

    def close(self) -> None:
        self.p.stdin.close()
        self.p.wait()


def title_card(w: Writer, size, fps):
    im, d = canvas(size)
    d.text((80, 200), "ShadowSplat", font=font(64, True), fill=FG)
    d.text((80, 290), "Edge OOD detection -> 3D Gaussian Splatting reconstruction", font=font(28), fill=FG)
    d.text((80, 330), "-> model fine-tuning -> evaluation, as one automated pipeline", font=font(28), fill=FG)
    steps = ["1 Detect", "2 Sync (CloudKit)", "3 Reconstruct (OmniRe)", "4 Render & fine-tune", "5 Evaluate"]
    x = 80
    for s in steps:
        tw = d.textlength(s, font=font(20, True))
        d.rounded_rectangle([x, 430, x + tw + 24, 470], 8, outline=ACC, width=2)
        d.text((x + 12, 438), s, font=font(20, True), fill=ACC)
        x += tw + 44
    d.text((80, size[1] - 40), CREDIT, font=font(14), fill=MUTED)
    w.add(im, fps * 4)


@torch.no_grad()
def detect_segment(w: Writer, size, fps, cfg, net, mcfg, ood: OodConfig, root: Path):
    h_in, w_in = mcfg["model"]["input_height"], mcfg["model"]["input_width"]
    for scene in cfg["detect_scenes"]:
        frames = scene_frames(root / "v1.0-mini", scene, "CAM_FRONT")[::2][: cfg["detect_frames"]]
        scores = []
        for i in range(0, len(frames), 8):
            x = np.stack([np.asarray(Image.open(root / f["filename"]).convert("RGB").resize((w_in, h_in)), np.float32)
                          for f in frames[i:i + 8]])
            lg = net(torch.from_numpy(x).permute(0, 3, 1, 2).cuda()).float().cpu().numpy()
            scores += [frame_score(pixel_energy(l, ood.temperature), ood.top_k_percent) for l in lg]
        ts = [f["timestamp"] for f in frames]
        evs = extract_events(ts, scores, OodConfig(**{**vars(ood), "margin_sec": 0.0}))
        lo, hi = min(min(scores), ood.threshold) - 0.3, max(max(scores), ood.threshold) + 0.3
        px0, px1, py0, py1 = 60, size[0] - 60, 560, 660
        X = lambda i: px0 + (px1 - px0) * i / max(1, len(frames) - 1)
        Y = lambda s: py1 - (py1 - py0) * (s - lo) / (hi - lo)
        for i, f in enumerate(frames):
            im, d = canvas(size)
            header(d, "1  DETECT  (edge model: SegFormer-B0)", f"Energy-based OOD score, front camera ({scene})", size)
            im.paste(fit(Image.open(root / f["filename"]), 760, 428), (60, 110))
            # show the event only once min_frames consecutive frames have been reached
            inside = any(e.start_us <= ts[i] <= e.end_us and i - next(j for j, t in enumerate(ts) if t >= e.start_us)
                         >= ood.min_frames - 1 for e in evs)
            s = scores[i]
            d.text((850, 130), "frame score", font=font(20), fill=MUTED)
            d.text((850, 158), f"{s:+.2f}", font=font(48, True), fill=OODC if s > ood.threshold else FG)
            d.text((850, 230), f"threshold {ood.threshold:+.2f}", font=font(20), fill=MUTED)
            d.text((850, 262), "(tuned in CARLA)", font=font(16), fill=MUTED)
            st, col = ("OOD EVENT", OODC) if inside else ("in distribution", (120, 210, 140))
            d.rounded_rectangle([850, 320, 1220, 380], 10, fill=col)
            d.text((866, 334), st, font=font(26, True), fill=BG)
            d.text((850, 400), "Only metadata (scene, time range,", font=font(16), fill=MUTED)
            d.text((850, 422), "score, thumbnail) is sent to CloudKit.", font=font(16), fill=MUTED)
            # timeline
            d.rectangle([px0, py0, px1, py1], outline=(60, 60, 70))
            for e in evs:  # shade only the part already played
                a = next(j for j, t in enumerate(ts) if t >= e.start_us)
                b = min(i, max(j for j, t in enumerate(ts) if t <= e.end_us))
                if a <= i:
                    d.rectangle([X(a), py0, X(b), py1], fill=(70, 30, 30))
            d.line([(px0, Y(ood.threshold)), (px1, Y(ood.threshold))], fill=OODC, width=1)
            d.line([(X(j), Y(v)) for j, v in enumerate(scores[: i + 1])], fill=ACC, width=3)
            d.line([(X(i), py0), (X(i), py1)], fill=FG, width=1)
            d.text((px0, py0 - 24), "frame score over time (red: threshold, shaded: detected event)", font=font(15), fill=MUTED)
            w.add(im)


def recon_segment(w: Writer, size, fps, cfg):
    for scene, idx in zip(cfg["recon_scenes"], ["000", "007"]):
        job = Path(glob.glob(str(ROOT / f"data/phase2/jobs/{scene}_*"))[0])
        frames = json.loads((job / "render/manifest.json").read_text())["frames"][: cfg["recon_frames"]]
        for f in frames:
            im, d = canvas(size)
            header(d, "3  RECONSTRUCT  (OmniRe / 3D Gaussian Splatting)",
                   f"Real camera vs. rendered novel views ({scene})", size)
            tiles = [(Image.open(ROOT / f"data/phase0/processed_10Hz/mini/{idx}/images/{f:03d}_0.jpg"), "real camera"),
                     (Image.open(job / f"render/offset_+0.0/frame_{f:04d}.png"), "render: same pose"),
                     (Image.open(job / f"render/offset_-1.0/frame_{f:04d}.png"), "render: 1 m left (never captured)"),
                     (Image.open(job / f"render/offset_+1.0/frame_{f:04d}.png"), "render: 1 m right (never captured)")]
            for k, (t, name) in enumerate(tiles):
                x, y = 60 + (k % 2) * 590, 105 + (k // 2) * 285
                im.paste(fit(t, 570, 270), (x, y))
                label(d, (x + 8, y + 8), name, color=FG if k == 0 else ACC, sz=16)
            w.add(im)


def automation_segment(w: Writer, size, fps):
    rows = [("detected", "edge event created (sceneId, time range, score)", "edge runtime"),
            ("queued", "watcher picks it up from CloudKit Public DB", "workstation"),
            ("extracting", "locate the event interval in the original log", "~1 s"),
            ("reconstructing", "preprocess, masks, OmniRe training (30k steps)", "~70 min"),
            ("rendered", "novel-view rendering + teacher pseudo labels", "~2 min"),
            ("finetuned", "fine-tune the edge model (SegFormer-B0)", "~1 min"),
            ("evaluated", "threshold re-tuned in CARLA, real-image eval", "run manually")]
    for n in range(1, len(rows) + 1):
        im, d = canvas(size)
        header(d, "2-4  AUTOMATED JOB  (no dedicated backend server)",
               "Status synced via CloudKit; the workstation advances it", size)
        for i, (st, desc, t) in enumerate(rows[:n]):
            y = 130 + i * 72
            d.rounded_rectangle([60, y, 300, y + 52], 10, fill=ACC if i == n - 1 else (45, 50, 62))
            d.text((76, y + 12), st, font=font(22, True), fill=BG if i == n - 1 else FG)
            d.text((330, y + 14), desc, font=font(20), fill=FG)
            d.text((1040, y + 14), t, font=font(18), fill=MUTED)
            if i < n - 1:
                d.line([(180, y + 52), (180, y + 72)], fill=MUTED, width=2)
        d.text((60, 650), "Measured on scene-1077 (RTX A6000). Times are approximate.", font=font(15), fill=MUTED)
        w.add(im, fps)
    w.add(im, fps * 2)


def eval_segment(w: Writer, size, fps, cfg, net, mcfg, ood: OodConfig):
    h_in, w_in = mcfg["model"]["input_height"], mcfg["model"]["input_width"]
    base = json.loads((ROOT / "data/phase5/eval/base.json").read_text())["by_k"][str(float(ood.top_k_percent))]
    acdc = json.loads((ROOT / "data/phase5/eval/acdc_base.json").read_text())["conditions"]
    tiles = []
    for c in cfg["carla_examples"]:
        p = sorted(glob.glob(str(ROOT / f"data/phase5/carla/{c}/route0/frame_*.jpg")))[20]
        x = torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((w_in, h_in)), np.float32))
        with torch.no_grad():
            lg = net(x.permute(2, 0, 1)[None].cuda()).float()[0].cpu().numpy()
        tiles.append((Image.open(p), c, frame_score(pixel_energy(lg, ood.temperature), ood.top_k_percent)))
    im, d = canvas(size)
    header(d, "5  EVALUATE", "Threshold tuned in CARLA, checked on real images", size)
    for k, (t, c, s) in enumerate(tiles):
        x, y = 60 + (k % 2) * 370, 110 + (k // 2) * 230
        im.paste(fit(t, 350, 197), (x, y))
        flag = s > ood.threshold
        label(d, (x + 6, y + 6), f"{c}  {s:+.2f}", color=OODC if flag else (120, 210, 140), sz=15)
    X0 = 830
    d.text((X0, 110), "CARLA (ID: day, OOD: night/fog/rain/low sun)", font=font(17, True), fill=FG)
    d.text((X0, 138), f"AUROC {base['pooled']['auroc']:.3f}   threshold {base['threshold']:+.2f}", font=font(17), fill=FG)
    d.text((X0, 166), "frame score = mean energy over all pixels", font=font(15), fill=MUTED)
    d.text((X0, 220), "Real adverse images vs. same-place normal", font=font(17, True), fill=FG)
    d.text((X0, 244), "(public adverse-condition dataset, val split)", font=font(15), fill=MUTED)
    for i, c in enumerate(["night", "fog", "rain", "snow"]):
        a = acdc[c]["ood_auroc_vs_ref"]
        y = 280 + i * 46
        d.text((X0, y), c, font=font(18), fill=FG)
        d.rectangle([X0 + 80, y + 2, X0 + 80 + 300, y + 24], outline=(60, 60, 70))
        d.rectangle([X0 + 80, y + 2, X0 + 80 + int(300 * a), y + 24], fill=ACC)
        d.text((X0 + 390, y), f"{a:.3f}", font=font(18, True), fill=FG)
    d.text((X0, 480), "OOD detection AUROC (higher is better)", font=font(15), fill=MUTED)
    w.add(im, fps * 7)


def end_card(w: Writer, size, fps):
    im, d = canvas(size)
    d.text((80, 220), "ShadowSplat", font=font(56, True), fill=FG)
    for i, t in enumerate(["Lightweight trigger at the edge, heavy data stays on the workstation",
                           "Real-world logs reconstructed, the edge model is the loop target",
                           "Every stage is a resumable CLI; status is synced via CloudKit"]):
        d.text((80, 310 + i * 40), "- " + t, font=font(24), fill=FG)
    d.text((80, size[1] - 40), CREDIT, font=font(14), fill=MUTED)
    w.add(im, fps * 4)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load((ROOT / "configs/phase6_demo.yaml").read_text())
    size, fps = tuple(cfg["size"]), cfg["fps"]
    mcfg = load_cfg(ROOT / "configs/phase1_model.yaml")
    net = build(mcfg).cuda().eval()
    ood = OodConfig.load()
    root = Path(os.environ["NUSCENES_RAW"])
    w = Writer(ROOT / cfg["out"], size, fps, os.environ[cfg["ffmpeg"]])
    try:
        title_card(w, size, fps)
        detect_segment(w, size, fps, cfg, net, mcfg, ood, root)
        automation_segment(w, size, fps)
        recon_segment(w, size, fps, cfg)
        eval_segment(w, size, fps, cfg, net, mcfg, ood)
        end_card(w, size, fps)
    finally:
        w.close()
    print(f"wrote {ROOT / cfg['out']} ({w.n} frames, {w.n / fps:.1f} s)")


if __name__ == "__main__":
    main()
