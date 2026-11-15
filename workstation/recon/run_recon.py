"""Wrapper around OmniRe (drivestudio). drivestudio itself is never modified.

Stages: preprocess -> masks -> train (each re-runnable on its own; skipped when the output exists)
Usage: python workstation/recon/run_recon.py --config configs/phase0_omnire_nuscenes_mini.yaml --stage preprocess
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "third_party" / "drivestudio"
log = logging.getLogger("recon")


def run(cmd: list[str]) -> None:
    log.info("run: %s", " ".join(cmd))
    env = {**os.environ, "PYTHONPATH": str(DS)}
    wsl_lib = "/usr/lib/wsl/lib"  # libcuda.so lives here on WSL2
    if Path(wsl_lib).exists():
        env["LD_LIBRARY_PATH"] = f"{wsl_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    subprocess.run(cmd, cwd=DS, env=env, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--stage", choices=["preprocess", "masks", "train"], required=True)
    p.add_argument("--data_root", type=Path, default=Path(os.environ.get("NUSCENES_RAW", "")))
    p.add_argument("--scene_idx", type=int, help="override scene_idx in the config (per-job reconstruction)")
    p.add_argument("--out_dir", type=Path, default=ROOT / "data" / "phase0")
    a = p.parse_args()
    a.out_dir = a.out_dir.resolve()  # cwd becomes drivestudio, so a relative path would write into the submodule

    cfg = yaml.safe_load(a.config.read_text())
    if a.scene_idx is not None:
        cfg["scene_idx"] = a.scene_idx
    # keep the original name for the config's own scene_idx so earlier runs are reused
    base_idx = yaml.safe_load(a.config.read_text())["scene_idx"]
    name = a.config.stem if cfg["scene_idx"] == base_idx else f"{a.config.stem}_s{cfg['scene_idx']:03d}"
    a.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.StreamHandler(), logging.FileHandler(log_dir / f"{name}_{a.stage}.log")])

    processed = a.out_dir / "processed"  # drivestudio creates processed_10Hz/<split_dir>
    data_root = a.out_dir / "processed_10Hz" / cfg["split_dir"]
    if a.stage == "preprocess":
        if (a.out_dir / "processed_10Hz" / cfg["split_dir"] / f"{cfg['scene_idx']:03d}" / "images").exists():
            log.info("processed exists, skip"); return
        run([sys.executable, "datasets/preprocess.py", "--data_root", str(a.data_root),
             "--target_dir", str(processed), "--dataset", cfg["dataset"], "--split", cfg["split"],
             "--start_idx", str(cfg["scene_idx"]), "--num_scenes", "1",
             "--interpolate_N", str(cfg["interpolate_n"]), "--workers", str(cfg["workers"]),
             "--process_keys", *cfg["process_keys"]])
    elif a.stage == "masks":
        # sky / fine dynamic masks; uses SegFormer-B5, so it runs in a separate conda env (SEGFORMER_PYTHON)
        scene_dir = data_root / f"{cfg['scene_idx']:03d}"
        if (scene_dir / "sky_masks").exists() and (scene_dir / "fine_dynamic_masks").exists():
            log.info("masks exist, skip"); return
        sf = Path(os.environ["SEGFORMER_REPO"])
        run([os.environ["SEGFORMER_PYTHON"], "datasets/tools/extract_masks.py",
             "--data_root", str(data_root), "--segformer_path", str(sf),
             "--checkpoint", os.environ["SEGFORMER_CKPT"],
             "--start_idx", str(cfg["scene_idx"]), "--num_scenes", "1", "--process_dynamic_mask"])
    else:
        run_dir = a.out_dir / "runs" / cfg["train"]["project"] / name
        if (run_dir / "checkpoint_final.pth").exists():
            log.info("trained run exists, skip: %s", run_dir); return
        t = cfg["train"]
        base = DS / "configs" / t["base_config"]
        drop = t.get("drop_model_nodes", [])
        if drop:  # write a derived config instead of editing drivestudio
            from omegaconf import OmegaConf
            c = OmegaConf.load(base)
            for k in drop:
                c.model.pop(k, None)
            base = a.out_dir / f"{name}_base.yaml"
            OmegaConf.save(c, base)
        launcher = [sys.executable, str(ROOT / "workstation/recon/train_launcher.py")]
        if t.get("min_init_dist") is not None:  # NaN guard for duplicate points (see train_launcher.py)
            launcher += ["--min_init_dist", str(t["min_init_dist"])]
        run([*launcher, "--", "--config_file", str(base.resolve()),
             "--output_root", str(a.out_dir / "runs"), "--project", t["project"],
             "--run_name", name, "dataset=" + t["dataset_config"],
             f"data.data_root={data_root.resolve()}", f"data.scene_idx={cfg['scene_idx']}", *t["overrides"]])


if __name__ == "__main__":
    main()
