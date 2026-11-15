"""Processes queued jobs: extract -> preprocess -> masks -> train -> render -> pseudo_label -> finetune.

  python -m workstation.watcher.runner --once

- Status: queued -> extracting -> reconstructing -> rendered (render + pseudo labels done) -> finetuned (fine-tuning done).
  evaluated is set by the evaluation step (workstation/eval).
- Completed stages are recorded in job.json "done"; after a restart the job resumes from there.
  Jobs stuck in extracting / reconstructing are resumed automatically.
- On failure the status becomes failed with errorMessage (failed -> queued re-runs it).
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

from workstation.extract.extract import extract
from workstation.watcher.store import STATUSES, EventStore
from workstation.watcher.watcher import ROOT, enqueue, job_logger, make_store

log = logging.getLogger("runner")
RECON_STAGES = ["preprocess", "masks", "train"]
POST_STAGES = ["render", "pseudo_label", "finetune"]
ALL_STAGES = ["extract", *RECON_STAGES, *POST_STAGES]
# Runtime (Python from .env) and script per stage. Environments differ, so each stage is a separate process.
POST = {"render": ("DRIVESTUDIO_PYTHON", "workstation/render/render_views.py"),
        "pseudo_label": ("SEGFORMER_PYTHON", "workstation/finetune/pseudo_label.py"),
        "finetune": (None, "workstation/finetune/finetune.py")}  # None = the runner's own environment (torch + transformers)


def _jobs(job_dir: Path) -> dict:
    return json.loads((job_dir / "job.json").read_text())


def _mark(job_dir: Path, stage: str) -> None:
    j = _jobs(job_dir)
    j.setdefault("done", [])
    if stage not in j["done"]:
        j["done"].append(stage)
    (job_dir / "job.json").write_text(json.dumps(j, indent=1))


def recon_metrics(rcfg: dict, rec_cfg: dict, scene_idx: int, base_idx: int, out_dir: Path) -> dict:
    stem = Path(rcfg["config"]).stem
    run = stem if scene_idx == base_idx else f"{stem}_s{scene_idx:03d}"
    pat = str(out_dir / rcfg["metrics_glob"].format(project=rec_cfg["train"]["project"], run=run))
    files = sorted(glob.glob(pat))
    if not files:
        return {}
    d = json.loads(Path(files[-1]).read_text())
    keep = {k.split("/")[-1]: round(v, 4) for k, v in d.items()
            if k.startswith("image_metrics/test/") and k.split("/")[-1] in ("psnr", "ssim", "lpips")}
    return {"recon": keep}


def process(store: EventStore, ev: dict, jobs_dir: Path, cfg: dict) -> None:
    name = ev["recordName"]
    jd = jobs_dir / name
    h = job_logger(jd)
    log.addHandler(h)
    try:
        job = _jobs(jd)
        done = job.get("done", [])
        cur = ev["status"]

        def go(status: str, **kw) -> None:
            """Advance the status. When resuming a job that is already further along, keep its status and only update metrics etc."""
            nonlocal cur
            if STATUSES.index(status) < STATUSES.index(cur) and cur != "failed":
                status = cur
            store.update(name, status, **kw)
            cur = status

        if cur == "queued":
            go("extracting", error_message="")
        if "extract" not in done:
            extract(Path(os.environ["NUSCENES_RAW"]), cfg["nuscenes_split_dir"], job, cfg["camera"],
                    jd / "extract.json")
            _mark(jd, "extract")
        info = json.loads((jd / "extract.json").read_text())
        go("reconstructing")
        rcfg = cfg["recon"]
        rec_path = ROOT / rcfg["config"]
        rec_cfg = yaml.safe_load(rec_path.read_text())
        out_dir = ROOT / rcfg["out_dir"]
        for st in RECON_STAGES:
            if st in _jobs(jd).get("done", []):
                continue
            cmd = [os.environ["DRIVESTUDIO_PYTHON"], str(ROOT / "workstation/recon/run_recon.py"),
                   "--config", str(rec_path), "--stage", st, "--scene_idx", str(info["scene_idx"]),
                   "--out_dir", str(out_dir)]
            log.info("stage %s: %s", st, " ".join(cmd))
            with open(jd / f"{st}.log", "a") as lf:
                subprocess.run(cmd, check=True, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT)
            _mark(jd, st)
        job = _jobs(jd)
        m = job.get("metrics") or recon_metrics(rcfg, rec_cfg, info["scene_idx"], rec_cfg["scene_idx"], out_dir)
        job["metrics"] = m
        (jd / "job.json").write_text(json.dumps(job, indent=1))
        go("reconstructing", metrics=m)
        log.info("recon done %s metrics=%s", name, m)
        run_dir = out_dir / "runs" / rec_cfg["train"]["project"] / (
            Path(rcfg["config"]).stem if info["scene_idx"] == rec_cfg["scene_idx"]
            else f"{Path(rcfg['config']).stem}_s{info['scene_idx']:03d}")
        for st in POST_STAGES:
            if st in _jobs(jd).get("done", []):
                continue
            envkey, script = POST[st]
            py = os.environ[envkey] if envkey else sys.executable
            cmd = [py, str(ROOT / script), "--job_dir", str(jd.resolve())]
            if st == "render":
                cmd += ["--run_dir", str(run_dir)]
            env = {**os.environ, "PYTHONPATH": str(ROOT)}
            log.info("stage %s: %s", st, " ".join(cmd))
            with open(jd / f"{st}.log", "a") as lf:
                subprocess.run(cmd, check=True, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
            _mark(jd, st)
            if st == "pseudo_label":
                go("rendered", metrics=m)
        go("finetuned", metrics=m)
        log.info("finetuned %s", name)
    except Exception as e:
        log.exception("failed %s", name)
        try:
            store.update(name, "failed", error_message=f"{type(e).__name__}: {e}"[:500])
        except Exception:
            log.exception("could not record failure")
    finally:
        log.removeHandler(h)
        h.close()


def pending(store: EventStore, jobs_dir: Path, fresh: list[dict] | None = None) -> list[dict]:
    """Jobs to process: queued, plus interrupted ones (extracting / reconstructing / rendered with unfinished stages)."""
    # CloudKit queries can lag behind writes, so events just queued are added directly
    out = list(fresh or []) + list(store.list_by_status("queued")) + list(store.list_by_status("extracting"))
    for st in ("reconstructing", "rendered"):
        for ev in store.list_by_status(st):
            jp = jobs_dir / ev["recordName"] / "job.json"
            if jp.exists() and set(ALL_STAGES) - set(_jobs(jp.parent).get("done", [])):
                out.append(ev)
    seen: set[str] = set()
    uniq = [e for e in out if not (e["recordName"] in seen or seen.add(e["recordName"]))]
    return [e for e in uniq if (jobs_dir / e["recordName"] / "job.json").exists()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watcher-config", type=Path, default=ROOT / "configs/phase2_watcher.yaml")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/phase3_runner.yaml")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    load_dotenv(ROOT / ".env")
    wcfg = yaml.safe_load(a.watcher_config.read_text())
    cfg = yaml.safe_load(a.config.read_text())
    jobs_dir = ROOT / wcfg["jobs_dir"]
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(jobs_dir / "runner.log")])
    store = make_store(wcfg)
    while True:
        fresh = enqueue(store, jobs_dir)
        for ev in pending(store, jobs_dir, fresh):
            process(store, ev, jobs_dir, cfg)
        if a.once:
            return
        time.sleep(wcfg["poll_interval_sec"])


if __name__ == "__main__":
    main()
