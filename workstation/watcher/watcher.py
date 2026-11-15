"""Daemon that watches the EventStore and enqueues detected events as jobs.

  python -m workstation.watcher.watcher --once            # process once
  python -m workstation.watcher.watcher                   # poll continuously
  python -m workstation.watcher.watcher --import-edge-sim data/phase1/edge_sim   # create events from the edge runtime output

Jobs are stored as <jobs_dir>/<recordName>/job.json (they survive restarts).
Processing after enqueueing (extract onwards) is done by runner.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

from workstation.watcher.store import CloudKitStore, EventStore, LocalStore

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("watcher")


def job_logger(job_dir: Path) -> logging.Handler:
    h = logging.FileHandler(job_dir / "job.log")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    return h


def enqueue(store: EventStore, jobs_dir: Path) -> list[dict]:
    """Enqueue detected events. If job.json exists it is not recreated; only the status advances (idempotent)."""
    queued: list[dict] = []
    for ev in store.list_by_status("detected"):
        name = ev["recordName"]
        jd = jobs_dir / name
        jd.mkdir(parents=True, exist_ok=True)
        jp = jd / "job.json"
        if not jp.exists():
            jp.write_text(json.dumps({"recordName": name, "sceneId": ev["sceneId"],
                                      "startTimestamp": ev["startTimestamp"],
                                      "endTimestamp": ev["endTimestamp"],
                                      "oodScore": ev["oodScore"]}, indent=1))
        h = job_logger(jd)
        log.addHandler(h)
        try:
            store.update(name, "queued")
            log.info("queued %s scene=%s score=%.2f", name, ev["sceneId"], ev["oodScore"])
            queued.append({**ev, "status": "queued"})
        except Exception:
            log.exception("failed to queue %s", name)
        finally:
            log.removeHandler(h)
            h.close()
    return queued


def make_store(cfg: dict) -> EventStore:
    if cfg["store"] == "cloudkit":
        load_dotenv(ROOT / ".env")
        return CloudKitStore.from_env(cfg["cloudkit"]["record_type"], cfg["cloudkit"]["query_limit"])
    return LocalStore(ROOT / cfg["local"]["events_dir"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/phase2_watcher.yaml")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--import-edge-sim", type=Path, help="create detected events from the edge runtime output (in the configured store)")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    jobs_dir = ROOT / cfg["jobs_dir"]
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(jobs_dir / "watcher.log")])
    store = make_store(cfg)
    if args.import_edge_sim:
        for p in sorted(args.import_edge_sim.glob("*/events.json")):
            for ev in json.loads(p.read_text()):
                log.info("created %s", store.create(ev))
        return
    while True:
        enqueue(store, jobs_dir)
        if args.once:
            return
        time.sleep(cfg["poll_interval_sec"])


if __name__ == "__main__":
    main()
