"""Locates the interval of an event (time range) in the nuScenes metadata.

Sensor data is not copied (it never goes through CloudKit; the workstation reads the original log directly).
Output extract.json: scene name, scene index (scene_idx), frame timestamps inside the event, keyframe count, etc.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger("extract")


def _load(meta: Path, name: str):
    return json.loads((meta / f"{name}.json").read_text())


def scene_index(meta: Path, scene_name: str) -> int:
    """drivestudio scene_idx (order in scene.json)."""
    for i, s in enumerate(_load(meta, "scene")):
        if s["name"] == scene_name:
            return i
    raise KeyError(scene_name)


def scene_frames(meta: Path, scene_name: str, camera: str) -> list[dict]:
    """All frames (keyframes + sweeps) of the given camera in the scene, in time order."""
    scene = next((s for s in _load(meta, "scene") if s["name"] == scene_name), None)
    if scene is None:
        raise KeyError(f"scene not found: {scene_name}")
    sd = {s["token"]: s for s in _load(meta, "sample_data")}
    calib = {c["token"]: c for c in _load(meta, "calibrated_sensor")}
    sensor = {s["token"]: s["channel"] for s in _load(meta, "sensor")}
    cur = next(s for s in sd.values()
               if s["sample_token"] == scene["first_sample_token"]
               and sensor[calib[s["calibrated_sensor_token"]]["sensor_token"]] == camera)
    out = []
    while cur:
        out.append(cur)
        cur = sd[cur["next"]] if cur["next"] else None
    return out


def extract(nuscenes_root: Path, split_dir: str, job: dict, camera: str, out: Path) -> dict:
    meta = nuscenes_root / split_dir
    frames = scene_frames(meta, job["sceneId"], camera)
    ts = [f["timestamp"] for f in frames]
    inside = [f for f in frames if job["startTimestamp"] <= f["timestamp"] <= job["endTimestamp"]]
    res = {
        "sceneId": job["sceneId"], "scene_idx": scene_index(meta, job["sceneId"]),
        "scene_start_us": ts[0], "scene_end_us": ts[-1],
        "event_start_us": job["startTimestamp"], "event_end_us": job["endTimestamp"],
        "n_frames_scene": len(frames), "n_frames_event": len(inside),
        "n_keyframes_event": sum(1 for f in inside if f["is_key_frame"]),
        "event_frame_timestamps_us": [f["timestamp"] for f in inside],
    }
    if not inside:
        raise ValueError("no frames inside the event interval")
    out.write_text(json.dumps(res, indent=1))
    log.info("extract %s: scene_idx=%d frames=%d/%d", job["sceneId"], res["scene_idx"], len(inside), len(frames))
    return res


def main() -> None:
    import os
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job_dir", type=Path, required=True, help="directory containing job.json")
    ap.add_argument("--nuscenes-root", type=Path, default=os.environ.get("NUSCENES_RAW"))
    ap.add_argument("--split_dir", default="v1.0-mini")
    ap.add_argument("--camera", default="CAM_FRONT")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    out = a.job_dir / "extract.json"
    if out.exists():
        log.info("exists, skip"); return
    extract(Path(a.nuscenes_root), a.split_dir, json.loads((a.job_dir / "job.json").read_text()), a.camera, out)


if __name__ == "__main__":
    main()
