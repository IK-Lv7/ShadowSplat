"""Collects front-camera images in CARLA for each weather / time-of-day condition.

  python -m sim.collect [--conditions id_noon ood_night]

Start the server separately (CarlaUE4.sh -RenderOffScreen).
Output: <out_dir>/<condition>/route<k>/frame_NNNN.jpg and <out_dir>/<condition>/meta.json
Routes that already have images are skipped (re-runnable). Synchronous mode, 10 Hz, same routes and seed for every condition.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("collect")


def set_rendering(world, on: bool) -> None:
    s = world.get_settings()
    if s.no_rendering_mode == (not on):
        return
    s.no_rendering_mode = not on
    world.apply_settings(s)


def collect_route(world, tm, cfg: dict, weather_cfg: dict, spawn_idx: int, out: Path) -> None:
    import carla

    n = cfg["frames_per_route"]
    if len(list(out.glob("frame_*.jpg"))) >= n:
        log.info("skip %s", out)
        return
    out.mkdir(parents=True, exist_ok=True)
    w = {k: v for k, v in weather_cfg.items() if k != "role"}
    world.set_weather(carla.WeatherParameters(**{k: float(v) for k, v in w.items()}))
    random.seed(cfg["carla"]["seed"])
    bp = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    actors = []
    try:
        ego = world.spawn_actor(bp.find("vehicle.tesla.model3"), spawns[spawn_idx])
        actors.append(ego)
        log.info("ego spawned at %d", spawn_idx)
        ego.set_autopilot(True, tm.get_port())
        for sp in random.sample(spawns, cfg["carla"]["n_npc_vehicles"]):
            if sp.location.distance(spawns[spawn_idx].location) < 15:
                continue
            v = world.try_spawn_actor(random.choice(bp.filter("vehicle.*")), sp)
            if v is not None:
                v.set_autopilot(True, tm.get_port())
                actors.append(v)
        latest: dict = {}
        stride = cfg["capture_stride"]
        total = cfg["warmup_frames"] + n * stride
        cam = None
        for i in range(total):
            cap = i >= cfg["warmup_frames"] and (i - cfg["warmup_frames"]) % stride == stride - 1
            if i == cfg["warmup_frames"]:
                cc = cfg["camera"]
                cb = bp.find("sensor.camera.rgb")
                cb.set_attribute("image_size_x", str(cc["width"]))
                cb.set_attribute("image_size_y", str(cc["height"]))
                cb.set_attribute("fov", str(cc["fov"]))
                cam = world.spawn_actor(cb, carla.Transform(carla.Location(x=cc["x"], z=cc["z"])), attach_to=ego)
                actors.append(cam)
                cam.listen(lambda img: latest.__setitem__("img", img))
            # software rendering takes ~3 s per frame, so rendering is enabled only on captured ticks
            set_rendering(world, cap)
            world.tick()
            if i % 30 == 0:
                log.info("tick %d/%d", i, total)
            if cap:
                img = latest["img"]
                a = np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)[:, :, 2::-1]
                Image.fromarray(a).save(out / f"frame_{(i - cfg['warmup_frames']) // stride:04d}.jpg", quality=92)
        set_rendering(world, True)
        cam.stop()
    finally:
        for a in reversed(actors):
            try:
                if hasattr(a, "stop"):
                    a.stop()  # destroying a sensor that is still listening aborts the client
                a.destroy()
            except Exception:
                pass


def main() -> None:
    import carla

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/phase5_carla.yaml")
    ap.add_argument("--conditions", nargs="*")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    out_root = ROOT / cfg["out_dir"]
    out_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_root / "collect.log")])
    c = cfg["carla"]
    client = carla.Client(c["host"], c["port"])
    client.set_timeout(c["timeout_sec"])
    world = client.get_world()
    log.info("server %s map %s", client.get_server_version(), world.get_map().name)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(c["seed"])
    orig = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, c["fixed_delta_sec"]
    world.apply_settings(s)
    try:
        for name, wc in cfg["conditions"].items():
            if a.conditions and name not in a.conditions:
                continue
            for k, si in enumerate(cfg["spawn_indices"]):
                collect_route(world, tm, cfg, wc, si, out_root / name / f"route{k}")
            (out_root / name / "meta.json").write_text(json.dumps(
                {"role": wc["role"], "weather": {k: v for k, v in wc.items() if k != "role"},
                 "map": world.get_map().name, "fps": round(1 / c["fixed_delta_sec"])}, indent=1))
            log.info("done %s", name)
    finally:
        world.apply_settings(orig)
        tm.set_synchronous_mode(False)


if __name__ == "__main__":
    main()
