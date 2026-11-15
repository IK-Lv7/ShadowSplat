# Phase 0: Single-scene OmniRe reconstruction on nuScenes mini

- Config: `configs/phase0_omnire_nuscenes_mini.yaml`
- Run: `workstation/recon/run_recon.py` (`--stage preprocess` → `--stage masks` → `--stage train`)
- Target: nuScenes v1.0-mini, scene_idx=0 (interpolated to 10 Hz, 6 cameras, loaded at 1/3 resolution)
- Environment: RTX A6000 (48 GB), conda `drivestudio` (torch 2.0.0+cu117)
- Training: 30,000 steps, 1 h 28 min, ~12 GB VRAM

## Results

| Metric | test (held out) | full (all frames) |
|---|---|---|
| PSNR | 23.23 | 26.05 |
| SSIM | 0.697 | 0.829 |
| LPIPS | 0.213 | 0.194 |

- test, vehicle regions: PSNR 21.29 / SSIM 0.584; human regions: PSNR 21.07 / SSIM 0.562.
- `full` includes training frames, so `test` is the number to look at for reconstruction quality.
- Source: `data/phase0/runs/shadowsplat_phase0/phase0_omnire_nuscenes_mini/metrics/images_{test,full}_*.json` (not committed).

## Conditions and caveats

- `test` holds out one frame in ten (`test_image_stride=10`). The 10 Hz frames include interpolation between 2 Hz keyframes, and the neighbours of a held-out frame are in the training set, so the numbers are optimistic for true novel views.
- The SMPL pedestrian representation is off (no SMPL body model), so a derived config without `SMPLNodes` is used. Pedestrians are modelled by `DeformableNodes`; the human-region numbers are under this condition.
- No comparison against the numbers reported in the paper was made.

## Issues and fixes

| Symptom | Cause | Fix |
|---|---|---|
| No camera images in nuScenes trainval | only LiDAR and metadata had been downloaded | used v1.0-mini |
| Masks written to the wrong directory | `--data_root` lacked the split directory (`mini`) | pass `processed_10Hz/mini` |
| Training fails on `smpl_human_all` | `SMPLNodes` requires SMPL data | derived config without `SMPLNodes` |
| Training aborts (`libcuda.so` not found) | `/usr/lib/wsl/lib` is not on the library path on WSL2 | the wrapper adds it to `LD_LIBRARY_PATH` |
