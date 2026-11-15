# Phase 3: Reconstructing scene-1077 (night) and a NaN at training start

- Config: `configs/phase0_omnire_nuscenes_mini.yaml` (`--scene_idx 7`); run: `python -m workstation.watcher.runner`
- Event: scene-1077, interval covers all 240 frames, OOD score 4.01

## Failure: NaN at step 0
- Symptom: preprocessing and mask extraction through the runner succeeded, but training stopped with `ValueError: NaN detected in loss Background_sharp_shape_reg at step 0`. The job became `failed` with the error message recorded. scene-0061 with the same config was fine.
- Cause: the initial background Gaussian scale is the log of the mean distance to the three nearest neighbours (`models/gaussians/vanilla.py`). Duplicate points give distance 0 → log(0) = -inf, the scale becomes 0, and the scale ratio in `sharp_shape_reg` becomes 0/0 = NaN. RigidNodes clamps the same value to (0.002, 100); the background does not. The LiDAR points contain no NaN/Inf. The failure reproduces on re-run.
- The number of duplicate points was not measured. "Duplicate points are the cause" is inferred from the NaN disappearing once a lower bound is applied.
- Fix: `workstation/recon/train_launcher.py` launches training with a lower bound on the initial distance (`min_init_dist: 0.002`, same as RigidNodes) without modifying drivestudio. Set by `train.min_init_dist` in `configs/phase0_omnire_nuscenes_mini.yaml`.
- Alternatives: patch drivestudio directly (not allowed by the project rules), or de-duplicate LiDAR points in preprocessing (needs the root cause pinned down first).
- Impact: the phase 0 result for scene-0061 was trained without the bound. Re-training scene-0061 now applies it (the behaviour is unchanged unless some initial distance is below 0.002).

## Results (held-out images, 30,000 steps)
| Scene | split | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| scene-0061 (day, phase 0) | test | 23.2316 | 0.6974 | 0.2126 |
| scene-1077 (night) | test | 26.4667 | 0.6676 | 0.5040 |
| scene-1077 (night) | full | 28.1449 | 0.7131 | 0.4861 |

- Night images are mostly dark, which raises PSNR, while LPIPS is about 2.4x the daytime value. PSNR alone does not describe the quality.
- Training took about 57 min (from timestamps in `runner.log`); mask extraction about 10.6 min.

## Operational mistake
- A manual re-run passed `--out_dir data/phase0` (a relative path), so the output went to `third_party/drivestudio/data/phase0/` relative to drivestudio's working directory. While moving it, `third_party/drivestudio/data` was deleted as a whole, including files tracked by drivestudio (ego masks). They were restored with `git checkout`, and the submodule was confirmed unchanged at `e59bda4`.
- Prevention: `run_recon.py` now resolves `--out_dir` to an absolute path.
