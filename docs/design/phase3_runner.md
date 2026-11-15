# Phase 3: Automatic reconstruction of detected intervals (runner)

- Config: `configs/phase3_runner.yaml` (reconstruction settings reused from `configs/phase0_omnire_nuscenes_mini.yaml`)
- Implementation: `workstation/extract/extract.py`, `workstation/watcher/runner.py`, `workstation/recon/run_recon.py`
- Run: `python -m workstation.watcher.runner [--once]` (enqueueing by the watcher runs in the same loop)

## Flow
`queued` → `extracting` (extract) → `reconstructing` (preprocess → masks → train) → `rendered` (rendering + pseudo labels, phase 4) → `finetuned` (phase 4). PSNR/SSIM/LPIPS of the reconstruction are written to `metrics`.
Completed stages are recorded in `job.json` under `done`; after a restart the job resumes from there. On failure the status becomes `failed` with `errorMessage` (`failed → queued` re-runs it).

## Design decisions

| Decision | Why | Rejected alternative |
|---|---|---|
| Reconstruct the whole scene containing the event, not just the interval | drivestudio preprocesses and trains per scene (~20 s); interval-only reconstruction makes initialization and training unstable | reconstruct only the event interval |
| Record the event interval as frame timestamps in `extract.json` (used to select frames for rendering and evaluation) | sensor data is never copied; the original log is referenced directly | copy the interval's images and LiDAR into a separate directory |
| Call `run_recon.py` as a subprocess | keeps the drivestudio / SegFormer environments (torch 2.0 / 1.8) separate from the runner's | run everything in one environment (dependency conflicts) |
| Outputs collected under `data/phase0`; existing runs reused | trained scenes are skipped (resumable); runs are named `<config>[_sNNN]` | a separate out_dir per scene |
| Events just queued are added to the work list directly | CloudKit queries can lag behind writes; the first run missed an event because of this | rely on query results only |

## Results
- On the CloudKit server, the scene-0061 event moved `queued` → `extracting` → `reconstructing`, and `metrics` was set to `{"recon": {"psnr": 23.2316, "ssim": 0.6974, "lpips": 0.2126}}`. An event for a non-existent scene was recorded as `failed` with its error message.
- scene-0061: event interval of 85/224 frames (about 7.2 s plus margins), scene_idx=0. Its reconstruction from phase 0 was reused.
- scene-1077 (night, scene_idx=7) ran preprocess → masks → train through the runner. Training first stopped with a NaN and the job became `failed`; after the fix, `failed → queued` re-ran it, skipping finished stages, and `metrics` (PSNR 26.4667 / SSIM 0.6676 / LPIPS 0.5040) was recorded. Details: `docs/experiments/phase3_scene1077_nan.md`.
- Not tested: resuming training from an intermediate checkpoint after stopping the runner mid-training (stage-level resumption and skipping of finished training were tested).
