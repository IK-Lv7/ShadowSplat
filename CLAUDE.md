# CLAUDE.md

Instructions for Claude when working in this repository.

## Purpose

ShadowSplat is a personal portfolio project: a data loop for autonomous driving that automatically runs edge OOD detection → CloudKit sync → 3DGS reconstruction → fine-tuning → re-evaluation.
**Working end to end and explainable design decisions take priority over research novelty.**

See README.md for details.

## Current phase

Phases 0–6 are complete. Fine-tuning on the rendered views is an optional stage; the ablation (`docs/experiments/phase4_ablation.md`) is the evidence for deciding whether to use it. Records are in `docs/experiments/` and `docs/design/`. The edge runs as a Python edge runtime (`models/edge_sim.py`) that replays recorded camera streams. CARLA closed-loop evaluation was not pursued.
Update this section when the project moves on.

## Architecture principles (do not change)

1. **Never carry heavy data over CloudKit.** Only event metadata is sent (scene ID, time range, OOD score, thumbnail, status). The workstation extracts sensor data from the dataset directly.
2. **No dedicated backend server.** Everything runs as resident scripts on the workstation.
3. **Reconstruct only real-world logs (nuScenes / Waymo).** Simulator output is never reconstructed.
4. **Fine-tune only the edge perception model (SegFormer-B0).**

If a change appears to conflict with these principles, explain why and get confirmation before implementing it.

## CloudKit

- The workstation (Python) connects to CloudKit Web Services with a server-to-server key. Server-to-server keys can only access the **Public Database**, so events live there.
- Keys and the container ID are kept in `.env` and never committed.
- Fields of the `OODEvent` record type:
  - `sceneId` (String), `startTimestamp` / `endTimestamp` (Int64, μs)
  - `oodScore` (Double), `thumbnail` (Asset, small JPEG only)
  - `status` (String: detected / queued / extracting / reconstructing / rendered / finetuned / evaluated / failed)
  - `errorMessage` (String), `metrics` (String, JSON)
- Only the workstation updates the status. The edge only creates events as `detected`.

## OOD detection spec

- Per-pixel energy score: `E(x) = -T * logsumexp(logits / T)` (default T=1)
- Frame score: mean of the top k% pixel scores (k is set in `configs/`; currently 100 = mean over all pixels)
- A run of consecutive above-threshold frames forms one event, with margins before and after
- The threshold is chosen from CARLA tuning, and the rationale is recorded in `docs/`

## Directories and responsibilities

| Path | Responsibility |
|---|---|
| `models/` | edge model, OOD score reference implementation, edge runtime |
| `workstation/watcher/` | CloudKit watching, job queue, status updates |
| `workstation/extract/` | locating event intervals in the dataset |
| `workstation/recon/` | OmniRe wrapper; drivestudio itself is never modified |
| `workstation/render/` | novel-view rendering |
| `workstation/finetune/` | pseudo labels and fine-tuning |
| `workstation/eval/` | computing and recording metrics |
| `sim/` | CARLA integration |
| `scripts/` | demo video |
| `third_party/drivestudio/` | git submodule. **Never edit directly** |

## Coding conventions

### Python
- Python 3.10+, type hints required
- No hard-coded settings; read them from YAML in `configs/`
- Every stage is an independent "input directory → output directory" CLI that can be re-run on its own
- Long-running work must be resumable (checkpoints, detection of finished work)
- Use the standard `logging` module, with a log file per job

### General
- Secrets (keys, container ID, paths) live in `.env` and are never committed
- Datasets, weights and reconstructions are never committed (excluded by `.gitignore`)

## Experiments and records

- Experiments are managed per config file in `configs/`, and results are stored under the config name
- Metrics (AUROC, FPR@95TPR, PSNR, SSIM, LPIPS, mIoU) are always recorded as numbers
- Experiments that did not work are also recorded in `docs/experiments/`, with an analysis of the cause

## Writing docs

- Use terms precisely. Do not use the term "HIL". Write "shadow-mode-style detection", not "shadow mode"
- Do not imply any relationship with specific companies or products
- Never claim to "surpass" existing methods; describe the work as reproduction and design lessons
- Every design decision states why it was made and which alternatives were rejected

## Workflow

- Present a short plan before large changes
- State the reason when adding a new dependency
- When a phase is complete, update the progress in README.md and "Current phase" in this file
