# Phase 1: Edge runtime and OOD scoring (nuScenes mini)

- Config: `configs/phase1_model.yaml`, `configs/phase1_ood.yaml`, `configs/phase1_edge_sim.yaml`
- Run: `python -m models.edge_sim` (env `shadowsplat-models`)
- Output: `data/phase1/edge_sim/<scene>/{scores.csv, events.json, thumb_*.jpg}` (not committed)

The edge runtime replays a recorded camera stream and runs the same steps an on-vehicle edge device would: SegFormer-B0 inference, per-pixel energy, frame score, thresholding and event extraction. Each event carries only metadata (scene, time range, score, a small thumbnail).

## Design decisions

| Decision | Why | Rejected alternative |
|---|---|---|
| The model outputs logits; the energy is computed outside the model | T and the aggregation can be changed in config, and the logits are reused for evaluation and fine-tuning | bake logsumexp into the model (T would be fixed) |
| Fixed 512x1024 input | balance between speed and detail for a B0-sized model | 1024x1024 (higher cost) |
| Reference implementation in NumPy (`models/ood_score.py`) | one definition shared by the edge runtime, CARLA tuning and evaluation | separate implementations per stage |

## Results (10 mini scenes, front camera, 2,342 frames, T=1)

These results used the initial setting of the frame score (mean of the top 5% pixels). Phase 5 later changed it to the mean over all pixels (`docs/experiments/phase5_carla_threshold.md`).

| Scene | Day/Night | Frames | Score min / mean / max |
|---|---|---|---|
| scene-0061 | day | 224 | -0.05 / 0.73 / 2.44 |
| scene-0103 | day | 229 | -0.56 / -0.27 / 0.08 |
| scene-0553 | day | 237 | 0.07 / 0.35 / 1.42 |
| scene-0655 | day | 237 | -0.64 / -0.06 / 1.43 |
| scene-0757 | day | 237 | -0.30 / 0.17 / 1.47 |
| scene-0796 | day | 234 | -0.61 / 0.14 / 1.32 |
| scene-0916 | day | 240 | -0.46 / 0.25 / 1.50 |
| scene-1077 | night | 240 | 0.78 / 2.68 / 4.01 |
| scene-1094 | night | 232 | 0.62 / 1.63 / 3.25 |
| scene-1100 | night | 232 | 1.17 / 1.82 / 3.10 |

### Sanity check (day/night as a proxy label for distribution shift)
- Frames from scenes whose description contains "Night" are positives (704), the rest negatives (1,638). Frame-level AUROC = **0.991**; FPR at 95% TPR = 0.040.
- This checks that the score reacts to a night-time shift; it is not an OOD detection benchmark. There are only three night scenes, and scene-specific factors (location, weather) are not separated.

## Notes
- The first provisional threshold (-5.0) was below the actual score range (-0.6 to 4.0), so every frame exceeded it. The energy range depends on the model, so thresholds must be chosen from real score distributions. The final threshold was chosen in CARLA (phase 5).
