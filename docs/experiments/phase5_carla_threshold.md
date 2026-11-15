# Phase 5: OOD threshold tuning in CARLA

- Config: `configs/phase5_carla.yaml`. Collection: `python -m sim.collect`; evaluation: `python -m workstation.eval.carla_threshold`; check on real images: `python -m workstation.eval.nuscenes_k_check`
- Raw results: `data/phase5/eval/*.json` (not committed)

## Data
CARLA 0.9.16 (Town10HD_Opt, front camera 800x450, one image every 0.3 s, 2 routes × 40 images per condition). Two in-distribution (ID) daytime conditions and four OOD conditions, 480 images in total.
| Condition | Role | Description |
|---|---|---|
| id_noon / id_overcast | ID | clear noon / overcast day |
| ood_night | OOD | night |
| ood_rain_night | OOD | night + heavy rain + light fog |
| ood_fog | OOD | day + dense fog |
| ood_low_sun | OOD | low sun (3° altitude) |

## Results: frame-score aggregation (mean of the top k% pixels)
Original B0 (Cityscapes). Frame-level AUROC with OOD as positives, ID (160) vs. OOD (320).
| k (top k%, 100 = all pixels) | AUROC | FPR@95TPR | Threshold (5% ID FPR) | OOD detection rate at that threshold |
|---|---|---|---|---|
| 1 | 0.516 | 0.994 | 1.47 | 0.356 |
| 5 (initial setting) | 0.634 | 0.994 | 0.50 | 0.378 |
| 10 | 0.694 | 0.994 | 0.08 | 0.475 |
| 25 | 0.773 | 0.700 | -0.52 | 0.488 |
| **100** | **0.885** | **0.500** | **-2.43** | **0.750** |

Per condition (k=100): night AUROC 1.000, rain at night 1.000, low sun 1.000, **dense fog 0.539** (barely separable). With k=5, low sun is 0.261 (scored lower than ID, i.e. inverted) and fog 0.529.

## Check on real images (nuScenes mini)
Night (704 frames) vs. day (1,638 frames) as proxy labels. AUROC: k=1: 0.954, 5: 0.991, 10: 0.998, 25: 1.000, 100: 1.000. **Larger k is better on real images too**, matching CARLA.
Applying the CARLA threshold (k=100, -2.43) to nuScenes gives an 8.9% false-positive rate on daytime frames (target 5%) and 100% detection of night frames. The mean daytime score is close between CARLA ID (-2.60) and nuScenes day (-2.78).

## Config update
`configs/phase1_ood.yaml`: `top_k_percent: 5 → 100`, `threshold: 1.5 (provisional) → -2.43`.

## Re-tuning for the fine-tuned model
After the phase 4 fine-tuning (scene-1077) the score scale changes, so the threshold was re-tuned.
| Model (k=100) | AUROC | FPR@95TPR | Threshold (5% ID FPR) | OOD detection rate | Fog AUROC |
|---|---|---|---|---|---|
| Original B0 | 0.885 | 0.500 | -2.43 | 0.750 | 0.539 |
| Fine-tuned | 0.944 | 0.475 | -3.61 | 0.875 | 0.775 |

The mean ID score moved from -2.60 to -3.82. Keeping the old threshold would break detection, so the threshold is re-tuned in CARLA every time the model is updated.

## Interpretation
- Small k performs worse probably because, even on ID frames, the top pixels (sky, light sources) already have high energy, which hides scene-wide changes (not verified per pixel).
- **On real ACDC images, fog is detectable (AUROC 0.90).** CARLA's fog may look different from real fog, so "dense fog is not detectable" holds for CARLA only. The CARLA threshold also did not transfer to all real conditions: on ACDC normal-condition images its false-positive rate reached 47% (`phase5_acdc_eval.md`).
- Fine-tuning raised the CARLA fog AUROC from 0.539 to 0.775, from a single run of a model trained on a night scene.
- The CARLA ID set is 160 images from 2 routes × 2 conditions, and frames 0.3 s apart are correlated, so the effective sample size is smaller than it looks (no confidence intervals computed). k was selected on the same data, so the numbers are optimistic; the direction was confirmed on real images.
- Rendering runs in software (no GPU Vulkan under WSL), so the amount of data was kept small. CARLA images look different from real images and have a different resolution (800x450).
