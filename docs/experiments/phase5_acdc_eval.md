# Phase 5: Evaluation on ACDC (real adverse-condition images with real labels)

- Config: `configs/phase5_acdc.yaml`; run: `python -m workstation.eval.acdc_eval`
- Data: ACDC val (106 night, 100 each of fog, rain and snow) and the same-place normal-condition images val_ref. Real Cityscapes 19-class labels (labelTrainIds). Not included in the repo (non-commercial license).
- Raw results: `data/phase5/eval/acdc_*.json` (including per-class IoU)
- Models: the original B0 (Cityscapes) and the B0 fine-tuned in phase 4 on renders of the reconstructed scene-1077 (one night scene in Singapore).

## 1. OOD detection (adverse val as positives, normal val_ref as negatives)
Frame score = mean over all pixels (k=100). Thresholds are the per-model CARLA values (original B0: -2.43, fine-tuned: -3.61).
| Model | Condition | AUROC | Detection rate at CARLA threshold | False-positive rate on normal images |
|---|---|---|---|---|
| Original B0 | night | 0.9966 | 1.00 | 0.19 |
| Original B0 | fog | 0.9029 | 0.49 | 0.06 |
| Original B0 | rain | 0.7910 | 0.69 | 0.28 |
| Original B0 | snow | 0.6865 | 0.90 | 0.47 |
| Fine-tuned | night | 0.9983 | 1.00 | 0.92 |
| Fine-tuned | fog | 0.9877 | 1.00 | 1.00 |
| Fine-tuned | rain | 0.7410 | 1.00 | 0.99 |
| Fine-tuned | snow | 0.7133 | 1.00 | 0.94 |

- **Fog is detectable on real images** (AUROC 0.90), unlike CARLA's dense fog (0.54); CARLA's fog likely looks different from real fog. Concluding "fog is undetectable" from the simulator alone would have been wrong.
- Rain and snow are separated less well (0.79 / 0.69).
- **The CARLA threshold does not transfer to ACDC as is.** Even for the original B0, the false-positive rate on normal images reaches 47% (snow), much worse than on nuScenes daytime frames (8.9%). The mean normal-image score (-2.42 to -2.93) is close to CARLA ID (-2.60), so the difference in spread is the likely cause.
- **The fine-tuned model's score scale varies strongly across datasets.** Its CARLA ID mean was -3.82, but ACDC normal images score -1.55 to -2.77, so the re-tuned CARLA threshold flags most of them. The ranking (AUROC) is preserved, so the conditions are still separable, but a threshold fixed in advance does not work.

## 2. Segmentation (mIoU against real labels)
Predictions are upsampled to the label resolution (1920x1080). Averaged over the classes present in each condition's ground truth.
| Condition | Before | After | Δ |
|---|---|---|---|
| night | 0.2331 | 0.2359 | +0.0028 |
| fog | 0.6649 | 0.6372 | -0.0277 |
| rain | 0.4951 | 0.4885 | -0.0066 |
| snow | 0.4754 | 0.4622 | -0.0132 |
| all | 0.4514 | 0.4478 | -0.0036 |

With this setup, fine-tuning did not improve mIoU on real labels. The night difference (+0.3 pt) is within run-to-run noise for a single run, and the other conditions dropped slightly.

> **Follow-up**: the ablation in `phase4_ablation.md` compares real frames, 0 m renders and ±2 m renders under the same recipe. Real frames gave +0.0074 mIoU over the original B0, while renders with lateral offsets gave -0.0031.

The phase 4 proxy metric (agreement with teacher B5 on night frames, 0.169 → 0.199) did improve; the proxy gain did not carry over to real labels. Possible reasons (not verified):
- The training data is 355 renders (533x300) of one night scene, while ACDC is Zurich, GoPro, 1920x1080: different place, camera and resolution.
- The pseudo labels are B5 predictions, and B5 also errs at night; moving toward the teacher is not moving toward the ground truth.
- The original training data (Cityscapes) was not mixed in, so some daytime knowledge may have been lost (drops on fog and snow).

## Conclusions
- The full loop (detect → reconstruct → render → fine-tune → evaluate) runs end to end. Improving the model through it needs more than this single-scene setup: more scenes, real-label validation in the loop, and mixing in the original training data.
- Thresholds must be tuned on data close to the deployment domain. CARLA was useful for choosing the aggregation (mean over all pixels), but the threshold value itself may not transfer to real images.
- After fine-tuning, the energy scale becomes unstable. Running threshold-based detection and fine-tuning in the same loop needs a way to keep the energy stable (mixing in the original data, energy regularization, or deployment-domain calibration data). None of these were tried.
