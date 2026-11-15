# Phase 4: Fine-tuning SegFormer-B0 on the reconstructed scene-1077 (night)

- Config: `configs/phase4_render.yaml`, `configs/phase4_finetune.yaml`
- Run: `python -m workstation.watcher.runner` (render → pseudo_label → finetune). Evaluation: `python -m workstation.eval.ood_eval` (after `workstation/eval/teacher_real.py`)
- Raw results: `data/phase4/eval/scene-1077_*.json` (not committed)

## Steps
1. **Render**: the reconstructed (OmniRe) front camera is shifted laterally by -2/-1/0/+1/+2 m, and every other frame of the 101-frame interval is rendered (505 images, ~2 min). Image size 533x300. Large shifts produce blur and floaters.
2. **Pseudo labels**: predictions of the teacher SegFormer-B5 (Cityscapes); pixels with max probability below 0.7 are ignored (10.9–17.5% depending on the viewpoint).
3. **Fine-tuning**: B0 (`nvidia/segformer-b0-finetuned-cityscapes-1024-1024`) on the 355 renders from the first 70% of the scene, 5 epochs (AdamW, lr 2e-5, batch 8, flip and brightness augmentation). Loss 0.492 → 0.180, ~40 s. Cityscapes is not mixed in.
4. **Evaluation**: **real** frames starting 2 s after the end of the training interval (48 frames, unused in training), and real frames of the daytime scene-0061 (56 frames, to check forgetting).

## Results
| Evaluation set | Metric | Before | After |
|---|---|---|---|
| Night, held-out real frames (48) | mIoU vs. teacher B5 | 0.1694 | 0.1993 |
| same | pixel agreement with teacher B5 | 0.6611 | 0.7356 |
| Day scene-0061 real frames (56) | mIoU vs. teacher B5 | 0.5768 | 0.5618 |
| same | pixel agreement with teacher B5 | 0.9522 | 0.9471 |
| same | agreement with the original B0 | – | 0.9565 |

**The energy score (mean of the top 5% pixels, T=1) increased after fine-tuning.**
| Evaluation set | Mean before | Mean after | Frames above threshold 1.5, before → after |
|---|---|---|---|
| Night held-out (48) | 2.8149 | 7.2444 | 48 → 48 |
| Day scene-0061 (56) | 0.7311 | 3.3530 | 6 → 56 |

> **Note (phase 5)**: on the real labels of ACDC, mIoU went from 0.4514 to 0.4478 (night +0.003, fog -0.028). The gain on the proxy metric above did not translate into a gain on real labels. See `phase5_acdc_eval.md`.

## Interpretation
- **On the night held-out frames, B0 moved closer to the teacher B5** (mIoU +0.030, pixel agreement +7.5 pt). On daytime frames mIoU dropped by 0.015; some forgetting is possible, but with a single run it is unclear whether the difference is noise.
- **The energy increase is not a measure of the effect.** Cross-entropy training does not constrain the overall logit level (logsumexp), so the score scale drifts during training. The daytime scene moved in the same direction, which suggests a global shift (the exact cause was not examined). With the old threshold every daytime frame would be flagged. **The threshold must be re-tuned whenever the model is updated.** Energy regularization was not tried.
- "Agreement with the teacher" is not agreement with ground truth. B5 can also be wrong at night, and learning B5's mistakes raises the value too.
- Single scene, single run, single seed. The 48 held-out frames are temporally correlated and come from the continuation of the training drive; generalization to other night scenes is not covered here. mIoU averages over classes, and night frames contain few classes, so the value is unstable.

## Design decisions
| Decision | Why | Rejected alternative |
|---|---|---|
| Pseudo labels from teacher B5, ignore confidence < 0.7 | reconstructed scenes have no ground truth; reduces learning from errors | use every pixel (learns errors), manual labelling (too many images) |
| Teacher input upscaled to (576, 1024) | renders (533x300) are smaller than the teacher's training resolution | feed the original resolution |
| Train on the first 70%, evaluate on real frames after a 2 s gap | never train and evaluate on the same interval | train on all frames and evaluate on renders (would measure render quality) |
| No Cityscapes in the mix | avoids acquiring the dataset; forgetting is checked on a daytime scene | mix it in (needs the dataset) |
| Evaluate on real images, not only the energy | the energy scale drifts, as shown above | treat a lower energy as success |
