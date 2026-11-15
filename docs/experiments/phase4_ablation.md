# Phase 4 ablation: is data from the 3DGS reconstruction useful for fine-tuning?

- Config: `configs/phase4_ablation.yaml`
- Run: `python -m workstation.finetune.ablation prepare` → teacher labelling (`workstation/finetune/pseudo_label.py --src_dir ... --dst_dir ...`) → `python -m workstation.finetune.ablation train` → `python -m workstation.eval.ablation_eval`
- Raw results: `data/phase4/ablation/summary.json` (not committed)

## Question
Phase 4 fine-tuned SegFormer-B0 on renders of the reconstructed scene-1077 (night) and saw no gain on real labels. That run had no baseline, so it could not say whether the reconstruction step helps. This ablation adds the missing comparison.

## Setup
Same teacher (SegFormer-B5), same pseudo-label settings (confidence ≥ 0.7), same recipe (AdamW, lr 2e-5, batch 8, flip + brightness augmentation) and **the same number of optimizer steps (220) for every arm**. Only the training images differ. Frames are the same 71 time steps of the first 70% of scene-1077 in every arm. 3 seeds per arm. Evaluation: mIoU on ACDC val (night 106, fog/rain/snow 100 each) with real labels.

| Arm | Training images | Count |
|---|---|---|
| `real` | real front-camera frames (1600x900) | 71 |
| `real_lr` | the same frames downsampled to the render resolution (533x300) | 71 |
| `render_0` | renders at the original camera pose | 71 |
| `render_all` | renders at 0, ±1, ±2 m lateral offsets (the phase 4 setting) | 355 |

`real_lr` exists to separate the effect of resolution (renders are 533x300) from the effect of the reconstruction itself.

## Results (mIoU on ACDC val, mean ± std over 3 seeds)
| Arm | night | fog | rain | snow | all |
|---|---|---|---|---|---|
| original B0 (no fine-tuning) | 0.2331 | 0.6649 | 0.4951 | 0.4754 | 0.4514 |
| `real` | 0.2390 ± 0.0016 | 0.6523 ± 0.0025 | 0.4995 ± 0.0026 | 0.4835 ± 0.0012 | **0.4588 ± 0.0012** |
| `real_lr` | 0.2304 ± 0.0015 | 0.6472 ± 0.0020 | 0.4850 ± 0.0022 | 0.4725 ± 0.0025 | 0.4506 ± 0.0012 |
| `render_0` | 0.2369 ± 0.0026 | 0.6469 ± 0.0014 | 0.5007 ± 0.0046 | 0.4758 ± 0.0007 | 0.4565 ± 0.0007 |
| `render_all` | 0.2382 ± 0.0011 | 0.6389 ± 0.0024 | 0.4842 ± 0.0033 | 0.4641 ± 0.0013 | 0.4483 ± 0.0013 |

Differences in the "all" column:
- `real` vs. original: **+0.0074**
- `render_all` vs. `real` (the question asked): **-0.0105**
- `render_all` vs. original: -0.0031
- `render_0` vs. `real`: -0.0023
- `render_all` vs. `render_0` (effect of adding ±1/±2 m views): **-0.0082**
- `real` vs. `real_lr` (effect of resolution): +0.0082
- `render_0` vs. `real_lr` (same resolution): +0.0059

## Interpretation
- **In this setup, adding novel-view renders (±1, ±2 m) did not help; it was worse than plain real frames** (-0.0105, about 9 seed standard deviations) and slightly worse than not fine-tuning at all. So this experiment does not support using the reconstruction to create new viewpoints for training.
- **The damage comes from the shifted views, not from the reconstruction as such**: renders at the original pose (`render_0`) are within -0.0023 of real frames, and adding the ±1/±2 m views lowers the result by 0.0082. The loss is not on night (0.2382 vs. 0.2390) but on fog, rain and snow (-0.013 to -0.019 vs. `real`), which fits shifted views carrying artifacts (blur, floaters, wrong geometry) and teacher labels computed on those artifacts. This explanation is not verified by an experiment.
- **Resolution matters**: real frames at full resolution beat the same frames downsampled to the render size by 0.0082, and the downsampled version is no better than not fine-tuning. A good part of the render arms' deficit relative to `real` may come from the 533x300 render size rather than from the reconstruction.
- `render_0` is above `real_lr` (+0.0059) at the same resolution. A possible reason is that renders are smoother and less noisy than the night camera images; not tested.
- Fine-tuning on a night scene lowers fog accuracy in every arm (0.665 → 0.639–0.652), a sign of some forgetting.

## Caveats
- One scene, one teacher, one training recipe; seeds vary the training randomness but not the data, so the standard deviations understate the uncertainty across scenes.
- Equal optimizer steps means `render_all` (355 images) sees each image about 5x less often than the 71-image arms. Equal compute was chosen on purpose, but a different budget could change the ranking. The learning rate was not tuned per arm.
- Evaluation data (ACDC, Zurich) comes from a different city and camera than the training scene (Singapore). The pseudo labels are teacher predictions, not ground truth, in every arm.
- The ±1, ±2 m offsets are one choice; smaller offsets (for example ±0.5 m) were not tried.

## Conclusion
With this recipe, fine-tuning on real frames of the detected scene is the best arm, renders at the original pose come close, and the lateral-offset renders do not add to it. The pipeline's job is to produce the reconstruction, the renders and the pseudo labels automatically; **whether to use those renders as training data is left to the user**, and this ablation is the evidence for that decision.

## Open questions (not tested)
- Smaller offsets (for example ±0.5 m) and higher-resolution renders.
- More scenes, and a control that fine-tunes on real frames of a scene that was *not* detected (does selecting detected scenes matter, or does any extra real data help?).
- **Multiple passes over the same place** (other days, lanes or vehicles) would widen the viewpoints that a reconstruction covers reliably. This is a hypothesis; the data here has one pass per place, and OmniRe is designed for a single sequence, so merging passes would need alignment and handling of lighting and moving objects.
