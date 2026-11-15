# Scaling beyond a single workstation

ShadowSplat was built and tested at personal scale: one GPU, nuScenes mini, a handful of events.
This note lists what would break first if the number of vehicles and events grew by orders of magnitude, using the values measured in this project as a reference.
Measured values are from this environment; anything beyond that is a rough estimate.

## 1. Edge → sync (sending detected events)

**Measured here**
- Per event: a few hundred bytes of metadata (scene ID, time range, score, status) plus a thumbnail (256 px JPEG, ~8 KB).
- On the 10 nuScenes mini scenes (~200 s of driving), 5 events were detected.

**At scale**
- **Event volume control**: with a fixed threshold, the number of events grows with mileage. Per-vehicle / per-period caps and de-duplication of similar events (by place and condition) are needed.
- **Sync backend limits**: the CloudKit Public Database is fine for this project, but has request and storage limits. A fleet would need a dedicated message queue and storage, and the "no dedicated backend server" premise would no longer hold.
- **Fetching the heavy data**: here, the sensor data already sits on the workstation as a dataset. With real vehicles, a separate mechanism is needed to request the detected interval from the vehicle (interval upload requests, waiting for connectivity).

## 2. Workstation (reconstruction)

**Measured here** (scene-1077, RTX A6000)
- Preprocessing ~2.5 min, masks ~11 min, OmniRe training (30,000 steps) ~57 min, rendering ~1–2 min, fine-tuning ~1 min.
- Per-event time is dominated by reconstruction training; one GPU handles about 20 events per day.

**At scale**
- **Throughput**: jobs must be spread across many GPUs. Jobs here are resumable per directory and per stage, so distribution is straightforward, but workers need locking so they do not pick up the same job.
- **Duplicate scenes**: one scene can produce several events (two in scene-1094). Reconstructions should be shared per scene, with events only referencing them; this project already reuses trained scenes.
- **Failure handling**: a night scene stopped with a NaN at training start (`docs/experiments/phase3_scene1077_nan.md`). With many events, data-dependent failures need automatic classification and a retry-or-skip policy.
- **Multiple passes over one place**: a single pass gives limited viewpoint coverage (the renders at ±1–2 m degrade). At fleet scale the same place is passed many times, so reconstructions could be built from several passes and cover a wider range of viewpoints. This is an untested hypothesis; it needs cross-pass alignment and handling of lighting and moving objects (see `docs/experiments/phase4_ablation.md`).
- **Storage**: a reconstruction (checkpoints, renders) takes hundreds of MB to several GB per scene. A retention and deletion policy is needed.

## 3. Fine-tuning and redeployment

**Observed here**
- Fine-tuning shifts the energy score scale; the threshold had to be re-tuned per model (`docs/experiments/phase5_carla_threshold.md`).
- A threshold tuned in the simulator raised the false-positive rate considerably on some real conditions (`docs/experiments/phase5_acdc_eval.md`).
- Higher agreement with the teacher model's pseudo labels did not mean higher accuracy on real labels.

**At scale**
- **Version the model and threshold together**: the threshold must be versioned and deployed with the weights. Shipping a new model with an old threshold would flood the system with false positives.
- **Calibration data close to deployment**: thresholds need real images from the deployment region and camera. The simulator was useful for controlled conditions and for choosing the aggregation.
- **Real-label evaluation sets**: decisions to ship need real labels, not proxy metrics, so human labelling has to be part of the loop.
- **Monitoring forgetting**: fine-tuning only on hard conditions can degrade others. Mixing in the original training data and per-condition regression tests are needed.

## 4. What carries over unchanged
- The split of responsibilities: the edge sends only light metadata, and the workstation pulls the heavy data from the original logs.
- Every stage is an independent "input directory → output directory" CLI that detects finished work and resumes.
- Only the workstation advances the status, and transitions are validated (invalid ones are rejected).
