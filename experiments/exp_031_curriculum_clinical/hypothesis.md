# Hypothesis — EXP 031: Clinical Curriculum vs Random (Breast Cancer)

**Date:** 2026-06-18  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

On **Wisconsin Breast Cancer** (full UCI, 30% holdout), epoch-matched **margin_batches**
curriculum (easy→hard) with `hybrid_sandwich` will achieve a **higher mean holdout accuracy**
than a **random-order** baseline across 10 publication seeds.

## Why I expect this

- exp_005 showed margin curriculum **hurts** on noisy circles, but clinical tabular features
  have meaningful centroid structure — margin ordering may help exposure bias.
- Breast cancer is highly learnable (~97% hybrid in exp_024); curriculum should not collapse
  training but may improve hard-example coverage before refine.

## What would prove me wrong

- Mean holdout: curriculum ≤ random (≤ 0 pp advantage)
- Curriculum holdout drops below learnability gate (< 90% mean random baseline)
- Paired per-seed wins < 5/10 with mean advantage ≤ 0 pp

## Metrics I will measure

- [ ] Per-seed holdout accuracy (random vs margin_batches)
- [ ] Mean advantage in percentage points
- [ ] Paired Wilcoxon (margin_batches vs random)
- [ ] Applicability gate (random mean ≥ 90%)

## Success criteria

- Random baseline passes learnability gate (mean ≥ 90%)
- Mean curriculum holdout **strictly exceeds** random mean (advantage > 0 pp)
- `make check-real` stays green after merge

## Known limitations

- Single clinical dataset; not a universal curriculum claim
- Hybrid quantum block on CPU (PennyLane); classical head only on GPU path
- Epoch-matched comparison (60 epochs total: 4×12 stages + 12 refine)
