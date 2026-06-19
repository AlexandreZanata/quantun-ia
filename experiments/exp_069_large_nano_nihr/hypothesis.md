# Hypothesis — EXP 069: LargeNanoMLP on NIHR synthetic CV (C2 anchor)

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

A **~1.11M-parameter** `LargeNanoMLP` trained with mini-batches on **nihr_cv_synthetic_v1**
(70K train / 15K val, 13 features, ~8% prevalence) will achieve **validation PR-AUC at least
1.0 pp above** logistic regression (QRISK-style baseline) on the same val split.

## Why I expect this

- Cardiovascular event prediction at realistic prevalence is nonlinear; PR-AUC is the primary
  metric (ROC-AUC can mask rare-event performance).
- exp_044 showed competitive ROC-AUC (0.8306) but failed the ROC advantage gate (−0.16 pp);
  PR-AUC may favor the deep model on imbalanced val.
- Full 2048→512→64 template with dropout 0.3 fits 70K rows on RTX 4060 (batch 2048).

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Val PR-AUC | LargeNanoMLP ≥ logistic + **1.0 pp** |
| Params | ≥ **1,000,000** |
| Protocol | Same train/val split and `StandardScaler` as exp_044 |

## Known limitations

- Synthetic NIHR cohort (CC0 Zenodo) — not clinical deployment.
- Single seed (42) — C2 anchor infrastructure gate, not multi-seed claim.
- Calibrated serve head deferred to exp_043 pattern on Synthea only.
