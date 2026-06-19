# Hypothesis — EXP 070: LargeNanoMLP on GoBug file-level defects (C3 anchor)

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

A **~1.14M-parameter** `LargeNanoMLP` (2048→512→64) trained on **code_defects_gobug_v1**
(27,172 train / 5,822 val, 23 static code metrics) will achieve **validation PR-AUC at least
2.0 pp above** logistic regression on the temporal val split.

## Why I expect this

- exp_045 used a reduced topology (512→128→32, ~82K params) and tied logistic (0.00 pp).
- Full nano template with dropout 0.3 may capture nonlinear interactions in code metrics.
- PR-AUC is primary for imbalanced defect detection (~15% positive rate).

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Val PR-AUC | LargeNanoMLP ≥ logistic + **2.0 pp** |
| Params | ≥ **1,000,000** |
| Protocol | Same temporal train/val split and `StandardScaler` as exp_045 |

## Known limitations

- GoBug combined subset (~39K rows); temporal proxy via commit-sha ordering.
- Single seed (42) — C3 anchor infrastructure gate.
- Hybrid QNN head deferred to exp_071.
