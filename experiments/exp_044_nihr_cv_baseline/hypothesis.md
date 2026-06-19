# Hypothesis — EXP 044: NIHR synthetic CV baseline

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

Training **LargeNanoMLP** on the NIHR synthetic cardiovascular dataset (~100k rows, ~6.6%
positive) will reach **val ROC-AUC ≥ 0.70** and beat a **QRISK-style logistic baseline**
by **≥ 0.5 pp** on the holdout split.

## Why I expect this

- NIHR Zenodo cohort has realistic prevalence (unlike Synthea ~99% positive).
- exp_034 showed LargeNanoMLP beats logistic on 40-feature clinical tabular.
- 13 NIHR features cover demographics, vitals, comorbidities, and FEV1 — sufficient signal.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Primary | Val ROC-AUC (nano) ≥ **0.70** |
| Secondary | nano − logistic ≥ **0.5 pp** val ROC-AUC |
| Data | `nihr_cv_synthetic_v1` ready in manifest; train-only median imputation |

## Known limitations

- Synthetic NIHR data (CC0) — not prospective clinical validation.
- Missingness imputed with train medians; no survival-time feature in classifier.
- Calibrated head deferred to exp_047 e2e chain.
