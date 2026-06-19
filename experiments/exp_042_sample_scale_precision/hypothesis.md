# Hypothesis — EXP 042: Sample-Scale Precision Curve (Synthea CV)

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Model:** `exp_034` LargeNanoMLP on `synthea_cv_risk_v1`  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Question

How stable is holdout **precision, recall, F1, and ROC-AUC** for the published
Synthea CV serve model when we evaluate on stratified subsamples of **100, 200, …, 2000**
validation rows — and what does a fixed **100-prediction** export look like in detail?

## What I expect to happen

- Val split has **~0.36% negatives** — stratified n=100 yields **0 negatives**; balanced fallback enforces **≥10**.
- For n≥200, stratified subsample keeps natural prevalence (e.g. 7 neg @ n=2000).
- **Accuracy / precision** will look high (~99%+) because of class imbalance; report **Neg/Pos counts**, **PR-AUC**, and **ECE** per n.
- **ROC-AUC @ n=2000** should match Model Lab (~**0.81**) with gate ≥ **0.78**.
- The **100-row export** should match the n=100 point on the curve (same seed, stratified draw).

## Success criteria

- All 20 sample sizes evaluate on RTX 4060 without checkpoint error.
- ROC-AUC @ n=**2000** ≥ **0.78**.
- 100 holdout predictions exported to `.local/out/predictions_100_synthea_cv.json`.
- Full curve exported to `.local/out/sample_scale_precision.json`.
- Results logged to `logs/experiments.jsonl` and documented in `results.md`.

## Known limitations

- Subsamples use sklearn **stratified** draws (same as Model Lab) — independent per n.
- Synthetic cohort prevalence ≠ real-world CV event rates.
- At n=100, ROC-AUC may be **undefined** (single class) — document, do not gate on it.
