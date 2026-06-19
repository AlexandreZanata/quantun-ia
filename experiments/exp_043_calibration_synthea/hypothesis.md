# Hypothesis — EXP 043: Probability Calibration (Synthea CV)

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Model:** `exp_034` LargeNanoMLP on `synthea_cv_risk_v1`  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Question

Can **isotonic calibration** on a balanced val slice improve **expected calibration error (ECE)**
without destroying **clinical case ranking** (Spearman ρ on exp_041 profiles)?

## What I expect to happen

- Raw model probabilities cluster near 0.97–1.0 (Synthea ~99% positive prevalence).
- Balanced subsample (≥100 negatives / 1000 rows) enables meaningful calibration fit.
- Isotonic regression is monotonic → Spearman ρ on 8 clinical cases stays ≥ **0.85**.
- ECE on held-out balanced slice drops to ≤ **0.085** after calibration.

## Success criteria

- Fit isotonic calibrator on RTX 4060 batch inference (no checkpoint error).
- ECE after calibration ≤ **0.085** on balanced eval holdout.
- Spearman ρ (raw vs calibrated clinical risks) ≥ **0.85**.
- Artifact saved to `artifacts/exp_043/.../calibration_isotonic.json`.
- Results logged to `logs/experiments.jsonl` and documented in `results.md`.

## Known limitations

- Calibration slice is synthetic Synthea — not real-world prevalence.
- Absolute calibrated % still research-only; ranking remains primary human metric.
- Platt scaling deferred; isotonic chosen for monotonicity guarantee.
