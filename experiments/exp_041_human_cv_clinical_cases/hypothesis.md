# Hypothesis — EXP 041: Human-Interpretable Clinical Case Validation

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Model:** `exp_034` LargeNanoMLP on `synthea_cv_risk_v1`  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Question

When we score **eight fixed patient profiles** grounded in cardiovascular epidemiology,
does the trained nano model **rank risk in the same order** clinicians would expect —
without exposing humans to raw `feature_0…feature_39`?

## Clinical cases (literature-backed)

Cases are ordered by **expected 12-month CV event risk** (low → high). Rationale follows
Framingham / ACC-AHA primary-prevention risk factors: age, sex, smoking, BP, lipids,
diabetes, prior ASCVD, multimorbidity (Whelton et al., 2018; D'Agostino et al., 2008).

| ID | Profile (summary) | Expected tier | Expected rank |
|----|-------------------|---------------|---------------|
| **L01** | 28y female, lean, normotensive, optimal lipids, non-smoker | very_low | 1 |
| **L02** | 34y male, active, normal vitals, non-smoker | very_low | 2 |
| **L03** | 48y female, mildly overweight, borderline BP | low | 3 |
| **L04** | 50y male, overweight, mild HTN, no diabetes | low | 4 |
| **H01** | 58y obese male, smoker, diabetic, hypertensive | high | 5 |
| **H02** | 67y male, prior MI, diabetic smoker | high | 6 |
| **H03** | 76y female, prior stroke, AF, diabetic | very_high | 7 |
| **H04** | 81y male, prior MI + stroke, COPD, CKD, smoker | very_high | 8 |

## What I expect to happen

- Model **risk %** should be **monotonically non-decreasing** with expected rank
  (Spearman ρ ≥ **0.85**); minor ceiling ties at ~100% are acceptable.
- All **L01–L04** (unlikely event) should score **below** all **H01–H04** (likely event):
  `min(H prob) − max(L prob) > 0` (separation gate on probabilities).
- Absolute percentages may be inflated because the Synthea v1 cohort has ~99% label prevalence;
  **relative ordering** is the primary human-facing validation.

## Success criteria

- All 8 cases score on RTX 4060 without checkpoint error.
- Spearman ρ(expected_rank, model probability) ≥ **0.85**.
- Separation gate: `min(H01–H04 prob) > max(L01–L04 prob)`.
- Results logged to `logs/experiments.jsonl` and documented in `results.md`.

## Known limitations

- Synthetic training data — not clinical deployment.
- Model outputs probability, not a calibrated clinical risk calculator.
- Latent feature fixed at 0 for hand-entered profiles (neutral).

## References

- Whelton PK et al. 2017 ACC/AHA hypertension guideline. *Hypertension* 2018.
- D'Agostino RB Sr et al. General cardiovascular risk profile. *Circulation* 2008.
- Arnett DK et al. 2019 ACC/AHA primary prevention guideline. *Circulation* 2019.
