# Results — EXP 041: Human Clinical Case Validation

**Run date:** 2026-06-18  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Model:** `exp_034` LargeNanoMLP · `synthea_cv_risk_v1` · seed 42

## Summary

| Metric | Value | Gate |
|--------|-------|------|
| Spearman ρ (rank vs risk %) | **0.9762** | ≥ 0.85 |
| Max risk — low tier (L01–L04) | **99.91%** | — |
| Min risk — high tier (H01–H04) | **100.00%** | — |
| Separation (min H − max L) | **+0.089 pp** | > 0 |
| Strict monotonic (ε=1e-4) | **True** | informational |
| Elapsed | **1.0s** | — |

## Verdict
**accepted** — model ordering vs literature-backed clinical expectation.

## Case-by-case results

| ID | Expected rank | Tier | Model risk % | Band | Title |
|----|---------------|------|--------------|------|-------|
| **L01** | 1 | very_low | **97.59%** | high | Young healthy woman |
| **L02** | 2 | very_low | **99.24%** | high | Active man in his 30s |
| **L03** | 3 | low | **99.55%** | high | Middle-aged woman, borderline factors |
| **L04** | 4 | low | **99.91%** | high | 50-year-old man, mild hypertension |
| **H01** | 5 | high | **100.00%** | high | Obese smoker with diabetes and HTN |
| **H02** | 6 | high | **100.00%** | high | Post-MI diabetic smoker |
| **H03** | 7 | very_high | **100.00%** | high | Elderly woman, stroke and atrial fibrillation |
| **H04** | 8 | very_high | **100.00%** | high | Multimorbid elderly man |

## Science notes (expected direction)

- **L01** — Young female, optimal BMI/BP/lipids, non-smoker — pooled-cohort 10y ASCVD risk typically <5% (Arnett et al., 2019).
- **L02** — Age 34, normal BMI and BP, favorable HDL — low short-term ASCVD event rate in primary prevention cohorts.
- **L03** — Age 48, mild overweight, BP near 130/80 — below treatment threshold but above optimal (ACC/AHA 2017).
- **L04** — Stage-1 hypertension range (130–139 systolic), overweight — intermediate lifetime risk without ASCVD history.
- **H01** — Smoking + diabetes + obesity + hypertension — multiplicative risk per Framingham and ACC/AHA charts.
- **H02** — Secondary-prevention profile: prior MI and diabetes — high recurrent event risk (secondary prevention guidelines).
- **H03** — Prior stroke + AF + diabetes in late 70s — very high 1y recurrence/stroke risk (CHADS-VASc context).
- **H04** — Age 81 with prior MI, stroke, COPD, CKD, smoker — maximal epidemiologic risk cluster.

## Interpretation for humans

- **L01–L04** are profiles where a short-term CV event is *unlikely* by epidemiology.
- **H01–H04** are profiles where an event is *likely* (prior ASCVD, multimorbidity, smoking).
- The Synthea v1 cohort has ~99% positive prevalence, so absolute % looks high for everyone.
- **What we validated:** the model *ranks* patients in clinically sensible order.
- **Ceiling note:** H02–H04 saturate near 100%; tiny ordering noise there is not clinically meaningful.

## Limitations

- Synthetic training data; not calibrated for real-world prevalence.
- Hand-entered profiles use `latent_risk_factor = 0` (neutral).
- Does not replace ACC/AHA pooled cohort equations or clinician judgment.
