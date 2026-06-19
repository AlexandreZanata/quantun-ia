# Results — EXP 043: Isotonic Calibration (Synthea CV)

**Run date:** 2026-06-18  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Model:** `exp_034` LargeNanoMLP · `synthea_cv_risk_v1` · seed 42

## Summary

| Metric | Value | Gate |
|--------|-------|------|
| Balanced rows | **1000** | ≥ 500 |
| Negatives | **100** | ≥ 50 |
| ECE before | **0.0962** | — |
| ECE after | **0.0443** | ≤ 0.085 |
| Spearman ρ (clinical) | **0.9386** | ≥ 0.85 |
| Elapsed | **0.9s** | — |

## Verdict
**accepted** — isotonic calibration preserves ranking while improving ECE.

## Artifact

- `/data/dev/projects/webstorm/quantun-ia/artifacts/exp_043/large_nano_mlp_synthea_cv_risk_v1/seed_42/calibration_isotonic.json`

## Limitations

- Synthetic Synthea cohort; calibrated % not for clinical use.
- Isotonic fit on balanced research slice only.
