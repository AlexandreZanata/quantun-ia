# Hypothesis — EXP 035: LargeNanoMLP Serve Parity (Synthea 10K)

**Date:** 2026-06-18  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

After publishing the `exp_034` checkpoint with train-fitted scaler to the nanotrainer
serve path (`large_nano_mlp × synthea_cv_risk_v1`), **batch_predict**, **REST API**, and
**score_synthea_cv_risk** chatbot tool return identical probabilities on a **10,000-row**
Synthea test slice (max |Δp| < 1e-5 per row).

## Why I expect this

- exp_033 validated the same path for HIGGS after the `TrainableMixin.eval()` fix.
- All three surfaces route through `predict_nanomodel.execute` with identical DTOs.

## What would prove me wrong

- Missing or mismatched scaler vs training
- Any surface bypasses the shared predict path
- max |Δp| ≥ 1e-5 on any row

## Success criteria

- All pairwise max |Δp| < **1e-5** on publication holdout (10K rows)
- CI smoke passes on 500 rows
- `make check-real` stays green (16+ real tests)

## Known limitations

- Synthetic EHR tabular — infrastructure validation only
