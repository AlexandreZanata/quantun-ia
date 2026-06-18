# Hypothesis — EXP 033: LargeNanoMLP Serve Parity (HIGGS 10K)

**Date:** 2026-06-18  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

After publishing the `exp_032` LargeNanoMLP checkpoint with train-fitted scaler to the
nanotrainer serve path (`large_nano_mlp` × `higgs_v1`), **batch_predict**, **REST API**, and
**chatbot tool** will return identical probabilities on a **10,000-row HIGGS holdout slice**
(max |Δp| < 1e-5 per row).

## Why I expect this

- exp_028/029 validated the same `predict_nanomodel.execute` path for hybrid_sandwich.
- All three surfaces route through identical DTOs once the serve artifact includes scaler + weights.
- HIGGS features follow `tabular_binary_v1` (`feature_0..27` raw, scaler applied at inference).

## What would prove me wrong

- Missing or mismatched scaler vs training → systematic drift
- Any surface bypasses `predict_nanomodel` → row-level divergence
- max |Δp| ≥ 1e-5 on any row across batch/API/tool triple

## Metrics I will measure

- [ ] Serve artifact published (`best.pt` + `scaler.joblib` + `input_dim=28`)
- [ ] max |Δp| batch vs API on N rows
- [ ] max |Δp| tool vs API on N rows
- [ ] Wall-clock batch vs API on RTX 4060

## Success criteria

- All pairwise max |Δp| < **1e-5** on publication holdout (10K rows)
- CI smoke passes on 500 rows (CPU or CUDA)
- `make check-real` stays green (13+ real tests)

## Known limitations

- HIGGS physics tabular — infrastructure validation only
- Test split subsample — no test-set leakage claim beyond serve parity
