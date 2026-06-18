# Hypothesis — EXP 034: LargeNanoMLP on Synthea CV Risk (700K train)

**Date:** 2026-06-18  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

A **~1.17M-parameter** `LargeNanoMLP` trained on the **synthea_cv_risk_v1** train split
(700K rows, 40 features) will achieve **validation ROC-AUC at least 1 pp above** logistic
regression on the same scaled features.

## Why I expect this

- Synthetic EHR tabular signal is nonlinear (conditions, vitals, labs interactions).
- 700K rows with dropout 0.3 matches the Phase L regularization recipe validated on HIGGS.
- Same serve path as exp_033 — scaler fit on train only, val held out for selection.

## What would prove me wrong

- Val ROC-AUC ≤ logistic + 1 pp after epoch budget
- OOM on RTX 4060 at batch 2048
- Train accuracy >> val AUC → overfitting

## Metrics I will measure

- [ ] Parameter count (target ≥ 1,000,000)
- [ ] Logistic val ROC-AUC (baseline)
- [ ] LargeNanoMLP val ROC-AUC (primary)
- [ ] Δ AUC in percentage points
- [ ] Wall-clock elapsed on RTX 4060

## Success criteria

- `n_params` ≥ **1,000,000**
- Training completes without OOM on RTX 4060 (700K train rows)
- Val AUC advantage reported vs logistic (may be negative — honest negative accepted for L7)
- Serve path (`exp_035`) parity gate passes after checkpoint publish

## Known limitations

- Synthea is **synthetic** — logistic may outperform deep MLP on val (document in `results.md`).
- Test split not used for model selection in this gate.
