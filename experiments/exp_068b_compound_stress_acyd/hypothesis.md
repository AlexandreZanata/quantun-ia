# Hypothesis — EXP 068b: Compound Stress Label on ACYD (H-Q12)

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`, PennyLane on CPU)

## What I expect to happen

On a **compound stress** binary label — low yield AND (drought proxy OR heat stress) — a frozen
**C4** backbone with trainable 4-qubit QNN head reaches val **ROC-AUC ≥ logistic + 1.0 pp**
on temporal val (2019–2021).

## Why I expect this (or not)

- Compound labels encode drought∧heat interactions that plain logistic may miss on tabular features.
- exp_062 hybrid head was −0.19 pp on the default yield label — interaction gains are uncertain.
- Drought uses train-fitted seasonal precipitation z-score (SPEI proxy); heat uses Tmax > 35 °C weeks ≥ 3.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Primary | hybrid val ROC-AUC ≥ logistic + **1.0 pp** |
| Secondary | logistic and hybrid val ROC-AUC logged |
| Protocol | Temporal val 2019–2021; compound label from raw ACYD weather |
| Backbone | Frozen `artifacts/exp_060/.../best.pt` (head-only training) |

## Known limitations

- SPEI approximated via train-set precipitation z-score (ERA5-Drought join deferred to Phase 4).
- Highly imbalanced compound label — stratified subsampling in CI only.
- Single seed (42) publication; multi-seed deferred.
