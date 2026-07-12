# Hypothesis — EXP 066: Depolarizing noise regularization on ACYD hybrid QNN (H-Q10 / H-Q5)

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`, PennyLane on CPU)

## What I expect to happen

Replicating **H-Q10 / H-Q5** (exp_055 protocol) on **ACYD Brazil soybean**: **depolarizing noise**
during QNN forward improves **ROC-AUC on the temporal test split** (years ≥ 2022) by **≥ 0.5 pp**
vs a noiseless `HybridSandwich`.

## Why I expect this (or not)

- exp_055 on GoBug used PR-AUC on a temporal sha-order holdout; ACYD uses calendar-year temporal test.
- Channel noise may act as a regularizer against climate drift across years.
- Small hybrid head is prone to overfit agro-climate tabular; noise may help or hurt.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Primary | noisy test **ROC-AUC** ≥ noiseless + **0.5 pp** |
| Secondary | Training completes on RTX 4060 without OOM |
| Dataset | `acyd_soy_brazil_v1` temporal test (≥2022) |

## Known limitations

- PennyLane QNN sim on CPU; ACYD tabular pre/post on CUDA.
- Temporal split is year-based; CI uses row caps and relaxed gate — not a publication claim.
