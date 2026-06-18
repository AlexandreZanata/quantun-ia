# Hypothesis — EXP 039: Regularized LargeNanoMLP on Synthea CV

**Date:** 2026-06-18  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

**exp_034** lost to logistic by **−0.82 pp** val AUC (0.7867 vs 0.7949) on 700K Synthea rows —
likely overfitting at ~1.17M params. Increasing **dropout (0.3→0.5)**, **weight decay (1e-4→1e-3)**,
and lowering **LR (0.001→0.0005)** will **close the gap** and match or beat logistic val AUC.

## Why I expect this (or not)

- Synthea has ~0.7 samples/param at 700K rows — heavy regularization is the standard fix.
- HIGGS at scale beat logistic (+1 pp) with dropout 0.3 — Synthea may remain harder (synthetic EHR noise).
- Honest negative remains possible if logistic is simply the right inductive bias.

## Success criteria

- Train completes on RTX 4060 without OOM (700K rows, batch 2048)
- Report val AUC vs logistic **and** vs **exp_034** nano AUC (0.7867 reference)
- **Accept:** val AUC ≥ exp_034 nano + **0.0 pp** AND ≥ logistic − **2.0 pp** (infrastructure)
- **Stretch:** val AUC ≥ logistic (beat linear baseline)

## Known limitations

- Same architecture size (~1.17M params) — regularization only, not width ablation
- Test split untouched; val AUC only for gate
