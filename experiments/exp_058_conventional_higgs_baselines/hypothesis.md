# Hypothesis — EXP 058: Conventional tabular baselines vs LargeNanoMLP on HIGGS

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

On **HIGGS v1** (805K train / 172.5K val, 28 features), the shipped **LargeNanoMLP**
(`exp_032`, ~1.14M params) achieves **validation ROC-AUC at least 0.5 pp above** the
best conventional baseline among:

- `sklearn` LogisticRegression
- `sklearn` MLPClassifier (2048→512→64, matched topology)
- `sklearn` HistGradientBoostingClassifier
- `xgboost` XGBClassifier (shallow, depth 3)

## Why I expect this

- `exp_032` already beat logistic by **+14.09 pp** on full val with PyTorch batched training.
- Conventional deep tabular stacks (sklearn MLP, gradient boosting) often under-use large-batch
  regularization (dropout 0.3, weight decay 1e-4) that `LargeNanoMLP` uses.
- This experiment isolates **training-stack advantage** — not quantum — for paper/external review.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Val ROC-AUC | LargeNanoMLP ≥ best conventional + **0.5 pp** |
| Protocol | Same train/val split and `StandardScaler` as `exp_032` |
| Our model | Shipped checkpoint `dist/serve_models/large_nano_mlp_higgs/best.pt` (no retrain) |

## Known limitations

- Single seed (42) — infrastructure comparison, not multi-seed claim.
- sklearn MLP on full 805K rows is CPU-bound and slow vs PyTorch CUDA path.
- Test split untouched; val-only selection aligned with `exp_032`.
- HistGradientBoosting / XGBoost hyperparameters are shallow defaults, not HPO-tuned.
