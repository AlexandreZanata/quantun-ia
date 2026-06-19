# Hypothesis — EXP 076: Conventional tabular baselines vs LargeNanoMLP on NIHR synthetic CV

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

On **nihr_cv_synthetic_v1** (70K train / 15K val, 13 features, ~8% prevalence), the
**exp_069 LargeNanoMLP** checkpoint (~1.11M params) achieves **validation PR-AUC at least
0.5 pp above** the best conventional baseline among:

- `sklearn` LogisticRegression
- `sklearn` MLPClassifier (2048→512→64, matched topology)
- `sklearn` HistGradientBoostingClassifier
- `xgboost` XGBClassifier (shallow, depth 3)

## Why I expect this

- Deep MLP with dropout may capture nonlinear clinical risk interactions beyond linear/logistic models.
- PR-AUC is primary for imbalanced CV risk (~8% event rate); ROC-AUC can mask ranking on rare positives.
- Mirrors **exp_058** (HIGGS) and **exp_061** (ACYD) protocol for honest classical-vs-nano comparison on anchor **C2**.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Val PR-AUC | LargeNanoMLP ≥ best conventional + **0.5 pp** |
| Protocol | Same train/val split and `StandardScaler` as `exp_069` |
| Our model | Checkpoint `artifacts/exp_069/large_nano_mlp/seed_42/best.pt` (no retrain) |

## Known limitations

- Single seed (42) — infrastructure comparison, not multi-seed claim.
- sklearn MLP on 70K rows is CPU-bound; PyTorch nano eval uses CUDA.
- exp_069 already failed PR-AUC vs logistic alone (−0.12 pp); conventional sweep may beat nano on full panel.
