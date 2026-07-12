# Hypothesis — EXP 083: Conventional tabular baselines vs LargeNanoMLP on ACYD Brazil maize

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

On **acyd_maize_brazil_v1** (151,956 train / 13,566 val, 37 features, temporal split), the
**exp_081 LargeNanoMLP** checkpoint (~1.16M params) achieves **validation ROC-AUC at least
0.5 pp above** the best conventional baseline among:

- `sklearn` LogisticRegression
- `sklearn` MLPClassifier (2048→512→64, matched topology)
- `sklearn` HistGradientBoostingClassifier
- `xgboost` XGBClassifier (shallow, depth 3)

## Why I expect this

- `exp_081` already beat logistic by **+11.03 pp** on temporal val with PyTorch batched CUDA training.
- Completes the Phase 2 / C4b classical floor for maize (mirrors **exp_061** on soybean).
- Gradient boosting may still compete on agro tabular panels — honest negative acceptable.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Val ROC-AUC | LargeNanoMLP ≥ best conventional + **0.5 pp** |
| Protocol | Same train/val temporal split and `StandardScaler` as `exp_081` |
| Our model | Checkpoint `artifacts/exp_081/large_nano_mlp/seed_42/best.pt` (no retrain) |

## Known limitations

- Single seed (42) — infrastructure comparison, not multi-seed claim.
- sklearn MLP on ~152K rows is CPU-bound and slower than PyTorch CUDA path.
- Test split (crop-years ≥ 2022) untouched; val-only selection aligned with `exp_081`.
- HistGradientBoosting / XGBoost hyperparameters are shallow defaults, not HPO-tuned.
