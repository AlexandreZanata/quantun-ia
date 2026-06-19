# Hypothesis — EXP 061: Conventional tabular baselines vs LargeNanoMLP on ACYD Brazil soybean

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

On **acyd_soy_brazil_v1** (50,107 train / 5,830 val, 37 features, temporal split), the
**exp_060 LargeNanoMLP** checkpoint (~1.16M params) achieves **validation ROC-AUC at least
0.5 pp above** the best conventional baseline among:

- `sklearn` LogisticRegression
- `sklearn` MLPClassifier (2048→512→64, matched topology)
- `sklearn` HistGradientBoostingClassifier
- `xgboost` XGBClassifier (shallow, depth 3)

## Why I expect this

- `exp_060` already beat logistic by **+3.86 pp** on temporal val with PyTorch batched CUDA training.
- Agro-climate tabular features are heterogeneous (soil static + weekly climate stats); gradient
  boosting may compete but often underperforms deep MLPs with dropout on medium-scale panels.
- Mirrors **exp_058** (HIGGS) protocol for honest classical-vs-nano comparison on anchor **C4**.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Val ROC-AUC | LargeNanoMLP ≥ best conventional + **0.5 pp** |
| Protocol | Same train/val temporal split and `StandardScaler` as `exp_060` |
| Our model | Checkpoint `artifacts/exp_060/large_nano_mlp/seed_42/best.pt` (no retrain) |

## Known limitations

- Single seed (42) — infrastructure comparison, not multi-seed claim.
- sklearn MLP on ~50K rows is CPU-bound and slower than PyTorch CUDA path.
- Test split (crop-years ≥ 2022) untouched; val-only selection aligned with `exp_060`.
- HistGradientBoosting / XGBoost hyperparameters are shallow defaults, not HPO-tuned.
