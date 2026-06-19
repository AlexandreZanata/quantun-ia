# Hypothesis — EXP 077: Conventional tabular baselines vs LargeNanoMLP on GoBug code defects

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

On **code_defects_gobug_v1** (27,172 train / 5,822 val, 23 features, temporal split), the
**exp_070 LargeNanoMLP** checkpoint (~1.14M params) achieves **validation PR-AUC at least
0.5 pp above** the best conventional baseline among:

- `sklearn` LogisticRegression
- `sklearn` MLPClassifier (2048→512→64, matched topology)
- `sklearn` HistGradientBoostingClassifier
- `xgboost` XGBClassifier (shallow, depth 3)

## Why I expect this

- exp_070 full nano template tied logistic (+0.03 pp); gradient boosting may capture code-metric interactions.
- PR-AUC is primary for imbalanced defect detection (~15% positive rate).
- Mirrors **exp_058** / **exp_076** protocol for honest classical-vs-nano comparison on anchor **C3**.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Val PR-AUC | LargeNanoMLP ≥ best conventional + **0.5 pp** |
| Protocol | Same temporal train/val split and `StandardScaler` as `exp_070` |
| Our model | Checkpoint `artifacts/exp_070/large_nano_mlp/seed_42/best.pt` (no retrain) |

## Known limitations

- Single seed (42) — infrastructure comparison, not multi-seed claim.
- GoBug combined subset (~39K rows); temporal proxy via commit-sha ordering.
- exp_070 already failed PR-AUC vs logistic gate (+0.03 pp); conventional panel may beat nano.
