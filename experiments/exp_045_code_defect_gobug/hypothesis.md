# Hypothesis — EXP 045: GoBug file-level defect prediction

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

Training **LargeNanoMLP** on the GoBug file-level subset (~39k rows, ~31% defective)
with **temporal commit-order split** will reach **val PR-AUC ≥ 0.55** and beat
**logistic regression** by **≥ 0.5 pp** PR-AUC.

## Why I expect this

- GoBug combines static + Go-specific metrics (complexity, goroutines, error handling).
- exp_044 showed MLP competitive on tabular; code metrics have nonlinear interactions.
- ~31% prevalence suits PR-AUC as primary metric (roadmap Phase 2 gate).

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Primary | Val **PR-AUC** (nano) ≥ **0.55** |
| Secondary | nano − logistic ≥ **0.5 pp** PR-AUC |
| Split | Commit-order temporal (sorted by `sha`); train 70% / val 15% / test 15% |

## Known limitations

- Subset from go-bug-collector `combined/` (~39k rows, not full IEEE 100k).
- Temporal proxy via `sha` ordering — not wall-clock timestamps in v1 export.
- Hybrid head ablation deferred to exp_050.
