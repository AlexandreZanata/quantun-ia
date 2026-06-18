# Hypothesis — EXP 036: Training Methodology Ablation on HIGGS

**Date:** 2026-06-18  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

On **HIGGS v1** with `LargeNanoMLP` (~1.17M params), at least one alternative trainer
(**curriculum**, **adaptive LR**, or **champion loop**) will beat plain Adam mini-batch
**baseline** val ROC-AUC by **≥0.5 pp** on a 3-seed CI slice (50K train / 10K val).

## Why I expect this (or not)

- BC curriculum gained +0.18 pp (exp_031) — may **not** transfer to physics tabular.
- Circles curriculum was **rejected** (−4.9 pp) — honest negatives exist in this repo.
- Adaptive LR was inconclusive on QNN (exp_015) — may help deep MLP convergence.
- Champion loop accepted on BC (exp_027) — stable promotion at scale is plausible.

## Methods compared (paired seeds)

| Run ID | Trainer | CI epochs |
|--------|---------|-----------|
| `baseline` | Adam mini-batch | 20 |
| `curriculum` | margin staged batches + refine | 60 matched |
| `adaptive` | gradient-variance LR (batched) | 50 |
| `champion` | two-cycle retrain, keep best val AUC | 30 |

## Success criteria

- All four methods complete on RTX 4060 without OOM (50K-row CI slice)
- Paired comparison vs baseline logged via `compare_conditions_batch`
- **Accept:** any alternative mean val AUC ≥ baseline + **0.5 pp** on CI (3 seeds)
- **Else:** document honest negative in `results.md` (expected for curriculum on HIGGS)

## Known limitations

- CI uses 50K-row slice — not full 805K train (RTX 4060 time budget)
- Val AUC only — test split untouched
- Champion sim is 2-cycle proxy, not full 4-week continuous_train
