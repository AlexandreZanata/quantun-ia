# Hypothesis — EXP 021: PennyLane Backend Parity

**Date:** 2026-06-17  
**Author:** Quantum ML Lab  
**Pre-registration:** https://osf.io/8k2nf (PennyLane backend parity on breast cancer QNN)

## What I expect to happen

On breast cancer (30 features, binary), an angle-encoding QNN trained with identical
hyperparameters on `default.qubit` and `lightning.qubit` will produce holdout accuracies
within **2 percentage points** (pp) across 10 seeds. Any difference is numerical noise or
simulator implementation detail, not a meaningful modeling effect.

## Why I expect this

Both backends implement the same variational circuit; `lightning.qubit` uses the adjoint
gradient rule while `default.qubit` uses backprop — holdout accuracy should still match.
exp_011 already shows stable QNN behaviour on this dataset; backend choice should affect
wall-clock time, not predictive accuracy.

## What would prove me wrong

- Mean holdout accuracy differs by **> 2 pp** between backends with Holm-significant Wilcoxon
- Per-seed paired deltas show systematic bias (e.g. lightning always higher)
- Gradient norms diverge materially at epoch 1 (implementation bug signal)

## Metrics I will measure

- [x] Holdout accuracy (10 seeds, bootstrap 95% CI) per backend
- [x] Paired Wilcoxon: `default.qubit` vs `lightning.qubit`
- [x] Cohen's d on paired seed deltas
- [x] Wall-clock training time per backend (diagnostic, not primary claim)

## Ablation follow-ups

- Halve `n_layers` to 1 — does parity hold with fewer parameters?
- Swap to `exp_012` PCA-MNIST subset — does parity hold on image-derived tabular data?
