# Hypothesis — EXP 056: Re-upload depth curriculum ladder (PCA-MNIST → BC → HIGGS)

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`, PennyLane on CPU)

## What I expect to happen

A curriculum that increases **re-upload layers (1→2→3)** synchronized with **margin-based batch
difficulty** beats a **fixed 3-layer** re-upload QNN on **≥ 2 of 3 ladder rungs**
(`pca_mnist_binary`, `breast_cancer`, `higgs_v1` 50k) — with **≥ 0.3 pp** advantage per winning
rung.

## Why I expect this

- exp_008 validated re-upload depth; exp_005 showed naive curriculum fails on circles — real tabular
  ladder may differ.
- Growing depth with easy→hard batches limits barren-plateau exposure early.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Primary | curriculum wins **≥ 2 / 3** rungs |
| Per rung | curriculum ≥ fixed + **0.3 pp** (accuracy or ROC-AUC) |
| Secondary | Training completes on RTX 4060 without OOM |

## Known limitations

- PennyLane QNN sim on CPU; HIGGS uses batched classical path on CUDA.
- CI uses row caps and relaxed gates — not a publication claim.
