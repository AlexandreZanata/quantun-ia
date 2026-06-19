# Hypothesis — EXP 072: Quantum warm-start on NIHR hybrid sandwich (C2 replication)

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`, PennyLane on CPU)

## What I expect to happen

Replicating **exp_052** on **NIHR synthetic CV** (C2): classical-first training for 70% of epochs,
then enabling the QNN block, yields **≥ +0.5 pp PR-AUC** over end-to-end `HybridSandwich` on val.

## Why I expect this (or not)

- exp_052 on HIGGS was **−0.42 pp** (honest negative) — warm-start may not transfer to clinical tabular.
- NIHR has low prevalence (~8%); PR-AUC is the primary metric (same as C2 anchors).
- Cross-domain replication tests whether H-Q2 failure generalizes.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Primary | warm-start val **PR-AUC** ≥ end-to-end hybrid + **0.5 pp** (multi-seed mean) |
| Statistical | Wilcoxon p < 0.05 after Holm · **3 seeds** (publication) |
| Dataset | `nihr_cv_synthetic_v1` temporal val only |

## Known limitations

- PennyLane QNN on CPU; HybridSandwich classical blocks on CUDA.
- CI uses 1 seed and relaxed gate — not a publication claim.
- Not using frozen C2 backbone — mirrors exp_052 sandwich protocol for fair cross-domain replication.
