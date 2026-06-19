# Hypothesis — EXP 074: Dynamic entanglement schedule on NIHR (C2 replication)

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`, PennyLane on CPU)

## What I expect to happen

Replicating **exp_053** on **NIHR clinical tabular** (C2): a curriculum from **`entanglement=none`**
to **`ring`** over **5 stages** on a re-upload `QuantumNetEntangled` yields **≥ +0.5 pp val PR-AUC**
over the best fixed topology (`none`, `chain`, or `ring`).

## Why I expect this (or not)

- exp_053 on breast cancer was an honest negative (−0.78 pp holdout accuracy vs `none`).
- NIHR has 13 imbalanced clinical features — dynamic entanglement may not beat fixed ring/none here either.
- Completes the NIHR row for H-Q3 entanglement schedule in the grand comparison matrix.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Primary | schedule mean val **PR-AUC** ≥ best fixed + **0.5 pp** (multi-seed mean) |
| Statistical | Wilcoxon vs best fixed · Holm · **3 seeds** (publication) |
| Dataset | `nihr_cv_synthetic_v1` val split only |

## Known limitations

- PennyLane QNN sim on CPU; classical pre/post on CUDA when available.
- Publication row cap (10K train) keeps PennyLane epoch cost feasible on RTX 4060.
- CI uses 1 seed and relaxed gate — not a publication claim.
- Standalone `QuantumNetEntangled` (not frozen C2 backbone).
