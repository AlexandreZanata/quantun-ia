# Hypothesis — EXP 073: Quantum warm-start on GoBug hybrid sandwich (C3 replication)

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`, PennyLane on CPU)

## What I expect to happen

Replicating **exp_052/072** on **GoBug code defects** (C3): classical-first training for 70% of epochs,
then enabling the QNN block, yields **≥ +0.5 pp PR-AUC** over end-to-end `HybridSandwich` on temporal val.

## Why I expect this (or not)

- exp_052 (HIGGS) and exp_072 (NIHR) were honest negatives (−0.42 pp, −0.35 pp).
- GoBug has software-defect temporal split; warm-start may not beat e2e hybrid here either.
- Completes C2/C3 warm-start replication row in the grand comparison matrix.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Primary | warm-start val **PR-AUC** ≥ end-to-end hybrid + **0.5 pp** (multi-seed mean) |
| Statistical | Wilcoxon p < 0.05 after Holm · **3 seeds** (publication) |
| Dataset | `code_defects_gobug_v1` temporal val only |

## Known limitations

- PennyLane QNN on CPU; HybridSandwich classical blocks on CUDA.
- CI uses 1 seed and relaxed gate — not a publication claim.
- Mirrors exp_052 sandwich protocol (not frozen C3 backbone).
