# Hypothesis — EXP 064: Dynamic entanglement schedule on ACYD (C4 / H-Q3)

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`, PennyLane on CPU)

## What I expect to happen

Replicating **H-Q3** (exp_053 / exp_074 protocol) on **ACYD agro-climate tabular** (C4): a curriculum
from **`entanglement=none`** to **`ring`** over schedule stages on a re-upload `QuantumNetEntangled`
yields **≥ +0.5 pp val ROC-AUC** over the best fixed topology (`none`, `chain`, or `ring`).

## Why I expect this (or not)

- exp_053 on breast cancer and exp_074 on NIHR were honest negatives vs best fixed topology.
- ACYD has 37 climate/soil features — dynamic entanglement may not beat fixed ring/none here either.
- Completes the ACYD row for H-Q3 entanglement schedule in the grand comparison matrix.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Primary | schedule mean val **ROC-AUC** ≥ best fixed + **0.5 pp** (multi-seed mean) |
| Statistical | Wilcoxon vs best fixed · Holm · **3 seeds** (publication) |
| Dataset | `acyd_soy_brazil_v1` val split only |

## Known limitations

- PennyLane QNN sim on CPU; classical pre/post on CUDA when available.
- Publication row cap keeps PennyLane epoch cost feasible on RTX 4060.
- CI uses 1 seed and relaxed gate — not a publication claim.
- Standalone `QuantumNetEntangled` (not frozen C4 backbone).
