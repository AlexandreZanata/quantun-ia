# Hypothesis — EXP 053: Dynamic entanglement schedule on breast cancer

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`, PennyLane on CPU)

## What I expect to happen

Starting with **`entanglement=none`** and linearly increasing to **`ring`** over **5 curriculum
stages** on **Wisconsin breast cancer** (re-upload QNN) improves holdout accuracy by **≥ 1.0 pp**
vs the best fixed topology (`none`, `chain`, or `ring`) — because early training avoids barren
plateaus before adding expressivity.

## Why I expect this

- exp_003/009 rejected **fixed** topology wins on circles/basic QNN.
- **Growing** entanglement is unexplored in our lab and targets plateau timing, not architecture search.
- Breast cancer is small tabular UCI — feasible for multi-stage PennyLane sim on RTX 4060.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Primary | schedule mean holdout ≥ best fixed + **1.0 pp** |
| Statistical | Wilcoxon vs best fixed · Holm (publication seeds) |
| Secondary | Training completes on RTX 4060 without OOM |

## Known limitations

- PennyLane QNN sim on CPU; classical pre/post on CUDA when available.
- CI uses 1 seed and relaxed gate — not a publication claim.
- Test split untouched; holdout accuracy only for gate.
