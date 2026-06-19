# Hypothesis — EXP 071: Hybrid QNN Head on Frozen LargeNanoMLP (GoBug C3)

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`, PennyLane on CPU)

## What I expect to happen

After freezing the **exp_070** `LargeNanoMLP` backbone (~1.14M params) and training only a
**4-qubit re-upload QNN head** (~300 trainable params), val PR-AUC on **code_defects_gobug_v1**
(temporal val split) will stay within **−1.0 pp** of the classical sigmoid head
(same backbone, same splits).

## Why I expect this (or not)

- exp_051/062 show hybrid head-only is feasible but rarely beats classical by large margins.
- GoBug C3 anchor barely beats logistic (+0.03 pp); QNN head unlikely to add large PR-AUC gain.
- Frozen C3 backbone preserves exp_070 representation; head fine-tune is the fair quantum ablation.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Val PR-AUC | hybrid ≥ classical head − **1.0 pp** |
| Protocol | Temporal train/val split; `StandardScaler` as exp_070 |
| Backbone | Checkpoint `artifacts/exp_070/large_nano_mlp/seed_42/best.pt` (frozen) |

## Known limitations

- CI uses 5K train / 1K val — not full GoBug panel (QNN CPU bottleneck).
- PennyLane sim on CPU; classical backbone on CUDA.
- Software defect benchmark — not production static analysis.
