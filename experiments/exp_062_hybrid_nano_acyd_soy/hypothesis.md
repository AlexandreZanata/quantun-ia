# Hypothesis — EXP 062: Hybrid QNN Head on Frozen LargeNanoMLP (ACYD Brazil Soy)

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`, PennyLane on CPU)

## What I expect to happen

After freezing the **exp_060** `LargeNanoMLP` backbone (~1.16M params) and training only a
**4-qubit re-upload QNN head** (~300 trainable params), val ROC-AUC on **acyd_soy_brazil_v1**
(temporal val 2019–2021) will stay within **−1.0 pp** of the classical sigmoid head
(same backbone, same splits).

## Why I expect this (or not)

- exp_037/051 show hybrid head-only is feasible but rarely beats classical by large margins.
- ACYD agro-climate tabular has heterogeneous features; QNN may not beat a tuned linear head.
- Frozen C4 backbone preserves exp_060 representation; head fine-tune is the fair quantum ablation.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Val ROC-AUC | hybrid ≥ classical head − **1.0 pp** |
| Protocol | Temporal train/val split; `StandardScaler` as exp_060 |
| Backbone | Checkpoint `artifacts/exp_060/large_nano_mlp/seed_42/best.pt` (frozen) |

## Known limitations

- CI uses 5K train / 1K val — not full 50K panel (QNN CPU bottleneck).
- PennyLane sim on CPU; classical backbone on CUDA.
- Test years ≥ 2022 untouched; val-only model selection.
