# Hypothesis — EXP 014: Sequence Model Baselines

**Date:** 2026-06-17  
**Author:** Quantum ML Lab

## What I expect to happen
On a synthetic sequential binary task, GRU (RNNMini) and TransformerMini will outperform a
flattened-input QNN because they preserve temporal structure.

## Why I expect this
The label depends on cumulative signals across timesteps; flattening destroys order
information that the QNN cannot recover with a single linear pre-layer.

## What would prove me wrong
- Flattened QNN beats both sequence models
- All models cluster near chance (~50%)

## Metrics I will measure
- [x] Holdout accuracy per architecture (10 seeds)
- [x] Paired Wilcoxon vs RNNMini baseline
