# Hypothesis — EXP 009: Entanglement Ablation (Basic QNN)

**Date:** 2026-06-16  
**Author:** Quantum ML Lab

## What I expect to happen
Without data re-upload, entanglement topologies (chain, ring) may outperform `none`
on circles — reversing the exp_003 result where `none` won with re-upload.

## Why I expect this
Re-upload adds expressivity independent of entanglement; isolating to a basic QNN
(4 qubits, 2 layers) should reveal whether entanglement helps when re-upload is off.

## What would prove me wrong
- `none` still beats all entangled topologies (entanglement irrelevant on circles)
- All topologies cluster near chance (~50%)
- chain_half significantly beats chain (unlikely without re-upload)

## Metrics I will measure
- [x] Holdout accuracy per topology (10 seeds, bootstrap CI)
- [x] Paired Wilcoxon vs `none` (Holm-Bonferroni corrected)
