# Hypothesis — EXP 017 (Poisoning × Topology)

**Date:** 2026-06-17  
**Author:** quantun-ia lab

## What I expect to happen

Hybrid **topology** determines poisoning robustness more than raw qubit count. `quantum_first`
will show the **largest accuracy drop** from 0% → 30% poison (quantum block sees raw features
before classical regularization). `hybrid_sandwich` and the EXP 016 NAS preset will degrade
less because classical layers buffer poisoned gradients.

## Why I expect this

EXP 004 showed encoding affects poison robustness; EXP 010 showed depth/LR matter for re-upload.
EXP 016 found `quantum_first` best on clean data — if that topology overfits poisoned labels,
clean-holdout evaluation should punish it hardest at 30% poison.

## What would prove me wrong

- All topologies show < 2 pp drop at 30% poison (insufficient poison signal).
- `classical_first` degrades most (quantum decision layer too sensitive).
- NAS preset is **less** robust than manual `hybrid_sandwich` at 30% poison.

## Metrics I will measure

- [x] Clean holdout accuracy per topology (0% poison, 10 seeds)
- [x] Holdout at 10%, 20%, 30% poison
- [x] `measure_robustness` drop per topology
- [x] Paired Wilcoxon at 0% and 30% poison (Holm-Bonferroni)
- [x] Cohen's d on primary comparison: `quantum_first` vs `hybrid_sandwich` at 30% poison

## Ablation plan

1. Remove `nas_preset` arm (manual topologies only).
2. Reduce poison_rates to [0.0, 0.3] for faster screening.
3. Disable re-upload on all hybrids (angle-only).
