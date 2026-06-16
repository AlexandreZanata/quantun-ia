# Hypothesis — EXP 010: Poisoning Re-upload Ablation

**Date:** 2026-06-16  
**Author:** Quantum ML Lab

## What I expect to happen
Re-upload underperformed amplitude in exp_004 due to over-capacity or LR mismatch.
Fewer layers (2 vs 3) or lower learning rate (0.01) should improve clean holdout
and poisoning robustness for re-upload models.

## Why I expect this
3-layer re-upload on poisoned training labels may overfit flipped labels; reducing
depth or LR regularizes the variational circuit.

## What would prove me wrong
- reupload_3l still best among re-upload variants
- All re-upload variants stay below amplitude at 0% poison
- LR change has no effect (optimizer already at plateau)

## Metrics I will measure
- [x] Holdout accuracy at 0% and 30% poison per variant (10 seeds)
- [x] Paired Wilcoxon vs reupload_3l baseline (Holm-Bonferroni corrected)
