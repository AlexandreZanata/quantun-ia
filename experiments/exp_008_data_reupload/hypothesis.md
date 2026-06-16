# Hypothesis — EXP 008: Data Re-uploading QNN on Circles

**Date:** 2026-06-16  
**Author:** Quantum ML Lab

## What I expect to happen
On the circles dataset (noise=0.2, n=500), a data re-uploading QNN will outperform the
standard single-upload QNN from exp_001 because re-embedding features at each variational
layer gives the circuit more expressive power without increasing qubit count.

A parameter-matched classical MLP should still match or beat both quantum models when
parameter counts are equalized.

## Why I expect this
Data re-uploading is a known technique to mitigate barren plateaus and shallow circuits.
Our exp_001 QNN plateaued near chance (~53%); re-uploading may unlock non-linear decision
boundaries needed for concentric circles.

## What would prove me wrong
- Re-upload QNN holdout mean stays below 55% (still at chance)
- Re-upload does not significantly beat basic QNN (Wilcoxon p ≥ 0.05)
- Parameter-matched classical beats re-upload by > 5 pp with p < 0.05

## Metrics I will measure
- [x] Holdout accuracy (10 seeds, bootstrap 95% CI)
- [x] Paired Wilcoxon: reupload vs basic, reupload vs classical_matched
- [x] Parameter count per model (logged to JSONL)
- [x] Training time per seed
