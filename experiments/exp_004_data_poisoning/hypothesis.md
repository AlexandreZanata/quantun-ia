# Hypothesis — EXP 004: Data Poisoning

**Date:** 2026-06-16  
**Author:** Quantum ML Lab

## What I expect to happen
Under label poisoning on the training set (0–30%), classical MLP will degrade more
gracefully than quantum models. **Re-upload** (project baseline) is compared against
amplitude encoding on clean holdout. Evaluation always uses a clean 30% holdout split.

## Why I expect this
Quantum models have fewer parameters and may overfit poisoned labels. Amplitude
encoding carries more information per qubit than angle encoding.

## What would prove me wrong
- Quantum amplitude more robust than classical at 30% poison (smaller accuracy drop)
- Re-upload reaches ≥ 60% clean holdout and beats amplitude significantly
- Poison has no measurable effect at 30% (data leakage suspected)

## Metrics I will measure
- [x] Holdout accuracy at each poison rate (10 seeds)
- [x] Accuracy drop from 0% to 30% poison per model
- [x] Paired Wilcoxon classical vs reupload and amplitude vs reupload (Holm corrected)
