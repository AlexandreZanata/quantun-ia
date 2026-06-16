# Hypothesis — EXP 004: Data Poisoning

**Date:** 2026-06-16  
**Author:** Quantum ML Lab

## What I expect to happen
Under label poisoning on the training set (0–30%), classical MLP will degrade more
gracefully than quantum models. Amplitude encoding will outperform angle encoding on
clean holdout because it uses more Hilbert space. Evaluation always uses a clean 30%
holdout split.

## Why I expect this
Quantum models have fewer parameters and may overfit poisoned labels. Amplitude
encoding carries more information per qubit than angle encoding.

## What would prove me wrong
- Quantum amplitude more robust than classical at 30% poison (smaller accuracy drop)
- Angle encoding reaches ≥ 60% clean holdout on circles
- Poison has no measurable effect at 30% (data leakage suspected)

## Metrics I will measure
- [x] Holdout accuracy at each poison rate (10 seeds)
- [x] Accuracy drop from 0% to 30% poison per model
- [ ] Paired Wilcoxon classical vs amplitude at 30% poison
