# Hypothesis — EXP 005: Quantum Curriculum Learning

**Date:** 2026-06-16  
**Author:** Quantum ML Lab

## What I expect to happen
With a learnable re-upload QNN base (4q, 3 layers), margin-based curriculum
(easy→hard batches) will outperform random example ordering on circles holdout.
The learnability gate must pass (random baseline mean ≥ 55%) before curriculum runs.

## Why I expect this
Curriculum learning stabilizes training on hard examples. Ordering by distance to
class centroid (easy-first) should prevent early collapse on noisy circle boundaries.

## What would prove me wrong
- Random baseline fails learnability gate → curriculum N/A
- margin_batches does not beat random (Wilcoxon p ≥ 0.05)
- margin_batches significantly worse than random (honest negative)

## Metrics I will measure
- [x] Applicability gate (random baseline vs 55% threshold)
- [x] Holdout accuracy: margin_batches vs random (10 seeds, bootstrap CI)
- [x] Paired Wilcoxon margin_batches vs random
