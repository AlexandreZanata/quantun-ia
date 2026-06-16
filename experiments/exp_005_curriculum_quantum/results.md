# Results — EXP 005

**Date:** 2026-06-16  
**Config:** 3 seeds, 4 stages × 12 epochs + 12 refine, QNN 4q/1L LR 0.02  
**Stats:** Wilcoxon margin_batches vs random

## What happened

| Method | Mean holdout | Std | 95% CI |
|--------|-------------|-----|--------|
| random | **81.1%** | ±2.7% | [77.8%, 84.4%] |
| margin_batches | 80.7% | ±5.2% | [73.3%, 84.4%] |

**Paired test** margin_batches vs random: Δ=−0.4%, p=1.0 → **not significant**.

Multi-seed evaluation shows both methods are equivalent on moons — curriculum neither helps nor hurts statistically.

## Comparison with hypothesis

Easy-first curriculum does not significantly improve QNN generalization vs random shuffle with 3 seeds.

## Unexpected finding

Single-seed runs were misleading (random 50% vs margin 81%). Multi-seed + Wilcoxon is essential for honest conclusions.

## Suggested next experiment

- 10 seeds for powered comparison
- Curriculum on harder dataset where training order matters more
