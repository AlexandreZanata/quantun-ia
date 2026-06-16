# Results — EXP 005

**Date:** 2026-06-16  
**Publication profile:** circles, n=500, noise=0.2, 10 seeds  
**Stats:** Wilcoxon margin_batches vs random

## What happened

| Method | Mean holdout | Std | 95% CI |
|--------|-------------|-----|--------|
| margin_batches | 48.9% | ±1.8% | [47.7%, 49.9%] |
| random | 48.5% | ±4.2% | [46.1%, 51.2%] |

**Paired Wilcoxon:** Δ=+0.4%, p=0.70 → **not significant**.

Both methods near chance on circles — curriculum cannot rescue a QNN that fails to learn the task.

## Comparison with hypothesis

Curriculum hypothesis **not supported** on hard dataset. Methodology is sound; the task exceeds current QNN capacity.

## Unexpected finding

On moons (prior), curriculum reached ~81%; on circles both methods ~49%. Dataset difficulty dominates training order.

## Suggested next experiment

- Only run curriculum after confirming base QNN > 60% holdout
- Progressive difficulty on circles with classical warm-start
