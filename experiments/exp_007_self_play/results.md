# Results — EXP 007

**Date:** 2026-06-16  
**Publication profile:** circles, n=500, noise=0.2, 10 seeds, hard_frac=0.3, best-checkpoint  
**Stats:** Wilcoxon self_play_best vs self_play_base

## What happened

| Phase | Mean holdout | Std | 95% CI |
|-------|-------------|-----|--------|
| Base | 53.3% | ±4.1% | [50.9%, 56.0%] |
| Best checkpoint | 53.5% | ±4.0% | [51.3%, 56.1%] |

**Paired Wilcoxon:** Δ=+0.2%, p=0.50 → **not significant**.

Self-play stable (no oscillation) but provides no measurable gain on circles.

## Comparison with hypothesis

Self-play hypothesis **not supported** when base model is weak (~53%). Hard-example fine-tuning cannot improve what barely learns.

## Unexpected finding

Checkpoint revert fix works (no 50%↔88% swings) but reveals the algorithm has no effect when base accuracy ≈ chance.

## Suggested next experiment

- Self-play only when base holdout > 65%
- Ensemble of seed checkpoints instead of iterative hard mining
