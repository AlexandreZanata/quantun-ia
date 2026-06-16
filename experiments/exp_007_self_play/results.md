# Results — EXP 007

**Date:** 2026-06-16  
**Config:** 3 seeds, hard_frac=0.3 cap, best-checkpoint tracking, revert on holdout drop >5%  
**Stats:** Wilcoxon self_play_best vs self_play_base

## What happened

| Phase | Mean holdout | Std | 95% CI |
|-------|-------------|-----|--------|
| Base (before self-play) | 83.3% | ±3.3% | [78.9%, 86.7%] |
| Best checkpoint (after self-play) | 83.3% | ±3.3% | [78.9%, 86.7%] |

**Paired test** best vs base: Δ=0.0%, p=1.0 → **not significant**.

Oscillation eliminated — `hard_frac` cap + best-checkpoint prevents collapse to 50%. Self-play neither improves nor degrades holdout on moons.

## Comparison with hypothesis

Re-training on capped hard examples does not improve generalization beyond a well-trained base model.

## Unexpected finding

Previous 50%↔88% oscillation was an artifact of uncapped hard sets and no checkpoint revert — methodology fix was more impactful than the algorithm.

## Suggested next experiment

- Self-play only when base holdout < 70% (weak base regime)
- Holdout-guided early stopping per round
