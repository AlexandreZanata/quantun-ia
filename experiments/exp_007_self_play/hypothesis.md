# Hypothesis — EXP 007: Quantum Self-Play

**Date:** 2026-06-16  
**Author:** Quantum ML Lab

## What I expect to happen
With a learnable re-upload QNN base, self-play fine-tuning on hard misclassified
examples will improve holdout over the base model. Best-checkpoint tracking prevents
oscillation from overfitting hard subsets. Gate blocks self-play when base holdout < 55%.

## Why I expect this
Hard-example mining focuses gradient updates on decision boundaries. Capped hard_frac
and revert-to-best avoid the 50%↔88% oscillation seen with basic QNN on moons.

## What would prove me wrong
- Base model fails learnability gate → self-play N/A
- self_play_best does not beat base (Wilcoxon p ≥ 0.05)
- Self-play destabilizes training (holdout drops > 5 pp vs base)

## Metrics I will measure
- [x] Applicability gate (base holdout vs 55% threshold)
- [x] Holdout: base vs best checkpoint (10 seeds, bootstrap CI)
- [x] Paired Wilcoxon self_play_best vs self_play_base
