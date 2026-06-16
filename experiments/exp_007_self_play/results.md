# Results — EXP 007

**Date:** 2026-06-16  
**Config:** 300 samples, 30% holdout, 5 self-play rounds, fine-tune on hard train examples only

## What happened

| Phase | Holdout test accuracy |
|-------|----------------------|
| Base model (before self-play) | 46.7% |
| After round 1 | 76.7% |
| After round 4 (final) | **74.4%** |

Self-play improved holdout from 46.7% → ~75%, but accuracy **oscillated** (50% at round 2). Previous 96% result was inflated by train/eval leakage — now fixed.

## Comparison with hypothesis

If the hypothesis was that re-training on hard examples improves generalization, it was **partially supported** — holdout improved substantially, but not monotonically.

## Unexpected finding

Fine-tuning only on misclassified train examples sometimes **hurt** holdout (round 2: 50%) — overfitting to the hard subset without stable generalization.

## Suggested next experiment

- Cap hard examples per round (e.g. top 20% by loss)
- Evaluate on a second held-out validation set each round to detect overfitting early
