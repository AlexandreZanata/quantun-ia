# Results — EXP 005

**Date:** 2026-06-16  
**Config:** 300 samples, 30% holdout, 4 curriculum stages × 12 epochs/stage

## What happened

| Method | Holdout test accuracy |
|--------|----------------------|
| random (shuffled) | **85.0%** |
| margin_batches (staged) | 81.7% |

Staged curriculum (`margin_batches`) fixed the earlier global-margin bug (was ~50%) but still **underperformed** random shuffling on holdout.

Curriculum stage progression: 48.6% → 61.0% → 68.6% → **81.7%** (test acc per stage).

## Comparison with hypothesis

If the hypothesis was that easy-to-hard ordering improves QNN training, it was **not supported** on this run. Random order generalized better.

## Unexpected finding

Global `margin` ordering (pre-fix) collapsed to 50% — exposure bias from seeing hard examples too early in a fixed order. Batched staging recovered to 81.7% but did not beat random.

## Suggested next experiment

- Curriculum with mini-batch epochs (shuffle within each stage)
- Increase `epochs_per_stage` to 20 and compare again
