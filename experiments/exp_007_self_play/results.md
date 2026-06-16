# Results — EXP 007

**Date:** 2026-06-16  
**Config:** 300 samples, 30% holdout, 5 self-play rounds, fine-tune on hard train examples only

## What happened

| Phase | Holdout test accuracy |
|-------|----------------------|
| Round 0 (base) | 50.0% |
| Round 1 | 87.8% |
| Round 2 | 50.0% |
| Round 3 | 87.8% |
| Round 4 (final) | **50.0%** |

Self-play **oscillates** between ~50% and ~88% holdout depending on the hard subset selected each round. No monotonic improvement; final holdout equals the untrained baseline.

Train pool is separate from holdout (no leakage). The oscillation pattern suggests overfitting to misclassified train points each round.

## Comparison with hypothesis

If the hypothesis was that re-training on hard examples improves generalization, it was **not supported** in this run — gains are ephemeral and reverse on alternating rounds.

## Unexpected finding

Rounds with large hard sets (n_hard=105) reach 87.8% holdout; rounds with small hard sets (n_hard≈24) collapse to 50%.

## Suggested next experiment

- Cap hard examples per round (top 20% by loss)
- Early stopping on holdout each round instead of fixed fine-tune epochs
