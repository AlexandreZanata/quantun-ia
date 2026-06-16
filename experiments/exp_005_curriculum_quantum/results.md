# Results — EXP 005

**Date:** 2026-06-16  
**Config:** 300 samples, 30% holdout, 4 stages × 12 epochs + 12 refine epochs  
**Fixes applied:** easy-first margin sort, shared optimizer across stages, final full-data refine

## What happened

| Method | Holdout test accuracy |
|--------|----------------------|
| margin_batches (staged) | **81.1%** |
| random (shuffled) | 50.0% |

**margin_batches stage holdout:** 50.0% → 50.0% → 63.3% → 76.7% → **81.1%** (after refine)

The margin sort bug (hard-first) is fixed. Curriculum now improves monotonically across stages. In this run, `random` failed to learn (50% holdout) while `margin_batches` reached 81% — high seed/model variance for QNN on a single split.

Earlier run (same day, pre-full-batch): margin_batches 85.6%, random 68.9%.

## Comparison with hypothesis

Curriculum **can** help when random training fails, but results are **not stable** across runs. Easy-first staging + refine is methodologically sound; more seeds needed for a firm conclusion.

## Unexpected finding

Root cause of prior 53% margin_batches was inverted sort order (hard examples first) plus optimizer reset each stage.

## Suggested next experiment

- 3-seed comparison: random vs margin_batches
- Shuffle mini-batches within each curriculum stage
