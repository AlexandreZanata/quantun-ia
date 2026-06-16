# Results — EXP 002

**Date:** 2026-06-16  
**Config:** 300 samples, 30% holdout, seeds [42, 123, 456], 50 epochs

## What happened

| Architecture | Mean holdout acc | Std |
|--------------|------------------|-----|
| HybridSandwich | **83.0%** | ±2.9% |
| QuantumFirst | 82.6% | ±2.8% |
| ClassicalFirst | 81.9% | ±1.4% |

All three hybrid variants performed similarly on holdout. No architecture clearly dominated.

## Comparison with hypothesis

If the hypothesis was that combining classical + quantum beats pure models, it was **not strongly supported**. Hybrids did not exceed `classical_32` from EXP 001 (84.8%).

## Unexpected finding

`ClassicalFirst` had the lowest mean but also the **lowest variance** (±1.4%) — most stable across seeds.

## Suggested next experiment

- Test hybrids on a dataset where quantum feature maps are known to help (e.g. concentric circles)
- Ablate the classical pre-layer depth in `ClassicalFirst`
