# Results — EXP 002

**Date:** 2026-06-16  
**Config:** 300 samples, 30% holdout, seeds [42, 123, 456], 50 epochs

## What happened

| Architecture | Mean holdout acc | Std |
|--------------|------------------|-----|
| hybrid_sandwich | **82.6%** | ±2.8% |
| classical_first | 82.2% | ±2.4% |
| quantum_first | 81.5% | ±3.7% |

All three hybrid variants performed similarly on holdout (~82%). No architecture clearly dominated.

## Comparison with hypothesis

If the hypothesis was that combining classical + quantum beats pure models, it was **not strongly supported**. Hybrids did not exceed `classical_32` from EXP 001 (85.2%).

## Unexpected finding

`quantum_first` had the highest variance (±3.7%) despite competitive mean — quantum front-end adds seed sensitivity.

## Suggested next experiment

- Test hybrids on concentric circles (quantum-friendly geometry)
- Ablate classical hidden size in `HybridSandwich`
