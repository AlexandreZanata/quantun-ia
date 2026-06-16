# Results — EXP 002

**Date:** 2026-06-16  
**Config:** 300 samples, 30% holdout, seeds [42, 123, 456], bootstrap 95% CI

## What happened

| Architecture | Mean holdout | Std | 95% CI |
|--------------|-------------|-----|--------|
| hybrid_sandwich | **83.0%** | ±3.7% | [78.9%, 87.8%] |
| quantum_first | 83.0% | ±2.3% | [80.0%, 85.6%] |
| classical_first | 81.9% | ±2.9% | [77.8%, 84.4%] |

All hybrids within ~1% mean holdout. CIs heavily overlap.

## Comparison with hypothesis

Hybrid architectures do not beat `classical_32` from EXP 001 (85.6%). No architecture dominates.

## Unexpected finding

`hybrid_sandwich` has highest variance (±3.7%) despite tied mean with `quantum_first`.

## Suggested next experiment

- Paired Wilcoxon across all three architectures
- Parameter-count-matched classical baseline
