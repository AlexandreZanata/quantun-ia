# Results — EXP 003

**Date:** 2026-06-16  
**Config:** 300 samples, 30% holdout, seeds [42, 123, 456], 4 qubits, 2 layers

## What happened

| Entanglement | Mean holdout acc | Std |
|--------------|------------------|-----|
| chain | **81.5%** | ±5.2% |
| ring | 79.6% | ±3.7% |
| chain_half | 62.6% | ±12.7% |
| none | 60.7% | ±3.7% |

Full `chain` entanglement generalized best on average. `none` and `chain_half` underperformed — partial or missing entanglement hurts on this task.

## Comparison with hypothesis

More entanglement helps when it is **consistent** (chain, ring). The `chain_half` ablation confirms that partial CNOT patterns create unstable landscapes (±12.7% std).

## Unexpected finding

`chain_half` variance remains very high across seeds — worse than `none` on mean holdout in this run.

## Suggested next experiment

- Learning rate sweep for `chain_half` only
- Plot per-seed learning curves to separate barren plateau from bad initialization
