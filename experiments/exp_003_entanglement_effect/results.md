# Results — EXP 003

**Date:** 2026-06-16  
**Config:** 300 samples, 30% holdout, seeds [42, 123, 456], 4 qubits, 2 layers

## What happened

| Entanglement | Mean holdout acc | Std |
|--------------|------------------|-----|
| ring | **80.0%** | ±2.7% |
| chain | 78.9% | ±1.8% |
| none | 76.3% | ±5.2% |
| chain_half | 66.3% | ±10.9% |

Ring entanglement generalized best on average. `chain_half` (ablation: half the CNOTs) was worst and most variable.

## Comparison with hypothesis

More entanglement does **not** monotonically help. Full chain was not the best — ring topology won. The `chain_half` ablation suggests partial entanglement can hurt more than help on this task.

## Unexpected finding

`chain_half` had ±10.9% std across seeds — highest variance of all variants. Partial CNOT patterns may create unstable gradient landscapes.

## Suggested next experiment

- Fix seed and plot learning curves per entanglement type
- Test `chain_half` with learning rate sweep (possible barren plateau on some seeds)
