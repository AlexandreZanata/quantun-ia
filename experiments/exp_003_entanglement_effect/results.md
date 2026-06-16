# Results — EXP 003

**Date:** 2026-06-16  
**Config:** 300 samples, 30% holdout, seeds [42, 123, 456], learnable pre-projection, tuned LR for `chain_half`/`none`  
**Stats:** Wilcoxon chain vs none

## What happened

| Entanglement | Mean holdout | Std | 95% CI |
|--------------|-------------|-----|--------|
| ring | **83.3%** | ±3.3% | [78.9%, 86.7%] |
| chain_half | 83.0% | ±1.9% | [81.1%, 85.6%] |
| none | 83.0% | ±2.3% | [80.0%, 85.6%] |
| chain | 81.9% | ±2.8% | [78.9%, 85.6%] |

**Paired test** chain vs none: Δ=−1.1%, p=0.50 → **not significant**.

After adding `input_dim` pre-projection and LR tuning, `chain_half` variance dropped from ±12.7% to ±1.9%.

## Comparison with hypothesis

Entanglement topology does not produce statistically significant holdout differences on moons with 3 seeds.

## Unexpected finding

`none` (no CNOT) matches `ring` after pre-projection fix — angle embedding + linear head may dominate over entanglement on this task.

## Suggested next experiment

- 10-seed powered comparison on a task requiring entanglement (e.g. parity)
