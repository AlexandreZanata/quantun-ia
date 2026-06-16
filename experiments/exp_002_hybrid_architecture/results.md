# Results — EXP 002

**Date:** 2026-06-16  
**Publication profile:** circles, n=500, noise=0.2, 10 seeds, bootstrap 95% CI

## What happened

| Architecture | Mean holdout | Std | 95% CI |
|--------------|-------------|-----|--------|
| quantum_first | **58.9%** | ±3.8% | [56.5%, 61.4%] |
| classical_first | 57.6% | ±2.4% | [56.2%, 59.1%] |
| hybrid_sandwich | 56.3% | ±5.3% | [53.0%, 59.5%] |

All hybrids ~56–59% — none approach classical_32 from EXP 001 (65.5%).

## Comparison with hypothesis

Hybrid quantum-classical stacks do not overcome circles difficulty. No architecture is statistically compared here but all underperform dedicated classical MLP.

## Unexpected finding

`quantum_first` leads slightly — opposite of moons runs where hybrids matched ~83%.

## Suggested next experiment

- Paired Wilcoxon across three architectures with 10 seeds
- Deeper classical pre-network in HybridSandwich
