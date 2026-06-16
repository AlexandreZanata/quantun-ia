# Results — EXP 003

**Date:** 2026-06-16  
**Publication profile:** circles, n=500, noise=0.2, 10 seeds  
**Stats:** Wilcoxon chain vs none

## What happened

| Entanglement | Mean holdout | Std | 95% CI |
|--------------|-------------|-----|--------|
| chain_half | **64.0%** | ±4.5% | [61.2%, 66.9%] |
| none | 60.7% | ±5.4% | [57.5%, 64.3%] |
| ring | 58.7% | ±5.5% | [55.7%, 62.6%] |
| chain | 57.2% | ±4.5% | [54.3%, 59.8%] |

**Paired Wilcoxon** chain vs none: Δ=−3.5%, **p=0.031 → significant** (chain worse than none).

## Comparison with hypothesis

Full chain entanglement does **not** help on circles — significantly worse than no entanglement. `chain_half` leads on mean but was not individually tested vs others.

## Unexpected finding

On moons, entanglement helped; on circles, `none` beats `chain`. Topology effects are dataset-dependent.

## Suggested next experiment

- Wilcoxon chain_half vs none (likely the meaningful comparison)
- Task-specific entanglement design (not blind ablation)
