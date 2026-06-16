# Results — EXP 001

**Date:** 2026-06-16  
**Config:** 300 samples, 30% holdout, seeds [42, 123, 456], 50 epochs, bootstrap 95% CI  
**Stats:** Wilcoxon paired test vs `quantum_4q_2l`

## What happened

| Model | Mean holdout | Std | 95% CI |
|-------|-------------|-----|--------|
| classical_32 | **85.6%** | ±2.7% | [82.2%, 88.9%] |
| classical_8 | 83.0% | ±2.3% | [80.0%, 85.6%] |
| quantum_6q_3l | 81.9% | ±2.3% | [78.9%, 84.4%] |
| quantum_4q_2l | 80.4% | ±6.8% | [72.2%, 88.9%] |

**Paired test** classical_32 vs quantum_4q_2l: Δ=+5.2%, p=0.75 → **not significant** (n=3 seeds).

## Comparison with hypothesis

Quantum does not significantly outperform classical on 2D moons with current circuit depth. Differences are within seed noise.

## Unexpected finding

`quantum_4q_2l` CI overlaps classical_32 — with only 3 seeds, claims of superiority require more runs or larger test sets.

## Suggested next experiment

- Increase to 10 seeds for powered Wilcoxon comparison
- Try circles dataset where quantum feature maps may separate better
