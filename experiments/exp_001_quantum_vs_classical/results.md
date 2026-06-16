# Results — EXP 001

**Date:** 2026-06-16  
**Publication profile:** circles, n=500, noise=0.2, 10 seeds, bootstrap 95% CI  
**Stats:** Wilcoxon classical_32 vs quantum_4q_2l

## What happened

| Model | Mean holdout | Std | 95% CI |
|-------|-------------|-----|--------|
| classical_32 | **65.5%** | ±4.1% | [62.9%, 68.1%] |
| classical_8 | 58.3% | ±3.8% | [56.1%, 60.9%] |
| quantum_6q_3l | 55.4% | ±5.6% | [52.1%, 58.9%] |
| quantum_4q_2l | 52.9% | ±5.0% | [49.7%, 55.7%] |

**Paired Wilcoxon** classical_32 vs quantum_4q_2l: Δ=+12.7%, **p=0.002 → significant**.

On the harder circles dataset, classical MLP **significantly outperforms** QNN at n=10 seeds.

## Comparison with hypothesis

Quantum advantage hypothesis **rejected** on circles/noise=0.2. QNN holdout clusters near 53% (barely above chance on concentric data).

## Unexpected finding

Moons (prior runs ~85%) masked QNN weakness — circles exposes that current circuits do not learn non-linear boundaries.

## Suggested next experiment

- Data re-uploading (multiple encoding layers)
- Stronger classical baseline matched by parameter count
