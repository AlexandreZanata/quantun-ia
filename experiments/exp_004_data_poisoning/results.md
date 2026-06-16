# Results — EXP 004

**Date:** 2026-06-16  
**Config:** 300 samples, 30% holdout, 3 seeds, poison on train only, clean holdout eval  
**Models:** angle 4q/1L, amplitude 4q/2L with learnable pre-projection (LR 0.02)

## What happened

| Model | 0% poison (mean ± std) | 30% poison (mean ± std) | 95% CI @30% |
|-------|------------------------|-------------------------|-------------|
| Classical MLP | 82.6% ± 2.8% | 84.1% ± 1.4% | [82.2%, 85.6%] |
| Quantum amplitude | **84.4% ± 2.4%** | 81.5% ± 2.3% | [78.9%, 84.4%] |
| Quantum angle | 77.4% ± 1.0% | 74.1% ± 5.9% | [66.7%, 81.1%] |

Amplitude encoding now learns (was ~50% with 2-qubit zero-padding). At 0% poison, amplitude **beats** angle and classical on mean holdout.

## Comparison with hypothesis

Classical remains robust under poison. Amplitude degrades modestly (−2.9%) at 30% poison. Angle is least stable (wide CI at 30%).

## Unexpected finding

Learnable `nn.Linear → amp_dim` projection was the key fix — not more qubits alone.

## Suggested next experiment

- Paired Wilcoxon: amplitude vs angle at each poison rate across seeds
