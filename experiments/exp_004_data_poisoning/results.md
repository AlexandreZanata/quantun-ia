# Results — EXP 004

**Date:** 2026-06-16  
**Publication profile:** circles, n=500, noise=0.2, 10 seeds, poison on train / clean holdout

## What happened

| Model | 0% poison | 30% poison | Drop @30% | 95% CI @30% |
|-------|-----------|------------|-----------|-------------|
| Classical MLP | 62.6% | 58.7% | −3.9% | [56.0%, 61.1%] |
| Quantum amplitude | 62.1% | 54.9% | −7.2% | [50.3%, 59.0%] |
| Quantum angle | 52.1% | 53.1% | +1.0% | [49.4%, 56.7%] |

Classical most robust. Amplitude learns (62% clean) but degrades more under poison. Angle stuck ~52% (near chance on circles).

## Comparison with hypothesis

Poison robustness: classical wins. Amplitude viable but less robust than classical at 30% poison. Angle encoding insufficient for circles regardless of poison.

## Unexpected finding

Angle "improves" at 30% poison (+1%) because it never learned — noise is not the signal.

## Suggested next experiment

- Multi-seed Wilcoxon classical vs amplitude at 30% poison
- Amplitude with data re-uploading on circles
