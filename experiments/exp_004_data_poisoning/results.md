# Results — EXP 004

**Date:** 2026-06-16  
**Config:** 300 samples, 30% holdout, poison on train only, eval on clean test

## What happened

| Model | 0% poison | 10% poison | 30% poison |
|-------|-----------|------------|------------|
| Classical MLP | **85.6%** | 85.6% | **86.7%** |
| Quantum angle | 50.0% | 84.4% | 50.0% |
| Quantum amplitude | 50.0% | 50.0% | 50.0% |

Classical MLP was robust across all poison rates (82–87% holdout). Angle encoding is **highly unstable** — collapses to 50% at 0%, 20%, and 30% poison in this run. Amplitude encoding (2 qubits) stays at chance level regardless of poison.

Holdout metrics are now persisted in `experiments.jsonl` via `test_accuracy`.

## Comparison with hypothesis

If the hypothesis was that QNNs are more robust to poisoned labels, it was **rejected**. Classical degrades gracefully; quantum angle shows no consistent robustness pattern.

## Unexpected finding

Angle encoding sometimes reaches 84% at 10% poison while failing at 0% — initialization/seed sensitivity, not a poison effect.

## Suggested next experiment

- Multi-seed poison sweep for angle encoding
- Amplitude encoding with 4+ qubits (2 qubits insufficient for moons)
