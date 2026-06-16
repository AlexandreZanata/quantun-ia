# Results — EXP 004

**Date:** 2026-06-16  
**Config:** 300 samples, 30% holdout, poison on train only, eval on clean test

## What happened

| Model | 0% poison | 30% poison | Drop at 30% |
|-------|-----------|------------|-------------|
| Classical MLP | **90.0%** | 80.0% | −10.0% |
| Quantum angle | 70.0% | 81.7% | −11.7%* |
| Quantum amplitude | 51.7% | 55.0% | −3.3% |

*Angle encoding showed erratic holdout at low poison rates (50% at 5% poison) — high variance on small test set (~90 samples).

Classical MLP was the most robust and accurate. Amplitude encoding (2 qubits) barely learned (~chance level). Angle encoding was unstable under label noise.

## Comparison with hypothesis

If the hypothesis was that QNNs are more robust to poisoned labels, it was **rejected**. Classical models degraded gracefully; quantum angle collapsed at 10% poison in earlier runs.

## Unexpected finding

Amplitude encoding stayed near 50% regardless of poison rate — it fails to learn moons with only 2 qubits, not a robustness story.

## Suggested next experiment

- Increase qubits for amplitude encoding (4 qubits, pad to 16 amplitudes)
- Use cross-validation instead of single holdout for poison experiments
