# Results — EXP 006

**Date:** 2026-06-16  
**Config:** 20 random initializations per qubit count, gradient variance via concatenated param grads

## What happened

| Qubits | Mean grad variance |
|--------|-------------------|
| 2 | 0.00426 |
| 4 | 0.00296 |
| 6 | 0.00179 |
| 8 | 0.00226 |
| 10 | **0.00099** |

Gradient variance **decreases** as qubit count grows (2 → 10), consistent with barren plateau theory. Fix from NaN (scalar `.var()` bug) to concatenated gradients was essential for valid measurement.

## Comparison with hypothesis

If the hypothesis was that deeper/wider circuits exhibit vanishing gradients, it was **supported** — variance at 10 qubits is ~4× lower than at 2 qubits.

## Unexpected finding

8 qubits showed a slight uptick vs 6 (0.00226 vs 0.00179) — likely sampling noise with only 20 inits.

## Suggested next experiment

- Increase `grad_samples` to 50 for smoother trend
- Compare parameter-shift rule vs autograd at 8+ qubits
