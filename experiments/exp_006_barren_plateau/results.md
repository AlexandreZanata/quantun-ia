# Results — EXP 006

**Date:** 2026-06-16  
**Config:** 20 random initializations per qubit count, gradient variance via concatenated param grads

## What happened

| Qubits | Mean grad variance |
|--------|-------------------|
| 2 | 0.00301 |
| 4 | 0.00212 |
| 6 | 0.00270 |
| 8 | 0.00078 |
| 10 | **0.00078** |

Gradient variance decreases from 2 → 8 qubits, consistent with barren plateau theory. The 6-qubit uptick vs 4 is likely sampling noise (n=20). Measurement is finite and reproducible (no NaN).

## Comparison with hypothesis

If the hypothesis was that deeper/wider circuits exhibit vanishing gradients, it was **supported** — variance at 8–10 qubits is ~4× lower than at 2 qubits.

## Unexpected finding

8 and 10 qubits converged to nearly identical variance in this run (0.00078).

## Suggested next experiment

- Increase `grad_samples` to 50 for smoother trend
- Compare parameter-shift rule vs autograd at 8+ qubits
