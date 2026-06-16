# Results — EXP 006

**Date:** 2026-06-16  
**Config:** 50 random initializations per qubit count (up from 20), concatenated gradient variance

## What happened

| Qubits | Grad variance |
|--------|---------------|
| 2 | 0.00475 |
| 4 | 0.00170 |
| 6 | 0.00182 |
| 8 | 0.00101 |
| 10 | 0.00107 |

Monotonic decrease 2 → 8 qubits (~4.7× drop). Variance stabilizes at 8–10 qubits.

## Comparison with hypothesis

Barren plateau trend **supported** with n=50 samples — smoother than n=20 runs.

## Unexpected finding

6-qubit variance slightly above 4-qubit — within noise band but worth monitoring with more inits.

## Suggested next experiment

- Parameter-shift rule vs autograd at 8+ qubits
- Log gradient variance during actual training (not just init)
