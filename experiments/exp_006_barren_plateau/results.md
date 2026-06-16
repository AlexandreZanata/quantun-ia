# Results — EXP 006

**Date:** 2026-06-16  
**Publication profile:** 50 gradient samples per qubit count, circles-compatible circuit

## What happened

| Qubits | Grad variance |
|--------|---------------|
| 2 | 0.00654 |
| 4 | 0.00237 |
| 6 | 0.00201 |
| 8 | 0.00115 |
| 10 | **0.00084** |

~7.8× decrease 2→10 qubits. Trend robust with n=50 inits.

## Comparison with hypothesis

Barren plateau **supported** — gradient variance vanishes with circuit width on this architecture.

## Unexpected finding

Gradient measurement is independent of dataset difficulty — barren plateau is real even when holdout accuracy is at chance.

## Suggested next experiment

- Correlate grad variance at init with final holdout per seed
- Parameter-shift rule comparison at 8+ qubits
