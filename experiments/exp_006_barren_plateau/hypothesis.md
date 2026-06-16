# Hypothesis — EXP 006: Barren Plateau

**Date:** 2026-06-16  
**Author:** Quantum ML Lab

## What I expect to happen
Gradient variance at random initialization will decrease as qubit count increases
(2 → 10 qubits), consistent with barren plateau theory. The decrease should be
statistically separable via bootstrap 95% CIs on 50 random initializations per width.

## Why I expect this
Deeper/wider variational circuits exhibit exponentially vanishing gradients in
expectation. This is independent of whether the task is learnable on holdout.

## What would prove me wrong
- Gradient variance flat across qubit counts (CIs overlap for 2q vs 10q)
- Variance increases with qubits (opposite trend)
- NaN gradients dominate at any width (implementation bug)

## Metrics I will measure
- [x] Per-qubit gradient variance (50 samples each)
- [x] Bootstrap 95% CI per qubit count
- [ ] Correlation between init grad variance and final holdout (suggested)
