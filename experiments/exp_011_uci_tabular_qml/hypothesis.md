# Hypothesis — EXP 011: UCI Tabular QML vs Classical

**Date:** 2026-06-17  
**Author:** Quantum ML Lab

## What I expect to happen
On breast cancer (30 features, binary), a parameter-matched classical MLP will match or beat
a 4-qubit angle-encoding QNN. The perceptron baseline should underperform both.

## Why I expect this
Tabular UCI data is well-suited to classical models; QNNs with few qubits may lack capacity
for 30-dimensional inputs despite the linear pre-projection layer.

## What would prove me wrong
- QNN holdout mean exceeds classical_matched by > 3 pp with Holm-significant Wilcoxon
- Perceptron beats quantum_angle (unexpected linear separability)

## Metrics I will measure
- [x] Holdout accuracy (10 seeds, bootstrap 95% CI)
- [x] Paired Wilcoxon vs classical_matched
- [x] Parameter counts (matched baseline)
- [x] HPO-tuned learning rate and qubit/layer config (Optuna)
