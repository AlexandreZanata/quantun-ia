# Hypothesis — EXP 002: Hybrid Architecture

**Date:** 2026-06-16  
**Author:** Quantum ML Lab

## What I expect to happen
On circles, hybrid stacks (QuantumFirst, ClassicalFirst, HybridSandwich) use a
**re-upload QNN baseline** (4 qubits, 3 layers). QuantumFirst may edge ahead by
letting the quantum layer extract non-linear features before a small classical head.

## Why I expect this
Hybrids combine classical capacity with quantum feature maps. On a hard non-linear
task, partial quantum processing may help even when a standalone QNN fails.

## What would prove me wrong
- Any hybrid beats classical_32 from exp_001 (≥ 65% holdout mean)
- All hybrids stay at chance (~50%) like basic QNN
- HybridSandwich significantly outperforms QuantumFirst (opposite of expectation)

## Metrics I will measure
- [x] Holdout accuracy per architecture (10 seeds, bootstrap CI)
- [x] Paired Wilcoxon across architectures (Holm-Bonferroni corrected)
- [x] Training time per seed
