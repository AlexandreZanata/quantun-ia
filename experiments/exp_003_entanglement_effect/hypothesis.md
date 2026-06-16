# Hypothesis — EXP 003: Entanglement Effect

**Date:** 2026-06-16  
**Author:** Quantum ML Lab

## What I expect to happen
On circles, entanglement (chain, ring) will improve QNN holdout over `none` because
correlated qubits increase expressivity of the variational circuit. `chain_half` may
sit between full chain and none as a capacity regularizer.

## Why I expect this
Entanglement is often cited as a quantum advantage for feature learning. More
connectivity should help model radial decision boundaries.

## What would prove me wrong
- `none` beats `chain` significantly (Wilcoxon p < 0.05) — entanglement hurts
- All topologies within noise of each other (no topology effect)
- chain_half is best by > 5 pp over all others

## Metrics I will measure
- [x] Holdout accuracy per entanglement type (10 seeds, bootstrap CI)
- [x] Paired Wilcoxon: chain vs none
- [ ] Wilcoxon chain_half vs none (suggested follow-up)
