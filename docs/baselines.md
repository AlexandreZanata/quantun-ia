# Literature Baselines — QML Comparison Reference

This document lists published benchmarks used to contextualize quantun-ia results.
Cite these works in `results.md` when comparing holdout metrics.

## Standard Variational Quantum Classifier (VQC)

| Reference | Task | Model | Reported metric | Our implementation |
|-----------|------|-------|-----------------|-------------------|
| Schuld & Petruccione (2018) | Binary classification | 4-qubit angle-encoding VQC, 2 layers | Varies by dataset | `QuantumNetBasic(n_qubits=4, n_layers=2)` — exp_001, exp_011 |
| Farhi & Neven (2018) | QAOA-style classifier | Layered VQC | N/A (theoretical) | Hybrid architectures in exp_002 |

## UCI Tabular Benchmarks

| Dataset | Classical baseline (literature) | Typical accuracy | Our experiment |
|---------|--------------------------------|------------------|----------------|
| Breast Cancer Wisconsin | SVM / MLP | ~95–97% (full dataset, CV) | exp_011 (30% holdout, 10 seeds) |
| Wine (binary subset) | Logistic regression | ~95%+ | Available via `wine_binary` loader |
| Iris (binary subset) | Perceptron | ~90%+ | Available via `iris_binary` loader |

> **Note:** Published UCI scores often use cross-validation on the full dataset.
> Our holdout protocol (30% test, multi-seed) yields lower absolute numbers but
> enables paired statistical comparison.

## MNIST Subset (QML literature)

| Reference | Encoding | Qubits | MNIST task | Reported | Our experiment |
|-----------|----------|--------|------------|----------|----------------|
| Schuld et al. (2020) | Amplitude / angle | 4–10 | Subset / downscaled | ~80–95% (varies) | exp_012 (0 vs 1, PCA-8, 4 qubits) |
| Cincio et al. (2022) | Angle + entanglement | 4 | Binary subsets | ~70–90% | Compare with exp_009, exp_012 |

## Published VQC Config (reproduced in this repo)

The following config matches the common 4-qubit angle-encoding baseline cited in QML reviews:

```yaml
# Published baseline — Schuld (2018) style VQC
n_qubits: 4
n_layers: 2
encoding: angle
entanglement: linear CNOT chain
optimizer: Adam
learning_rate: 0.02
```

Implemented as `QuantumNetBasic(n_qubits=4, n_layers=2)` in exp_001 and exp_011.

## How to compare results

1. Run experiment with `publication` or `publication_large` profile.
2. Read multi-seed summary from `logs/experiments.jsonl` or `make dashboard-local`.
3. Compare bootstrap CI against the table above — focus on **relative ordering**, not absolute match.
4. Document in `experiments/exp_NNN/results.md` with citation and protocol differences.

## References

- Schuld, M., & Petruccione, F. (2018). *Supervised Learning with Quantum Computers*. Springer.
- Schuld, M., Bocharov, A., Svore, K., & Wiebe, N. (2020). Circuit-centric quantum classifiers. *Physical Review A*, 101(3), 032308.
- Farhi, E., & Neven, H. (2018). Classification with quantum neural networks on near term processors. arXiv:1802.06002.
- Cincio, C., et al. (2022). Cost function dependent barren plateaus in shallow quantum neural networks. *Nature Communications*, 13, 1798.
