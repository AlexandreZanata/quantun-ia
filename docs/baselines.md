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
| HIGGS (UCI open) | Logistic / GBDT / MLP | AUC varies by protocol | exp_032, exp_058 (val ROC-AUC, train-only scaler) |

### Conventional sklearn/XGBoost stack (exp_058)

| Model | Library | Role in exp_058 |
|-------|---------|-----------------|
| LogisticRegression | sklearn | Linear tabular baseline |
| MLPClassifier (2048→512→64) | sklearn | Matched-topology deep baseline |
| HistGradientBoostingClassifier | sklearn | Strong default GBDT on CPU |
| XGBClassifier (depth 3, 50 trees) | xgboost | Shallow boosted trees |
| LargeNanoMLP | PyTorch (quantun-ia) | Shipped exp_032 checkpoint |

Run: `python experiments/exp_058_conventional_higgs_baselines/run.py --profile publication --write-results`

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

## Adaptive LR (this repo — Phase 4)

| Reference | Method | Our experiment |
|-----------|--------|----------------|
| McClean et al. (2018) | Barren plateau theory | exp_006 diagnostics → exp_015 `var_target` |
| Grant et al. (2019) | Initialization strategies | Compared via fixed vs adaptive LR |

exp_015 implements variance-scaled Adam (`src/training/adaptive_lr.py`) with Cohen's d reporting.
Full method specification: [method_adaptive_lr.md](method_adaptive_lr.md).

## How to compare results

1. Run experiment with `publication` or `publication_large` profile.
2. Read multi-seed summary from `logs/experiments.jsonl` or `make dashboard-local`.
3. Compare bootstrap CI against the table above — focus on **relative ordering**, not absolute match.
4. Document in `experiments/exp_NNN/results.md` with citation and protocol differences.
5. Report Cohen's d with magnitude labels and check MDE via `make power-analysis`.

## Our protocol vs literature

| Aspect | Typical literature | quantun-ia (this repo) |
|--------|-------------------|------------------------|
| Train/test split | Often k-fold CV on full dataset | Single 70/30 stratified holdout **before** scaling |
| Seeds | Frequently single run or unreported | 10 seeds (`publication` profile) |
| Significance | Varies; often none | Paired Wilcoxon + Holm-Bonferroni |
| Effect size | Rarely reported | Cohen's d (paired) in JSONL and `results.md` |
| Classical baseline | SVM / tuned MLP | Parameter-matched perceptron / MLP (`param_match.py`) |
| Qubit budget | Inconsistent | Documented per experiment; ≤8 qubits default |
| Preprocessing | Sometimes on full data | Fit scaler/PCA on train only (`scaling.py`) |

> Use this table in `results.md` Limitations when citing external accuracy numbers.

## References

- Schuld, M., & Petruccione, F. (2018). *Supervised Learning with Quantum Computers*. Springer.
- Schuld, M., Bocharov, A., Svore, K., & Wiebe, N. (2020). Circuit-centric quantum classifiers. *Physical Review A*, 101(3), 032308.
- Farhi, E., & Neven, H. (2018). Classification with quantum neural networks on near term processors. arXiv:1802.06002.
- Cincio, C., et al. (2022). Cost function dependent barren plateaus in shallow quantum neural networks. *Nature Communications*, 13, 1798.
