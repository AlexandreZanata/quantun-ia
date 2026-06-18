---
title: QuantumNano-BC
language: en
license: mit
tags:
  - quantum-machine-learning
  - tabular-classification
  - breast-cancer
datasets:
  - breast_cancer
---

# QuantumNano-BC — Hybrid QML Nano Model

**Generated:** 2026-06-18  
**Experiment:** `exp_024` (QuantumNano-BC flagship)  
**Architecture:** `hybrid_sandwich` (4 qubits, 2 re-upload layers)  

## Intended use

Research benchmark for holdout-fair comparison of hybrid quantum–classical classifiers
on the Wisconsin Breast Cancer (UCI) dataset. **Not for clinical deployment.**

## Training data

- **Dataset:** Wisconsin Breast Cancer (`sklearn.datasets.load_breast_cancer`)
- **Samples:** 569 (full dataset, no subsampling in publication profile)
- **Features:** 30 diagnostic measurements
- **Split:** 30% stratified holdout before `StandardScaler` (train-fit only)

## Evaluation results

| Model | Mean holdout accuracy | 95% CI |
|-------|----------------------|--------|
| **hybrid_sandwich** | 97.4% | [97.0%, 97.8%] |
| logistic_regression | 97.9% | — |
| xgboost_shallow | 96.2% | — |

## How to reproduce

```bash
qml-train --model hybrid_sandwich --dataset breast_cancer --profile publication
# or full benchmark:
python experiments/exp_024_quantum_nano_bc/run.py --profile publication
```

## Artifacts

- Checkpoint: run with `--profile publication` and `save_checkpoints=true`

## Limitations

- Simulator-only quantum execution (PennyLane `default.qubit`)
- Nano parameter budget (~150–300 trainable parameters)
- Single holdout protocol; no nested cross-validation
- Results vary by seed list in `config/experiments.yaml`

## Citation

See [CITATION.cff](../CITATION.cff) and `experiments/exp_024_quantum_nano_bc/hypothesis.md`.
