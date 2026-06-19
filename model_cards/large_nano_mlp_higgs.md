---
title: LargeNanoMLP HIGGS
language: en
license: mit
tags:
  - tabular-classification
  - higgs
  - open-data
datasets:
  - higgs_v1
---

# LargeNanoMLP — HIGGS

**Registry key:** `large_nano_mlp_higgs`  
**Experiment:** `exp_032`  
**Architecture:** `large_nano_mlp` (~1.14M parameters)

## Intended use

Million-row tabular benchmark on UCI HIGGS open dataset. Infrastructure validation, not physics discovery.

## Ship

```bash
qml-ship --model large_nano_mlp_higgs --skip-train
qml-download --model large_nano_mlp_higgs
```

## Validation gate

Val ROC-AUC advantage vs logistic ≥ 1.0 pp (exp_032 publication gate).
