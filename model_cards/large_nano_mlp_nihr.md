---
title: LargeNanoMLP NIHR Synthetic CV
language: en
license: mit
tags:
  - tabular-classification
  - clinical
  - nihr
  - open-data
datasets:
  - nihr_cv_synthetic_v1
---

# LargeNanoMLP — NIHR Synthetic CV (C2 anchor)

**Registry key:** `large_nano_mlp_nihr`  
**Experiment:** `exp_069`  
**Architecture:** `large_nano_mlp` (~1.11M parameters, 13 → 2048 → 512 → 64 → 1)

## Intended use

Realistic-prevalence synthetic cardiovascular event benchmark (NIHR Zenodo CC0).
Primary metric: **PR-AUC** on imbalanced val split. Research infrastructure — not clinical deployment.

## Ship

```bash
qml-ship --model large_nano_mlp_nihr --skip-train
qml-download --model large_nano_mlp_nihr
```

## Validation gate

Val PR-AUC advantage vs logistic ≥ 1.0 pp (exp_069 publication gate).
