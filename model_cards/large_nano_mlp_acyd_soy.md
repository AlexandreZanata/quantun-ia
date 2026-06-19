---
title: LargeNanoMLP ACYD Brazil Soybean
language: en
license: mit
tags:
  - tabular-classification
  - agro-climate
  - acyd
  - open-data
datasets:
  - acyd_soy_brazil_v1
---

# LargeNanoMLP — ACYD Brazil Soybean (C4 anchor)

**Registry key:** `large_nano_mlp_acyd_soy`  
**Experiment:** `exp_060`  
**Architecture:** `large_nano_mlp` (~1.16M parameters, 37 → 2048 → 512 → 64 → 1)

## Intended use

Agro-climate tabular benchmark on ACYD Brazil soybean municipal panels. Binary label:
municipal yield below state-year median. Temporal split (train ≤ 2018, val 2019–2021).
Infrastructure validation — not operational planting advice.

## Ship

```bash
qml-ship --model large_nano_mlp_acyd_soy --skip-train
qml-download --model large_nano_mlp_acyd_soy
```

## Validation gate

Val ROC-AUC advantage vs logistic ≥ 2.0 pp (exp_060 publication gate).
