---
title: LargeNanoMLP ACYD Brazil Maize
language: en
license: mit
tags:
  - tabular-classification
  - agro-climate
  - acyd
  - maize
  - open-data
datasets:
  - acyd_maize_brazil_v1
---

# LargeNanoMLP — ACYD Brazil Maize (C4b anchor)

**Registry key:** `large_nano_mlp_acyd_maize`  
**Experiment:** `exp_081`  
**Architecture:** `large_nano_mlp` (~1.16M parameters, 37 → 2048 → 512 → 64 → 1)

## Intended use

Agro-climate tabular benchmark on ACYD Brazil maize (corn) municipal panels. Binary label:
municipal yield below state-year median. Temporal split (train ≤ 2018, val 2019–2021).
Infrastructure validation — not operational planting advice.

## Ship

```bash
qml-ship --model large_nano_mlp_acyd_maize --skip-train
qml-download --model large_nano_mlp_acyd_maize
```

## Validation gate

Val ROC-AUC advantage vs logistic ≥ 2.0 pp (exp_081 publication gate).
