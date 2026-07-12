---
title: ResidualNano Distill ACYD Brazil Maize
language: en
license: mit
tags:
  - tabular-classification
  - agro-climate
  - acyd
  - maize
  - distillation
  - open-data
datasets:
  - acyd_maize_brazil_v1
---

# ResidualNanoMLP Distill — ACYD Brazil Maize (Cycle v2 Phase E)

**Registry key:** `residual_nano_distill_acyd_maize`  
**Experiment:** `exp_092`  
**Architecture:** `residual_nano_distill` (ResidualNanoMLP, HistGB soft targets)

## Intended use

Agro-climate tabular benchmark on ACYD Brazil maize (corn) municipal panels. Binary label:
municipal yield below state-year median. Temporal split (train ≤ 2018, val 2019–2021).
Infrastructure validation — not operational planting advice.

## Metrics (publication, seed 42)

| Model | Val ROC-AUC |
|-------|-------------|
| HistGB teacher | ~0.8178 |
| ResidualNano distill student | ~0.8130 (−0.48 pp vs teacher) |
| LargeNanoMLP (exp_081 C4b) | ~0.8086 |

## Ship

```bash
make ship-residual-maize
# or
qml-ship --model residual_nano_distill_acyd_maize --profile ci --skip-train --skip-gate
qml-download --model residual_nano_distill_acyd_maize
```

## Validation gate

Student within **1.0 pp** of HistGB teacher on maize temporal val (exp_092 publication; met at −0.48 pp).
