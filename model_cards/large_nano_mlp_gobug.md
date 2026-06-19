---
title: LargeNanoMLP GoBug Code Defects
language: en
license: mit
tags:
  - tabular-classification
  - software-defects
  - gobug
  - open-data
datasets:
  - code_defects_gobug_v1
---

# LargeNanoMLP — GoBug Code Defects (C3 anchor)

**Registry key:** `large_nano_mlp_gobug`  
**Experiment:** `exp_070`  
**Architecture:** `large_nano_mlp` (~1.14M parameters, 23 → 2048 → 512 → 64 → 1)

## Intended use

File-level software defect prediction on GoBug static code metrics. Primary metric: **PR-AUC**
on temporal val split. Research benchmark — not production static analysis.

## Ship

```bash
qml-ship --model large_nano_mlp_gobug --skip-train
qml-download --model large_nano_mlp_gobug
```

## Validation gate

Val PR-AUC advantage vs logistic ≥ 2.0 pp (exp_070 publication gate).
