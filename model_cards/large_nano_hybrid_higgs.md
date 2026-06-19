---
title: LargeNanoHybrid HIGGS
language: en
license: mit
tags:
  - quantum-machine-learning
  - tabular-classification
  - higgs
datasets:
  - higgs_v1
---

# LargeNanoHybrid — HIGGS

**Registry key:** `large_nano_hybrid_higgs`  
**Experiment:** `exp_037`  
**Architecture:** frozen `LargeNanoMLP` + 4-qubit re-upload QNN head

## Intended use

Holdout-fair comparison of quantum decision head vs classical head on frozen nano backbone.

## Ship

```bash
qml-ship --model large_nano_hybrid_higgs --skip-train
```

## Limitations

- QNN simulation on CPU (PennyLane)
- Requires frozen backbone from `large_nano_mlp_higgs`
