---
title: LargeNanoMLP Synthea CV
language: en
license: mit
tags:
  - tabular-classification
  - cardiovascular-risk
  - synthetic-ehr
datasets:
  - synthea_cv_risk_v1
---

# LargeNanoMLP — Synthea CV Risk

**Registry key:** `large_nano_mlp_synthea`  
**Experiment:** `exp_034`  
**Architecture:** `large_nano_mlp` (~1.17M parameters)

## Intended use

Research serve model for cardiovascular event ranking on synthetic Synthea CV cohort.
**Not for clinical deployment.** Use calibrated variant for human-facing scores.

## Ship

```bash
qml-ship --model large_nano_mlp_synthea --skip-train
qml-download --model large_nano_mlp_synthea
```

## Limitations

- Synthetic EHR with extreme label prevalence (~99% positive)
- Headline metric: ROC-AUC on balanced subsample, not accuracy
- Human validation: exp_041 Spearman ρ on 8 clinical cases
