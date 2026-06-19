# Hypothesis — EXP 060: LargeNanoMLP on ACYD Brazil soybean (C4 anchor)

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

A **~1.16M-parameter** `LargeNanoMLP` trained with mini-batches on **acyd_soy_brazil_v1**
(50K train rows, 37 climate/soil features) will achieve **validation ROC-AUC at least 2 pp
above** logistic regression on the same temporally held-out val split (2019–2021).

## Why I expect this

- Low-yield classification from weather/soil is nonlinear; logistic is a weak ceiling on tabular agro.
- 50K train rows with dropout 0.3 supports ~1.16M params on RTX 4060 (batch 2048).
- Train-only `StandardScaler`; temporal split prevents future-year leakage.

## What would prove me wrong

- Val ROC-AUC ≤ logistic + 2 pp after epoch budget
- OOM at batch 2048 (reduce batch or hidden width)
- Train accuracy >> val AUC with flat val curve → overfitting

## Metrics I will measure

- [ ] Parameter count (target ≥ 1,000,000)
- [ ] Logistic val ROC-AUC (baseline)
- [ ] LargeNanoMLP val ROC-AUC (primary)
- [ ] Δ AUC in percentage points (nano − logistic)
- [ ] Wall-clock elapsed on RTX 4060

## Success criteria

- `n_params` ≥ **1,000,000**
- Val AUC advantage ≥ **+2.0 pp** vs logistic on same val split
- Training completes without OOM on RTX 4060
- `make check-real` stays green after merge

## Known limitations

- ACYD Brazil soybean — agro-climate research benchmark, not operational ZARC/insurance.
- Test split (≥2022) not used for model selection in this gate.
- Classical-only — hybrid QNN head deferred to exp_062.
