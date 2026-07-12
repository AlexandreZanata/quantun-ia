# Hypothesis — EXP 081: LargeNanoMLP on ACYD Brazil maize (C4b anchor)

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

A **~1.16M-parameter** `LargeNanoMLP` trained with mini-batches on **acyd_maize_brazil_v1**
(same 37 climate/soil features as C4 soybean, temporal split) will achieve **validation ROC-AUC
at least 2 pp above** logistic regression on the temporally held-out val split (2019–2021).

## Why I expect this

- Low-yield classification from weather/soil is nonlinear; logistic is a weak ceiling on tabular agro.
- The C4 soybean recipe (exp_060) already cleared +2 pp on the same feature template.
- Maize (ACYD corn) expands the agro panel without changing the LargeNanoMLP architecture.

## What would prove me wrong

- Val ROC-AUC ≤ logistic + 2 pp after epoch budget
- OOM at batch 2048 (reduce batch or hidden width)
- Train accuracy >> val AUC with flat val curve → overfitting
- Empty temporal join after maize ingest (builder bug)

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
- Dataset `acyd_maize_brazil_v1` ready with temporal crop-year split

## Known limitations

- ACYD Brazil maize (corn yield file) — agro-climate research benchmark, not operational ZARC/insurance.
- Season weeks 10–40 match soybean builder; maize phenology may differ by region/safrinha.
- Test split (≥2022) not used for model selection in this gate.
- Classical-only C4b anchor — hybrid QNN head deferred.
