# Hypothesis — EXP 097: SPEI-proxy curriculum on ACYD maize (D-T3)

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v2 · Phase D (deferred curriculum)

## What I expect to happen

On `acyd_maize_brazil_v1` temporal val, training **ResidualNanoMLP** with a
**SPEI-proxy curriculum** (order batches by drought severity easy→hard using the
seasonal precipitation-mean feature as a SPEI surrogate) will beat the same
architecture trained with a **random-order staged curriculum** by ≥ **+0.5 pp**
ROC-AUC, matched epoch budget.

## Why I expect this

- Agro low-yield risk is drought-skewed; presenting wet/easy seasons first may
  stabilize early gradients before hard drought rows.
- Distillation (exp_092) already closed; curriculum is the next deferred D-T3 arm.
- Precipitation mean is feature index 9 in the ACYD 37-d extractor (first weather
  block) — after StandardScaler, lower values remain drier (order preserved).

## What would prove me wrong

- SPEI curriculum &lt; random + 0.5 pp
- Flat / collapsed training / OOM on RTX 4060
- SPEI ordering identical to random within noise (no drought signal in precip mean)

## Metrics I will measure

- [x] Random-order staged curriculum val ROC-AUC
- [x] SPEI easy→hard staged curriculum val ROC-AUC
- [x] Δ pp SPEI − random
- [x] HistGB honesty AUC (not primary gate)
- [x] Trainable params; wall-clock

## Success criteria

- **Primary (D-T3):** SPEI ≥ random + **0.5 pp**
- `make check` green; ci smoke only in tests (no `tests/real/`)

## Known limitations

- SPEI is a precipitation-mean z-order proxy, not a full multi-timescale SPEI index
- Curriculum uses cumulative stages + refine (same total epochs as random)
- Agro research benchmark — not operational planting advice
