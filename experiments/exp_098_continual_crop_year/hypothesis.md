# Hypothesis — EXP 098: Continual crop-year fine-tune (D-T4)

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v2 · Phase D (continual crop-year)

## What I expect to happen

On ACYD maize with temporal val years {2019, 2020, 2021}, a **ResidualNanoMLP**
trained **year-by-year** (chronological fine-tune on each train crop year ≤ 2018)
will reach val ROC-AUC ≥ a **joint** ResidualNano trained on all train years at
once, within **−1.0 pp** (no catastrophic forgetting relative to joint).

## Why I expect this

- Agro climate drifts slowly year-to-year; sequential fine-tune may retain prior
  year signal while adapting to new seasons.
- Hard temporal drift (exp_094) already showed ResidualNano tracks HistGB under
  stronger year shift; continual is the deferred D-T4 protocol.
- Matched epoch budget: joint epochs ≈ n_years × epochs_per_year.

## What would prove me wrong

- Continual val AUC &lt; joint − 1.0 pp
- Strong forgetting (mean backward AUC on prior years collapses)
- OOM / wall-clock blow-up on RTX 4060

## Metrics I will measure

- [x] Joint ResidualNano val ROC-AUC
- [x] Continual year-by-year ResidualNano val ROC-AUC
- [x] Δ pp continual − joint
- [x] Mean backward AUC on prior train years (forgetting probe)
- [x] HistGB honesty on val (not primary gate)
- [x] Wall-clock

## Success criteria

- **Primary (D-T4):** continual ≥ joint − **1.0 pp**
- `make check` green; ci smoke only in tests (no `tests/real/`)

## Known limitations

- Year column rebuilt from raw ACYD (standard parquet lacks year)
- Naive fine-tune (no EWC / replay) — honest lower bound for continual methods
- Agro research benchmark — not operational planting advice
