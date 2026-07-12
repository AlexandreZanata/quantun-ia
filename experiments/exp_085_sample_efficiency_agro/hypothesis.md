# Hypothesis — EXP 085: Sample-efficiency curves (HistGB vs distill nano) on ACYD maize

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v2 · Phase A/C (H-N2)

## What I expect to happen

At **5%** and **20%** stratified train-row budgets, a **soft-label ResidualNano**
(HistGB teacher fit on the same budget) will match or beat HistGB val ROC-AUC.
Overall, nano wins on **≥ 2 of 4** budgets `{1%, 5%, 20%, 100%}`.

## Why I expect this

- Distillation (exp_092) already closed most of the full-data gap (−0.48 pp).
- Neural nets often use labeled data more sample-efficiently than deep trees when
  teacher soft labels regularize the student.
- Processed parquet has no crop-year column — budgets are **stratified row fractions**
  (documented proxy for year-subsampling).

## What would prove me wrong

- Nano loses at both 5% and 20% budgets
- Nano wins on < 2 of 4 budgets
- Area-under-learning-curve (AULC) for nano < HistGB
- OOM / crash on RTX 4060

## Metrics I will measure

- [ ] Val ROC-AUC per budget for HistGB, hard nano, distill nano
- [ ] Wins count (distill ≥ HistGB) across budgets
- [ ] AULC (trapezoid over fraction axis) distill vs HistGB
- [ ] Wall-clock on RTX 4060
- [ ] JSON curves artifact under `artifacts/exp_085/`

## Success criteria

- Distill nano ≥ HistGB on **≥ 2/4** budgets (primary gate)
- Prefer wins at **5%** and **20%** (H-N2 strong claim; secondary)
- `make check` green with ci smoke only

## Known limitations

- Row-fraction proxy (no year IDs in processed parquet)
- Single seed; same temporal val split for all budgets
- Agro research benchmark — not operational planting advice
