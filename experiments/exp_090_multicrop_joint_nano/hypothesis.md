# Hypothesis — EXP 090: Multi-crop joint ResidualNano (soy + maize)

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v2 · Phase C (C-T1…C-T3)

## What I expect to happen

A **shared ResidualNano** trained on concatenated ACYD soy + maize train rows
(37 climate features + crop indicator → 38-d) will match a **maize-solo** ResidualNano
on maize temporal val within **0.5 pp** ROC-AUC (joint ≥ solo − 0.5 pp).

## Why I expect this

- Soy and maize share the same 37-d ACYD climate/soil schema and temporal cutoffs.
- Extra soy climate diversity may regularize the shared trunk without hurting maize ranking.
- Crop indicator (post-scaling) lets the model keep crop-specific bias without leakage.

## What would prove me wrong

- Joint maize val < solo − 0.5 pp
- Joint collapses toward chance / worse than logistic
- Scaling bug (crop bit scaled) or year leakage

## Metrics I will measure

- [ ] Maize-solo ResidualNano val ROC-AUC
- [ ] Joint ResidualNano maize-only val ROC-AUC
- [ ] Joint ResidualNano soy-only val ROC-AUC (secondary)
- [ ] HistGB maize val ROC-AUC (honesty floor)
- [ ] Δ pp joint − solo (primary)
- [ ] Train row counts by crop; wall-clock

## Success criteria

- **Primary (C0):** joint_maize ≥ solo_maize − **0.5 pp**
- `make check` green; ci smoke only in tests

## Known limitations

- Processed parquet has no year column — reuse existing temporal splits
- Single seed; hard labels (distill optional later)
- Agro research benchmark — not operational planting advice
