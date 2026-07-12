# Hypothesis — EXP 095: CY-Bench maize US slice ResidualNano vs HistGB (C-T5)

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v2 · Phase C (C-T5)

## What I expect to happen

On the official **AgML CY-Bench maize US sample** (designed feature table,
EUPL-1.2), **ResidualNanoMLP** reaches HistGB within **1.0 pp** ROC-AUC on the
temporal val years (2012–2015) for **low-yield** binary classification
(yield ≤ train-period median).

## Why I expect this

- Cycle v2 hard-drift ResidualNano already tracks HistGB within 1 pp on ACYD.
- CY-Bench US sample is an external agro panel with designed climate/RS features.
- Full Zenodo archive (~6 GB) exceeds practical download bandwidth on this
  workstation; AgML documents the sample_data path for getting started.

## What would prove me wrong

- ResidualNano < HistGB − 1.0 pp on val
- Build/download failure or empty temporal splits
- Label leakage (threshold fit on val/test)

## Metrics I will measure

- [ ] HistGB val ROC-AUC
- [ ] ResidualNano val ROC-AUC
- [ ] Δ pp (nano − HistGB)
- [ ] Train/val/test row counts; n_features; wall-clock

## Success criteria

- **Primary (C-T5):** nano ≥ HistGB − **1.0 pp**
- `make check` green; ci smoke only in tests
- Dataset registered as `cybench_maize_us_v1` in `data/open/manifest.json`

## Known limitations

- Sample US slice only (not full 38-country maize archive)
- Binary low-yield proxy — not official CY-Bench regression metrics (nRMSE/R²)
- Single seed; agro research benchmark — not operational advice
