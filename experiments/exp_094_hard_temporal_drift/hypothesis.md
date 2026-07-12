# Hypothesis — EXP 094: Hard temporal drift on ACYD maize

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v2 · Phase C (C-T4)

## What I expect to happen

Under a **harder temporal split** (train ≤ **2016**, val **2017–2018**, test ≥ **2022**),
**ResidualNanoMLP** will stay within **1.0 pp** val ROC-AUC of **HistGB** on the same
hard-drift val set (nano ≥ HistGB − 1.0 pp).

## Why I expect this

- Standard split (train ≤ 2018 / val 2019–2021) already shows HistGB > nano (~0.9 pp).
- Harder drift widens the distribution shift; both models should degrade, but HistGB
  often retains ranking under covariate shift.
- This stress test is valuable even as an honest negative for paper cycle 2.

## What would prove me wrong

- Nano < HistGB − 1.0 pp on hard-drift val
- Empty / tiny splits after rebuild
- Year leakage (future features in train)

## Metrics I will measure

- [ ] Hard-drift train / val / test row counts
- [ ] HistGB val ROC-AUC (hard drift)
- [ ] ResidualNanoMLP val ROC-AUC (hard drift)
- [ ] Δ pp nano − HistGB (primary)
- [ ] Optional: standard-split HistGB reference (honesty)
- [ ] Wall-clock (build + train)

## Success criteria

- **Primary (C-T4):** nano ≥ HistGB − **1.0 pp** on hard-drift val
- `make check` green; ci smoke only in tests (no `tests/real/`)

## Known limitations

- Processed standard parquet has no year — hard-drift rebuilds from raw ACYD maize
- Single seed; agro research benchmark — not operational planting advice
- Honest negative expected if boosting remains more drift-robust
