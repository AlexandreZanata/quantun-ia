# Hypothesis — EXP 084b: ResidualNano soy transfer vs HistGB

**Date:** 2026-07-14  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v2 · Phase A (A-T4)

## What I expect to happen

**ResidualNanoMLP** (best arch from rejected exp_084 maize search) trained from scratch on
`acyd_soy_brazil_v1` will **not** beat HistGradientBoosting by ≥ **+0.5 pp** ROC-AUC on
temporal val. Transfer of the maize-chosen architecture is an honesty check that the boosting
gap is crop-agnostic, not maize-specific.

## Why I expect this

- exp_084 maize: ResidualNano 0.8086 vs HistGB 0.8178 (−0.92 pp).
- exp_061 soy: LargeNanoMLP 0.6777 vs HistGB 0.6941 (−1.64 pp).
- Architecture alone did not close the maize gap → same recipe on soy should still trail HistGB.

## What would prove me wrong

- ResidualNano ≥ HistGB + **0.5 pp** on soy val → unexpected soy win; revisit distill / publish.
- Honest tie (|Δ| ≤ 0.5 pp) → document parity; no win claim.
- OOM or train/val leakage → abort and fix split/scaler before interpreting AUC.

## Metrics I will measure

- [ ] HistGB val ROC-AUC (retrained same soy split)
- [ ] ResidualNanoMLP val ROC-AUC
- [ ] Δ AUC pp (ResidualNano − HistGB)
- [ ] Parameter count · wall-clock on RTX 4060

## Success criteria

- **Win:** ResidualNano ≥ HistGB + **0.5 pp**
- **Honest tie:** |Δ| ≤ 0.5 pp
- **Reject:** ResidualNano < HistGB − 0.5 pp → close A-T4; keep maize distill as serve default
- `make check` green with ci smoke only (no `check-real`)

## Known limitations

- Single seed; trains from scratch on soy (not weight transfer from maize checkpoint).
- Same architecture hyperparameters as exp_084 ResidualNano — no soy-specific retune.
- Agro research benchmark — not operational planting advice.
