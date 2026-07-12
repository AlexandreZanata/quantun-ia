# Hypothesis — EXP 084: Residual / FT-lite nano vs HistGB on ACYD maize

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v2 · Phase A (H-N1)

## What I expect to happen

At least one of **ResidualNano**, **NarrowDeepNano**, or **FT-lite** will match or beat the
exp_083 **HistGradientBoosting** floor (**0.8178** ROC-AUC) on `acyd_maize_brazil_v1`
temporal val (2019–2021). A publication **win claim** requires ≥ **+0.5 pp** vs HistGB
(3 architectures, seed 42 primary; multi-seed deferred).

## Why I expect this

- Cycle-1 `LargeNanoMLP` (2048-512-64) cleared logistic by +11 pp but lost to HistGB by −0.92 pp.
- Residuals and feature tokenization are standard tabular upgrades that HistGB-style
  interaction capture may need; NarrowDeep trades width for depth on the same 37 features.
- Same temporal split and scaling as exp_081/083 — only architecture changes.

## What would prove me wrong

- Best nano val AUC < HistGB − 0.5 pp after all three architectures
- OOM at batch 2048 on RTX 4060 (reduce batch / d_model)
- Train accuracy >> val AUC with flat val → overfitting
- HistGB retrain on same split diverges materially from 0.8178 (split/scaler bug)

## Metrics I will measure

- [ ] HistGB val ROC-AUC (floor, retrained same split)
- [ ] ResidualNano / NarrowDeepNano / FT-lite val ROC-AUC
- [ ] Δ AUC pp (best nano − HistGB)
- [ ] Parameter counts per architecture
- [ ] Wall-clock on RTX 4060

## Success criteria

- **Win:** best nano ≥ HistGB + **0.5 pp**
- **Honest tie:** |Δ| ≤ 0.5 pp — document, no win claim
- **Reject:** best nano < HistGB − 0.5 pp → prefer Phase D distillation next
- Training completes without OOM; `make check` green with ci smoke only

## Known limitations

- Single seed in this gate (multi-seed Holm deferred to publication_seeds profile)
- Agro research benchmark — not ZARC / insurance advice
- Test years ≥2022 unused for model selection
