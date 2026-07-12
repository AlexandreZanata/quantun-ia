# Hypothesis — EXP 092: HistGB → ResidualNano soft-label distillation (ACYD maize)

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v2 · Phase D (H-N3)

## What I expect to happen

A **ResidualNanoMLP** student trained with **HistGradientBoosting soft labels**
(`predict_proba`) on `acyd_maize_brazil_v1` will reach val ROC-AUC within **1.0 pp**
of the HistGB teacher (floor ~0.8178 from exp_083/084), closing more of the agro
boosting gap than hard-label ResidualNano (−0.92 pp in exp_084).

## Why I expect this

- exp_084 showed architecture search alone cannot beat HistGB.
- Soft targets transfer the teacher's ranking geometry; soft BCE fits the existing
  sigmoid + `BCELoss` batched trainer without KL/logits refactors.
- ResidualNano (~0.84M params) remains a shippable nano vs a large ensemble teacher.

## What would prove me wrong

- Student val AUC < HistGB − 1.0 pp after epoch budget
- Distilled student ≤ hard-label ResidualNano (no distill benefit)
- OOM at batch 2048 on RTX 4060
- Soft-label train accuracy metrics look noisy while val AUC collapses (bug)

## Metrics I will measure

- [ ] HistGB teacher val ROC-AUC
- [ ] Hard-label ResidualNano control val ROC-AUC
- [ ] Distilled ResidualNano val ROC-AUC (primary)
- [ ] Δ pp vs teacher (student − HistGB) and vs hard control
- [ ] Student parameter count
- [ ] Wall-clock on RTX 4060

## Success criteria

- Distilled student ≥ HistGB − **1.0 pp** on temporal val (hard labels)
- Training completes without OOM; ci unit/smoke + `make check` green
- Prefer student that also beats hard-label control (secondary, not gate)

## Known limitations

- Soft BCE (not temperature-scaled KL) — natural for sigmoid students
- Single seed; HistGB `max_iter` matched to exp_084 publication
- Agro research benchmark — not operational planting advice
