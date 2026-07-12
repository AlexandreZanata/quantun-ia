# Hypothesis — EXP 099: Masked climate SSL pretrain (D-T5)

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v2 · Phase D (SSL climate)

## What I expect to happen

On `acyd_maize_brazil_v1` temporal val, a **ResidualNanoMLP** whose stem+blocks are
**pretrained** by reconstructing **masked weather features** (indices 9–36), then
fine-tuned on low-yield labels, will beat an identical **from-scratch** ResidualNano
by ≥ **+0.5 pp** ROC-AUC (matched supervised fine-tune epochs).

## Why I expect this

- Weather aggregates dominate agro ranking; a reconstruction pretext may learn
  climate structure before the scarce yield label.
- Distillation / SPEI / continual are closed; SSL is the last deferred D-T5 arm.
- Masking only weather columns (not lat/lon/soil) matches the “masked weather week”
  claim without inventing weekly tensors from seasonal aggregates.

## What would prove me wrong

- SSL fine-tune &lt; scratch + 0.5 pp
- Pretrain collapses (constant reconstructions) / OOM on RTX 4060
- SSL hurts ranking (negative transfer)

## Metrics I will measure

- [x] From-scratch ResidualNano val ROC-AUC
- [x] SSL-pretrained → fine-tune ResidualNano val ROC-AUC
- [x] Δ pp SSL − scratch
- [x] Pretrain reconstruction MSE (train)
- [x] HistGB honesty AUC (not primary gate)
- [x] Wall-clock

## Success criteria

- **Primary (D-T5):** SSL ≥ scratch + **0.5 pp**
- `make check` green; ci smoke only in tests (no `tests/real/`)

## Known limitations

- Pretext uses seasonal weather aggregates (mean/std/min/max), not raw weekly tensors
- Matched supervised epochs; SSL also spends pretrain epochs (honest SSL protocol)
- Agro research benchmark — not operational planting advice
