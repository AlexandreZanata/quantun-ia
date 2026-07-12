# Hypothesis — EXP 089: Measurement-dropout QNN calibration (H-Q2.4)

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v2 · Phase B (H-Q2.4 / B-T4)

## What I expect to happen

On frozen **exp_092 distill ResidualNano** bottleneck features (`acyd_maize_brazil_v1`),
a **4-qubit re-upload head with stochastic measurement masking** (Bernoulli dropout on
Pauli-Z expectations) will reduce val **ECE by ≥ 20% relative** vs a plain QNN head,
without dropping ROC-AUC by more than **0.5 pp**.

## Why I expect this

- H-Q2.1–H-Q2.3 failed on AUC lifts; calibration is a different claim surface.
- Measurement dropout regularizes overconfident qubit readouts similarly to MC-dropout.
- Maize MC-dropout uncertainty already ships classically; quantum-side masking is the
  natural quantum analogue.

## What would prove me wrong

- Relative ECE improvement &lt; 20%
- AUC drop &gt; 0.5 pp vs plain QNN
- Both arms far below classical distill AUC (parity fail optional floor −1.0 pp)
- OOM / TorchLayer crash on RTX 4060

## Metrics I will measure

- [ ] Classical distill ResidualNano val ROC-AUC
- [ ] Plain QNN val ROC-AUC + ECE
- [ ] Measurement-dropout QNN val ROC-AUC (MC mean) + ECE
- [ ] Relative ECE improvement (%)
- [ ] Δ AUC pp (dropout − plain)
- [ ] Trainable params; wall-clock

## Success criteria

- **Primary (H-Q2.4):** ECE relative improvement ≥ **20%**
- **AUC floor:** dropout AUC ≥ plain − **0.5 pp**
- `make check` green; ci smoke only in tests

## Known limitations

- Frozen distill backbone; PennyLane TorchLayer on CPU
- Hybrid fine-tune row budget capped for QNN wall-time on 4060
- Agro research benchmark — not operational planting advice
