# Hypothesis — EXP 087: Fourier climate re-upload vs flat angle head (H-Q2.2)

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v2 · Phase B (H-Q2.2)

## What I expect to happen

On frozen **exp_092 distill ResidualNano** bottleneck features (`acyd_maize_brazil_v1`),
a **Fourier (sin/cos) feature map into a 4-qubit re-upload head** will beat a **flat
tanh→AngleEmbedding head** on ≥ **2/3** re-upload depth rungs (layers ∈ {1,2,3})
by val ROC-AUC.

## Why I expect this

- Agro climate covariates are seasonal / quasi-periodic; Fourier features can expose
  harmonic structure that a single angle embedding misses.
- H-Q2.1 residual skip failed (+0.07 pp); Fourier encoding is a **different** mechanism.
- Re-upload depth ladder tests whether Fourier helps more as depth grows.

## What would prove me wrong

- Fourier wins &lt; 2/3 rungs vs flat
- Both arms &lt; classical distill − 1.0 pp (parity fail)
- Barren / flat loss on either arm
- OOM / TorchLayer crash on RTX 4060

## Metrics I will measure

- [ ] Classical distill ResidualNano val ROC-AUC
- [ ] Flat re-upload AUC at layers 1 / 2 / 3
- [ ] Fourier re-upload AUC at layers 1 / 2 / 3
- [ ] Rung wins (Fourier &gt; flat)
- [ ] Δ pp each arm vs classical (parity)
- [ ] Trainable params; wall-clock

## Success criteria

- **Primary (H-Q2.2):** Fourier wins ≥ **2/3** rungs
- **Parity:** both arms ≥ classical − **1.0 pp** at deepest rung
- `make check` green; ci smoke only in tests

## Known limitations

- Hybrid fine-tune may cap train rows (QNN TorchLayer CPU-bound)
- Single seed; PennyLane `default.qubit`
- Agro research benchmark — not operational planting advice
