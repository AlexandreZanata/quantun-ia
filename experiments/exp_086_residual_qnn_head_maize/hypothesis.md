# Hypothesis — EXP 086: Residual-skip QNN head vs plain QNN on distill ResidualNano

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v2 · Phase B (H-Q2.1)

## What I expect to happen

On frozen **exp_092 distill ResidualNano** bottleneck features (`acyd_maize_brazil_v1`),
a **4-qubit re-upload head with classical residual skip** will beat a **plain QNN head**
by ≥ **+0.5 pp** val ROC-AUC. Both heads should stay within **1.0 pp** of the full
classical distill ResidualNano (parity gate).

## Why I expect this

- Skip connections stabilize deep classical nets; a classical skip around a shallow QNN
  may preserve ranking signal if the quantum branch underfits.
- Cycle-1 plain hybrids often tie classical within 1 pp; residual skip is a **new**
  mechanism (not warm-start / entangle / fusion).
- Distill backbone (0.8130 AUC) is the strongest maize nano trunk available.

## What would prove me wrong

- Residual-skip Δ vs plain QNN < +0.5 pp
- Either hybrid < classical − 1.0 pp (parity fail)
- Barren / flat loss from epoch 1 on the QNN branch
- OOM / TorchLayer crash on RTX 4060 workstation

## Metrics I will measure

- [ ] Classical distill ResidualNano val ROC-AUC
- [ ] Plain QNN hybrid val ROC-AUC
- [ ] Residual-skip QNN hybrid val ROC-AUC
- [ ] Δ pp residual − plain (primary)
- [ ] Δ pp each hybrid − classical (parity)
- [ ] Trainable param counts; wall-clock

## Success criteria

- **Primary (H-Q2.1):** residual_skip ≥ plain + **0.5 pp**
- **Parity:** both hybrids ≥ classical − **1.0 pp**
- `make check` green; ci smoke only in tests

## Known limitations

- Hybrid head fine-tune may use a soy-scale train-row budget for QNN wall-time on 4060
- Single seed; PennyLane `default.qubit` TorchLayer on CPU
- Agro research benchmark — not operational planting advice
