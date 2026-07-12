# Hypothesis — EXP 088: Classical-shadow / Pauli features → NarrowDeepNano (ACYD maize)

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v2 · Phase B (H-Q2.3)

## What I expect to happen

On `acyd_maize_brazil_v1` temporal val, a **64-d Pauli-expectation feature map**
(exact analytic limit of classical shadows on a fixed 4-qubit angle-encoded circuit)
fed into **NarrowDeepNano** will:

1. Match raw-feature NarrowDeepNano within **0.5 pp** val ROC-AUC  
2. Beat logistic regression by ≥ **+2.0 pp**

## Why I expect this

- Classical shadows / Pauli banks can expose multi-qubit correlations that a shallow
  MLP on raw climate columns may underuse.
- H-Q2.1 residual QNN failed (+0.07 pp); a **different mechanism** (feature-map, not
  trainable TorchLayer head) is the fail-forward next step.
- NarrowDeepNano is the VRAM-friendly classical head from Phase A.

## What would prove me wrong

- Shadow nano < classical NarrowDeep − 0.5 pp
- Shadow nano < logistic + 2.0 pp
- Feature extraction OOM / wall-clock blow-up on RTX 4060
- Flat / collapsed features (near-constant columns)

## Metrics I will measure

- [ ] LogisticRegression val ROC-AUC (raw 37-d)
- [ ] NarrowDeepNano val ROC-AUC (raw 37-d) — classical head
- [ ] NarrowDeepNano val ROC-AUC (64-d Pauli/shadow features)
- [ ] HistGB val ROC-AUC (honesty floor; not primary gate)
- [ ] Δ pp shadow − classical head; Δ pp shadow − logistic
- [ ] Feature extract + train wall-clock

## Success criteria

- **Primary (H-Q2.3):** shadow ≥ classical_head − **0.5 pp** AND shadow ≥ logistic + **2.0 pp**
- `make check` green; ci smoke only in tests (no `tests/real/`)

## Known limitations

- Pauli features use analytic `default.qubit` expectations (infinite-shot shadow limit),
  not finite-shot hardware shadows
- Publication may cap train rows for feature-extract wall-time on 4060
- Agro research benchmark — not operational planting advice
