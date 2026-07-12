# Hypothesis — EXP 093: Projected quantum kernel ridge head (H-Q2.6)

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v2 · Phase B (H-Q2.6)

## What I expect to happen

On `acyd_maize_brazil_v1` temporal val, a **projected quantum kernel (PQK)** built from
**1-local Pauli expectations** of a fixed 4-qubit angle-encoded circuit, with a
**KernelRidge** head on those projections, will beat **LogisticRegression on raw
features** by ≥ **+0.5 pp** ROC-AUC.

## Why I expect this

- PQK (Huang-style projections → classical kernel) is a different mechanism from
  trainable TorchLayer heads (H-Q2.1/2.2/2.4/2.5) and from feeding a large Pauli
  bank into NarrowDeep (H-Q2.3 failed hard).
- Local projections keep the quantum map cheap on a laptop 4060 while the RBF
  kernel can express non-linear agro ranking.
- Experiment id `pqk_ridge_head` matches KernelRidge on projected features.

## What would prove me wrong

- PQK KernelRidge &lt; logistic + 0.5 pp
- Projections collapse (near-constant φ) / OOM / wall-clock blow-up on RTX 4060
- Nyström linear head ≫ KernelRidge (kernel choice broken) without clearing gate

## Metrics I will measure

- [x] LogisticRegression val ROC-AUC (raw 37-d)
- [x] Linear / logistic head on quantum projections φ (honesty)
- [x] KernelRidge (RBF) on φ — primary PQK arm
- [x] Nyström RBF → logistic (PQK + linear head honesty)
- [x] HistGB val ROC-AUC (honesty floor; not primary gate)
- [x] Δ pp KernelRidge − logistic; feature-extract + train wall-clock

## Success criteria

- **Primary (H-Q2.6):** KernelRidge(PQK) ≥ logistic + **0.5 pp**
- `make check` green; ci smoke only in tests (no `tests/real/`)

## Known limitations

- Analytic `default.qubit` projections (infinite-shot), not hardware shots
- Publication caps train rows for per-row QNode wall-time on 4060
- Soft PQK (1-local projections + classical RBF), not full fidelity kernel
- Agro research benchmark — not operational planting advice
