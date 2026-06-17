# Hypothesis — EXP 023: Encoding × Backend Interaction on PCA-MNIST

**Date:** 2026-06-17  
**Author:** Quantum ML Lab  
**Pre-registration:** Pending OSF link before publication-profile runs

## What I expect to happen

On PCA-reduced MNIST digits (0 vs 1, 8 components, 4 qubits), the holdout accuracy gap between
angle and amplitude encoding will be **within 2 percentage points (pp)** on both `default.qubit`
and `lightning.qubit`. Backend choice (exp_021) and encoding choice (exp_012) should not interact:
each factor affects wall-clock or expressivity diagnostics, not a systematic cross-term in accuracy.

## Why I expect this

exp_012 compares encodings on a single backend; exp_021 shows backend parity for angle encoding on
breast cancer. Combining both on image-derived tabular data tests whether simulator implementation
details couple with encoding — they should not, because both backends implement the same variational
ansatz after the encoding layer.

## What would prove me wrong

- Encoding gap (amplitude minus angle) differs by **> 2 pp** between backends with Holm-significant Wilcoxon
- Backend gap within either encoding exceeds **> 2 pp** (violates exp_021 parity on this dataset)
- Either encoding stays near chance (~50%) after PCA compression across all backends

## Metrics I will measure

- [x] Holdout accuracy per encoding×backend cell (10 seeds, bootstrap 95% CI)
- [x] Paired Wilcoxon: encoding comparison within each backend
- [x] Paired Wilcoxon: backend comparison within each encoding
- [x] Cohen's d on paired seed deltas for primary contrasts

## Known limitations

- `amplitude_lightning` may fail on lightning's Mottonen decomposition (logged, seed skipped) — publication runs document per-cell coverage

## Ablation follow-ups

- Halve `n_layers` to 1 — does interaction vanish with fewer parameters?
- Swap to breast cancer (exp_021 dataset) — does encoding×backend decouple on pure tabular features?
