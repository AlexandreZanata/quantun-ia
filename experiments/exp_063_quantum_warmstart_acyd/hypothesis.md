# Hypothesis — EXP 063: Quantum warm-start on ACYD hybrid sandwich (C4 / H-Q9)

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`, PennyLane on CPU)

## What I expect to happen

Replicating **exp_052** on **ACYD Brazil soybean** (C4 / H-Q9): classical-first training for 70% of
epochs (phenology-style schedule), then enabling the QNN block, yields **≥ +0.5 pp ROC-AUC** over
end-to-end `HybridSandwich` on val. An honest negative is acceptable and informative.

## Why I expect this (or not)

- exp_052 on HIGGS was **−0.42 pp** (honest negative) — warm-start may not transfer to agro-climate tabular.
- ACYD has seasonal climate structure; classical-first may still find a useful basin before variational fine-tuning.
- Cross-domain replication tests whether H-Q9 / C4 warm-start fails or recovers on ROC-AUC.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Primary | warm-start val **ROC-AUC** ≥ end-to-end hybrid + **0.5 pp** (multi-seed mean) |
| Statistical | Wilcoxon p < 0.05 after Holm · **3 seeds** (publication) |
| Dataset | `acyd_soy_brazil_v1` temporal val only |

## Known limitations

- PennyLane QNN on CPU; HybridSandwich classical blocks on CUDA.
- CI uses 1 seed and relaxed gate — not a publication claim.
- Mirrors exp_052 sandwich protocol (not frozen C4 backbone).
