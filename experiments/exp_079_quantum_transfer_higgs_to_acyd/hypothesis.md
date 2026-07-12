# Hypothesis — EXP 079: Cross-domain quantum head transfer (HIGGS → ACYD)

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`, PennyLane on CPU)

## What I expect to happen

A QNN head pretrained on frozen **C1 (HIGGS)** (`exp_037`), then fine-tuned head-only on
frozen **C4 (ACYD)** backbone, does **not** beat a scratch QNN head trained on the same
frozen C4 by **≥ +0.5 pp** val ROC-AUC.

## Why I expect this (honest-negative design)

- Backbone feature spaces differ (28-d physics vs 37-d agro); only the head tensors transfer
  (`head_proj` 4×64, `qlayer` 2×4×3, `post`).
- H-Q13 purpose: publish an honest negative on "quantum transfer" hype across domains.
- Backbone weights are **not** transferred (shape mismatch).

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Primary | Transfer − scratch val ROC-AUC **< +0.5 pp** → hypothesis **confirmed** (honest negative) |
| Dataset | `acyd_soy_brazil_v1` temporal val 2019–2021 |
| Arms | Scratch head vs HIGGS-init head; identical frozen exp_060 backbone + train protocol |

## Known limitations

- Head-only transfer; not full-model transfer.
- Single seed (42) unless extended; CI uses relaxed threshold.
- Research benchmark — not operational planting advice.
