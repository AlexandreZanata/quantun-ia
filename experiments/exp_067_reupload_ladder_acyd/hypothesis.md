# Hypothesis — EXP 067: Re-upload depth curriculum on ACYD climate feature blocks (H-Q11)

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`, PennyLane on CPU)

## What I expect to happen

A curriculum that increases **re-upload layers (1→2→3)** beats a **fixed 3-layer** re-upload QNN on
**≥ 2 of 3** ACYD climate feature-block rungs (`temp_only`, `temp_precip`, `full_37`) — with
**≥ +0.3 pp ROC-AUC** advantage per winning rung (H-Q11).

## Why I expect this (or not)

- exp_056 validated the depth curriculum on a multi-dataset ladder; here the ladder is feature blocks
  within a single agro-climate domain.
- Growing depth with increasing climate complexity (temps → precip → full 37) may limit early
  barren-plateau exposure.
- Honest negative acceptable if fixed L=3 already saturates on narrow climate slices.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Primary | curriculum wins **≥ 2 / 3** rungs |
| Per rung | curriculum ≥ fixed L=3 + **0.3 pp ROC-AUC** |
| Dataset | `acyd_soy_brazil_v1` with column masks per rung |

## Known limitations

- PennyLane QNN sim on CPU; batched path on CUDA when available.
- Feature slices are fixed index ranges (not learned selection).
- CI uses row caps and relaxed gates — not a publication claim.
