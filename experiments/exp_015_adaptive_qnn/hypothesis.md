# Hypothesis — EXP 015: Adaptive QNN (Gradient-Variance LR)

**Date:** 2026-06-17  
**Author:** Quantum ML Lab  
**Direction:** Phase 4 Option A — noise-aware quantum training

## Research question

Can a gradient-variance-aware learning rate schedule improve holdout accuracy on
plateau-prone variational circuits **without reducing qubit count**?

## Null hypothesis (H₀)

Adaptive LR (scaled from per-step gradient variance, calibrated via exp_006) does **not**
improve holdout accuracy vs fixed Adam LR at the same architecture (6 qubits, 3 layers).

## Alternative hypothesis (H₁)

Adaptive LR yields higher mean holdout accuracy on 6q×3l QNN with effect size
|Cohen's d| ≥ 0.5 vs fixed LR across 10 publication seeds.

## Why we expect a signal

exp_006 confirmed vanishing gradient variance as qubits increase (barren plateau).
When variance drops below a target, increasing LR may escape flat regions while
`min_scale` / `max_scale` caps prevent instability. Cincio et al. (2022) link cost
landscape geometry to trainability; variance-based scaling is a lightweight countermeasure.

## Models compared

| Model | Description |
|-------|-------------|
| `quantum_6q_3l_fixed` | Plateau-prone baseline, fixed LR |
| `quantum_6q_3l_adaptive` | Same architecture, variance-scaled LR |
| `quantum_4q_2l_fixed` | Shallower circuit control (higher init variance) |
| `classical_matched_h*` | Parameter-matched classical baseline (6q target) |

## Pre-defined ablations (≥3)

1. **Adaptive vs fixed** at 6q×3l — primary claim test.
2. **Qubit depth** — adaptive benefit larger at 6q than 4q (interaction with plateau severity).
3. **`var_target` sensitivity** — sweep {0.005, 0.015, 0.03} in follow-up runs (documented; default 0.015 from exp_006 4q scale).
4. **Warmup epochs** — {0, 3, 5} ablation if primary result is inconclusive.

## Metrics

- [x] Holdout accuracy (10 seeds, bootstrap 95% CI)
- [x] Paired Wilcoxon: adaptive vs fixed (seed-aligned)
- [x] Cohen's d effect size (not p-values alone)
- [x] Holm-Bonferroni across planned comparisons
- [x] Mean final learning rate and gradient variance logged per epoch

## Success criteria

- Mean holdout gain ≥ 2 pp for adaptive vs fixed at 6q, **or**
- Cohen's d ≥ 0.5 with Holm-significant Wilcoxon, **or**
- Honest negative documented in `results.md` with limitations

## What would prove us wrong

- Adaptive LR matches fixed LR within 1 pp (CIs overlap entirely)
- Adaptive LR **degrades** stability (holdout variance increases > 2×)
- Benefit only appears at 4q (trivial regime) but not 6q

## Literature anchors

See `docs/literature_review.md` — McClean et al. (2018), Cincio et al. (2022), Grant et al. (2019).

## Limitations (to fill after run)

- [ ] Circles dataset only in primary run — generalization to UCI/MNIST pending
- [ ] Single optimizer (Adam) — SPSA not tested
- [ ] `var_target` hand-tuned from exp_006, not learned
