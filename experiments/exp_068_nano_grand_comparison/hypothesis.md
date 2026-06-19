# Hypothesis — EXP 068: Nano Grand Comparison (C1–C4 Synthesis)

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** CPU synthesis (aggregates RTX 4060 publication runs; no new training)

## What I expect to happen

**No single quantum recipe** beats its same-domain classical anchor on **≥ 3/4 domains**
(HIGGS, NIHR, GoBug, ACYD) with **≥ +0.5 pp** on the primary metric (ROC-AUC or PR-AUC).

## Why I expect this

- exp_037/051/062 show QNN head-only parity (±0.2 pp), not large wins.
- exp_052/053/056/057 are honest negatives on ablations.
- exp_055 noise reg on GoBug is inconclusive (+0.50 pp on test only).
- exp_071 (GoBug hybrid) is still pending — matrix marks C3 QNN as `pending`.

## Pre-registered gates

| Gate | Threshold |
|------|-----------|
| Artifact export | `dist/leaderboards/nano_grand_comparison.json` written |
| LaTeX table | `paper/tables/grand_comparison.tex` written |
| JSONL log | `exp_068` multi_seed_summary appended |
| Win definition | quantum recipe Δ ≥ **+0.5 pp** vs classical reference |

## Known limitations

- Single-seed publication numbers from closed experiments (not re-run here).
- GoBug QNN head row incomplete until exp_071.
- Cross-domain claims require ≥2 domain wins with Holm correction (deferred to exp_080).
