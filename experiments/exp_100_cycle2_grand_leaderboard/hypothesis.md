# Hypothesis — EXP 100: Cycle v2 Grand Leaderboard (Synthesis)

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** CPU synthesis (aggregates RTX 4060 publication runs; no new training)

## What I expect to happen

Across Research Cycle v2 (`exp_084`–`exp_099`):

1. **No quantum recipe** (residual QNN, Fourier re-upload, shadow features, measurement dropout, circuit-cut, PQK) beats its classical maize floor by **≥ +0.5 pp** ROC-AUC.
2. The curated leaderboard lists exactly the five accepts already closed on RTX 4060: **091, 092, 094, 096, 097**.
3. Distill ResidualNano (`exp_092`) remains the only classical nano within **1.0 pp** of HistGB on the standard ACYD maize temporal val.

## Why I expect this

- Phase A/B/C/D publication runs already closed with those verdicts.
- Phase F paper tables cover a subset; this experiment is the full Cycle v2 scorecard.

## Pre-registered gates

| Gate | Threshold |
|------|-----------|
| Coverage | All `exp_084`–`exp_099` rows present with verdict + primary metric |
| Quantum honesty | 0 quantum arms with Δ ≥ **+0.5 pp** vs classical floor |
| Accepts | Set equals `{091, 092, 094, 096, 097}` |
| Artifacts | `dist/leaderboards/cycle2_grand_leaderboard.json` + `paper/tables/cycle2_grand_leaderboard.tex` |
| JSONL | `exp_100` synthesis record appended |

## Known limitations

- Single-seed curated metrics from closed `results.md` files (not re-trained here).
- GoBug (`exp_096`) uses PR-AUC; maize arms use ROC-AUC — compare within-arm only.
- Zenodo DOI / arXiv ID paste remains an external upload step (not this experiment).
