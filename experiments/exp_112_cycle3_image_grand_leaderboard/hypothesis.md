# Hypothesis — EXP 112: Cycle v3 Image Grand Leaderboard (Synthesis)

**Date:** 2026-07-14  
**Author:** Quantum ML Lab  
**Hardware:** CPU synthesis (aggregates RTX 4060 publication runs; no new training)  
**Cycle:** Research v3 · Phase K (K-T5)

## What I expect to happen

Across Research Cycle v3 image nano (`exp_101`–`exp_111`):

1. The curated leaderboard lists exactly the four accepts already closed on RTX 4060:
   **101, 102, 106, 109**.
2. No **rejected** quantum arm secretly clears a claim win
   (CLIP Δ ≥ **+0.5** or FID Δ ≤ **−2.0**).
3. Artifacts land at `dist/leaderboards/cycle3_grand_leaderboard.json` and
   `paper/tables/cycle3_grand_leaderboard.tex`.

## Why I expect this

- Phases G–J publication runs already closed with those verdicts.
- Residual (106) and circuit-cut (109) are accepted parity/advantage wins — not false claims.

## Pre-registered gates

| Gate | Threshold |
|------|-----------|
| Coverage | All `exp_101`–`exp_111` (incl. `105b`) rows with verdict + primary |
| Accepts | Set equals `{101, 102, 106, 109}` |
| Quantum honesty | 0 rejected quantum arms clearing CLIP/+FID claim thresholds |
| Artifacts | JSON + LaTeX written |
| JSONL | `exp_112` synthesis record appended |

## Known limitations

- Single-seed curated metrics from closed `results.md` files (not re-trained here).
- Absolute FID for latent quantum arms remains high vs NanoUNet I2I floor.
- TinyDiT Flickr ship deferred (exp_103 rejected); classical serve = NanoUNet CIFAR.
- Zenodo DOI / arXiv ID paste remains external (F-T4b).
