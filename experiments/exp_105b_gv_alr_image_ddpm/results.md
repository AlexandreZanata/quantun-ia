# Results — EXP 105b: GV-ALR on NanoUNet DDPM (2026-07-14)

**Verdict:** Rejected
**Profile:** `publication` · **Device:** `cuda`
**Params:** 476,675

## Metrics

| Metric | Value |
|--------|-------|
| Fixed epochs | 12 |
| GV-ALR epochs | 8 |
| Epoch fraction | 0.667 |
| FID-R18 fixed-LR | 272.64 |
| FID-R18 GV-ALR | 308.58 |
| Relative FID delta | 0.132 |
| Fixed wall (s) | 49.7 |
| GV-ALR wall (s) | 34.8 |
| Elapsed (s) | 99.2 |

## Gate (H-I3)

- FID within ±3% relative **and** epochs ≤ 70% of fixed.
- Outcome: **Rejected** (Δ=0.132, frac=0.667).

## Ablation suggestion

- What if you adapt every N mini-batches instead of once per epoch?

*Logged via ExperimentLogger · 2026-07-14T11:38:38.517865*
