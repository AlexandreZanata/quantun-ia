# Results — EXP 111: Measurement-scheduled CFG (2026-07-14)

**Verdict:** Rejected
**Profile:** `publication` · **Device:** `cuda`
**Params:** 334,320

## Metrics

| Metric | Value |
|--------|-------|
| CLIPScore classical CFG | 17.64 |
| CLIPScore measurement schedule | 17.74 |
| Δ CLIP (meas − cfg) | 0.10 |
| Final train loss | 0.3150 |
| Elapsed (s) | 6.3 |

## Gate (H-Q3.6)

- Win: `CLIP_meas ≥ CLIP_cfg + 0.5`.
- Outcome: **Rejected** (Δ=0.10).

## Ablation suggestion

- What if keep_floor tracks continuous Softmax masks instead of Bernoulli?

*Logged via ExperimentLogger · 2026-07-14T12:44:45.509083*
