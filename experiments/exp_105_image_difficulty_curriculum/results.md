# Results — EXP 105: Image difficulty curriculum (2026-07-14)

**Verdict:** Rejected
**Profile:** `publication` · **Device:** `cuda`
**Params:** 476,675

## Metrics

| Metric | Value |
|--------|-------|
| FID-R18 random staged | 307.65 |
| FID-R18 sharpness curriculum | 309.01 |
| Relative FID win vs random | -0.004 |
| Elapsed (s) | 93.7 |

## Gate (H-I2)

- Relative FID win ≥ 0.05 vs random staged (matched epoch budget).
- Outcome: **Rejected** (win=-0.004).

## Ablation suggestion

- What if you order by FFT high-frequency energy instead of Laplacian variance?

*Logged via ExperimentLogger · 2026-07-14T11:28:45.269676*
