# Results — EXP 102: NanoUNet CIFAR-10 I2I (2026-07-14)

**Verdict:** Confirmed
**Profile:** `publication` · **Device:** `cuda`
**Params:** 1,429,891

## Metrics

| Metric | Value |
|--------|-------|
| Final train noise-MSE | 0.0797 |
| Val denoise MSE | 0.080467 |
| FID-R18 (model vs val) | 153.93 |
| FID-R18 (noise null vs val) | 652.36 |
| Relative FID improvement | 0.764 |
| LPIPS-proxy (VGG) | 5.3706 |
| Elapsed (s) | 305.6 |

## Gate

- Primary H0: relative FID improvement ≥ 0.20 vs noise null.
- Outcome: **Confirmed** (Δ_rel = 0.764).

## Ablation suggestion

- What if you halve `base_channels` (capacity) vs double training epochs?

*Logged via ExperimentLogger · 2026-07-14T10:58:38.274140*
