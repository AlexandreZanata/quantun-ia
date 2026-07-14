# Results — EXP 109: Circuit-cut latent 6q (2026-07-14)

**Verdict:** Confirmed
**Profile:** `publication` · **Device:** `cuda`
**VAE params:** 382,291 · Classical 8,360 · Cut 8,726

## Metrics

| Metric | Value |
|--------|-------|
| VAE train loss | 0.1208 |
| FID-R18 classical latent | 579.60 |
| FID-R18 circuit-cut 6q | 576.59 |
| Δ FID (cut − classical) | -3.01 |
| Parity (≤ +1.0) | True |
| Elapsed (s) | 18.0 |

## Gate (H-Q3.4)

- Parity: circuit-cut FID ≤ classical + 1.0.
- Outcome: **Confirmed** (Δ=-3.01).

## Ablation suggestion

- What if you remove the classical MLP base (pure cut residual only)?

*Logged via ExperimentLogger · 2026-07-14T12:13:06.912170*
