# Results — EXP 106: Latent residual QNN (2026-07-14)

**Verdict:** Confirmed
**Profile:** `publication` · **Device:** `cuda`
**VAE params:** 382,291 · Classical 8,360 · Quantum 8,588

## Metrics

| Metric | Value |
|--------|-------|
| VAE train loss | 0.1208 |
| FID-R18 classical latent | 579.60 |
| FID-R18 quantum residual | 576.63 |
| Δ FID (q − classical) | -2.97 |
| Parity (≤ +1.0) | True |
| Advantage (≤ −2.0) | True |
| Elapsed (s) | 7.6 |

## Gate (H-Q3.1)

- Parity: quantum FID ≤ classical + 1.0; advantage only at ≤ classical − 2.0.
- Outcome: **Confirmed** (Δ=-2.97).

## Ablation suggestion

- What if you remove the classical MLP path (pure QNN residual only)?

*Logged via ExperimentLogger · 2026-07-14T11:47:43.036356*
