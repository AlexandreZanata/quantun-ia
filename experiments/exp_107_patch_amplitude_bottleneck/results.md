# Results — EXP 107: Patch amplitude bottleneck (2026-07-14)

**Verdict:** Rejected
**Profile:** `publication` · **Device:** `cuda`
**Classical params:** 1,600 · **Quantum params:** 1,696

## Metrics

| Metric | Value |
|--------|-------|
| Classical patch MSE | 0.0094 |
| Quantum patch MSE | 0.0714 |
| FID-R18 classical recon | 236.21 |
| FID-R18 quantum recon | 566.27 |
| |Δ FID| | 330.06 |
| Elapsed (s) | 11.6 |

## Gate (H-Q3.2)

- Parity: `|FID_q − FID_classical| ≤ 1.0`.
- Outcome: **Rejected** (|Δ|=330.06).

## Ablation suggestion

- What if you use grayscale 4×4 patches (exact 16-d amp, no RGB projection)?

*Logged via ExperimentLogger · 2026-07-14T11:56:01.940936*
