# Results — EXP 104: Teacher→NanoUNet image distill (2026-07-14)

**Verdict:** Rejected
**Profile:** `publication` · **Device:** `cuda`
**Teacher params:** 2,991,363 · **Student params:** 476,675

## Metrics

| Metric | Value |
|--------|-------|
| FID-R18 teacher | 207.85 |
| FID-R18 hard student | 323.29 |
| FID-R18 distill student | 306.99 |
| Distill / teacher FID ratio | 1.477 |
| Relative FID win vs hard | 0.050 |
| Elapsed (s) | 487.6 |

## Gate (H-I1)

- Distill FID ≤ teacher × 1.10 **and** relative win vs hard ≥ 0.05.
- Outcome: **Rejected** (ratio=1.477, win=0.050).

## Ablation suggestion

- What if you remove soft targets (alpha=0) vs pure teacher (alpha=1)?

*Logged via ExperimentLogger · 2026-07-14T11:16:30.856789*
