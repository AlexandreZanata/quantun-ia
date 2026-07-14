# Results — EXP 108: Quantum flow coupling TinyDiT (2026-07-14)

**Verdict:** Rejected
**Profile:** `publication` · **Device:** `cuda`
**Classical params:** 235,312 · **Unitary params:** 231,216

## Metrics

| Metric | Value |
|--------|-------|
| FID-R18 classical coupling | 480.38 |
| FID-R18 unitary coupling | 499.42 |
| Relative improvement | -0.040 |
| Final loss classical | 0.2249 |
| Final loss unitary | 0.2365 |
| Elapsed (s) | 10.0 |

## Gate (H-Q3.3)

- Win: `FID_unitary ≤ FID_classical × 0.95` (≥ 5% relative).
- Outcome: **Rejected** (rel=-0.040).

## Ablation suggestion

- What if you move the unitary coupling to every other block (full flow stack)?

*Logged via ExperimentLogger · 2026-07-14T12:05:06.663850*
