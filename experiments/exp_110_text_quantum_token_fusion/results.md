# Results — EXP 110: Text–quantum token fusion (2026-07-14)

**Verdict:** Rejected
**Profile:** `publication` · **Device:** `cuda`
**Classical params:** 334,320 · **Quantum params:** 262,796

## Metrics

| Metric | Value |
|--------|-------|
| CLIPScore classical (null-quantum) | 17.62 |
| CLIPScore quantum fusion | 17.99 |
| Δ CLIP (q − classical) | 0.37 |
| Final loss classical | 0.3483 |
| Final loss quantum | 0.2793 |
| Elapsed (s) | 20.5 |

## Gate (H-Q3.5)

- Win: `CLIP_q ≥ CLIP_classical + 0.5`.
- Outcome: **Rejected** (Δ=0.37).

## Ablation suggestion

- What if you fuse CLIP token sequences (not pooled) via multi-token cross-attn?

*Logged via ExperimentLogger · 2026-07-14T12:35:39.424911*
