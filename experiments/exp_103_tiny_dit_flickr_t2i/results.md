# Results — EXP 103: TinyDiT Flickr8k T2I (2026-07-14)

**Verdict:** Rejected
**Profile:** `publication` · **Device:** `cuda`
**Params (model+embedder):** 505,776

## Metrics

| Metric | Value |
|--------|-------|
| CLIPScore model | 18.08 |
| CLIPScore noise | 17.36 |
| CLIPScore null caption | 21.36 |
| Δ vs noise | 0.72 |
| Δ vs null | -3.28 |
| Final loss | 0.2364 |
| Elapsed (s) | 13.8 |

## Gate (H-T4 / H1)

- CLIP_model ≥ CLIP_noise and CLIP_model ≥ CLIP_null + 0.5.
- Outcome: **Rejected** (Δnull=-3.28).

## Ablation suggestion

- What if you replace hash embedder with frozen OpenCLIP text features?

*Logged via ExperimentLogger · 2026-07-14T12:25:17.266597*
