# Results — EXP 030: Publication Large Scale Stability

**Run date:** 2026-06-18  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Scale stability

- n_samples: **1000**
- Seeds: **30** (reference: first **10**)
- Mean (reference): **62.37%**
- Mean (all): **64.30%**
- |Δmean|: **1.93 pp**
- Elapsed: **63.625s**

## Verdict
**accepted** — 30-seed hybrid mean within 2.0 pp of 10-seed reference.

## Limitations
- Circles synthetic data only; research prototype.
