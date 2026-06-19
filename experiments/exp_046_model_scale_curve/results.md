# Results — EXP 046: Model scale curve on HIGGS

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Scale curve

| Variant | Params | Val ROC-AUC | Peak VRAM (MB) | Wall time (s) | Status |
|---------|--------|-------------|----------------|---------------|--------|
| nano_s | 84,673 | 0.8270 | 69.1 | 31.037 | ok |
| nano_m | 308,609 | 0.8313 | 104.1 | 29.776 | ok |
| nano_l | 1,141,377 | 0.8316 | 180.8 | 34.295 | ok |
| nano_xl | 4,445,441 | 0.8314 | 359.2 | 58.638 | ok |
| nano_xxl | 9,034,241 | 0.8316 | 429.2 | 105.226 | ok |

- Train rows: **805,000**
- Val rows: **172,500**
- nano_xl − nano_l: **-0.03 pp** (gate ≥ 0.3 pp)

## Verdict
**rejected / inconclusive**

## Limitations
- Single seed; multi-seed Wilcoxon deferred to overnight profile (Phase 1.2).
- Val-only metrics; test split untouched.
