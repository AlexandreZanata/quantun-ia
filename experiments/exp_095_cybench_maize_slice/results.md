# Results — EXP 095: CY-Bench maize US slice (ResidualNano vs HistGB)

**Run date:** 2026-07-12  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)
**Profile:** `publication`
**Dataset:** `cybench_maize_us_v1` (AgML sample US designed features, EUPL-1.2)

## Validation gate

- Train rows: **7,170** | Val rows: **2,185** | Features: **26**
- HistGB AUC: **0.7855**
- ResidualNano AUC: **0.7474** (params 834,689)
- Δ nano − HistGB: **-3.80 pp** (gate ≥ -1.0)
- Elapsed: **2.27s**

## Verdict
**rejected** — Phase C C-T5 CY-Bench maize sample slice.

## Limitations
- AgML sample US slice only (full Zenodo maize archive ~6 GB not downloaded).
- Binary low-yield proxy — not official CY-Bench regression nRMSE/R².
- Yield-lag / `yield_trend` columns excluded to avoid trivial leakage; `label` excluded from features.
- Agro research benchmark — not operational planting advice.
