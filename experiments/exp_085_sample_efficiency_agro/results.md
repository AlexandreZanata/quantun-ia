# Results — EXP 085: Sample-efficiency curves (ACYD maize)

**Run date:** 2026-07-12  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)
**Profile:** `publication`

## Validation gate

- Distill wins: **0/4** (gate ≥ 2)
- AULC distill: **0.7831** | HistGB: **0.7973**
- Student params: **840,321** | Val rows: **13,566**
- Elapsed: **27.283s**

| Budget | Train rows | HistGB | Hard nano | Distill nano | Distill ≥ HistGB |
|--------|------------|--------|-----------|--------------|------------------|
| 1% | 1,520 | 0.6935 | 0.6916 | 0.6508 | no |
| 5% | 7,598 | 0.7807 | 0.7436 | 0.7461 | no |
| 20% | 30,391 | 0.8046 | 0.7749 | 0.7874 | no |
| 100% | 151,956 | 0.8178 | 0.8079 | 0.8130 | no |

## Verdict
**rejected** — Phase A/C H-N2 sample-efficiency (row-fraction proxy).

## Limitations
- Stratified **row** budgets (processed parquet has no crop-year column).
- Single seed; temporal val fixed across budgets.
- Agro-climate benchmark — not operational planting advice.
