# Results — EXP 045: GoBug file-level defect baseline

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Holdout metrics (PR-AUC primary)

| Model | Val PR-AUC |
|-------|------------|
| Logistic | 0.3097 |
| LargeNanoMLP | 0.3097 |

- Train rows: **27,172**
- Val rows: **5,822**
- Params: **82,113**
- nano − logistic: **0.00 pp**
- Wall time: **5.171s**

## Verdict
**rejected / inconclusive**

## Limitations
- go-bug-collector combined subset (~39k rows); temporal proxy via sha ordering.
- Hybrid head ablation deferred to exp_050.
