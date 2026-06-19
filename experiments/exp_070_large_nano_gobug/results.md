# Results — EXP 070: LargeNanoMLP on GoBug file-level defects (C3 anchor)

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate (PR-AUC primary)

- Params: **1,131,137**
- Train rows: **27,172**
- Val rows: **5,822**
- Logistic val PR-AUC: **0.3097**
- LargeNanoMLP val PR-AUC: **0.3100**
- Advantage: **0.03 pp**
- Elapsed: **4.215s**

## Verdict
**rejected** — val PR-AUC beats logistic by ≥ 2.0 pp.

## Limitations
- GoBug combined subset (~39K rows); temporal proxy via sha ordering.
- Software defect benchmark — not production static analysis.
