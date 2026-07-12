# Results — EXP 096: GoBug streaming ResidualNano

**Run date:** 2026-07-12  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)
**Profile:** `publication`

## Validation gate

- Train rows: **27,172** | Val rows: **5,822**
- Stream batches: **8** | Params: **833,153**
- Logistic PR-AUC: **0.3097**
- Joint ResidualNano PR-AUC: **0.2995** (16 epochs)
- Streaming ResidualNano PR-AUC: **0.3069** (2 epochs/batch)
- Streaming vs joint: **0.75 pp** (gate ≥ -1.0)
- Elapsed: **5.675s**

## Verdict
**accepted** — Phase C C-T6 GoBug streaming nano.

## Limitations
- Sha sort is a temporal proxy, not true commit timestamps.
- Naive chronological fine-tune (no replay/EWC).
- GoBug research benchmark — not production defect triage.
