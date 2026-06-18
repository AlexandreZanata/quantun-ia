# Results — EXP 029: Batch Calculation vs API Parity

**Run date:** 2026-06-18  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Parity

- Rows scored: **569**
- Max |Δp|: **2.98e-08**
- Mean |Δp|: **5.24e-11**
- Batch elapsed: **0.179s**
- API elapsed: **0.267s**

## Verdict
**accepted** — batch script matches API within 1e-5 per row.

## Limitations
- Wisconsin Breast Cancer only; research prototype.
