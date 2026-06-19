# Results — EXP 060: LargeNanoMLP on ACYD Brazil soybean

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate

- Params: **1,159,809**
- Train rows: **50,107**
- Val rows: **5,830**
- Logistic val AUC: **0.6391**
- LargeNanoMLP val AUC: **0.6777**
- Advantage: **3.86 pp**
- Elapsed: **8.1s**

## Verdict
**accepted** — val AUC beats logistic by ≥ 2.0 pp.

## Limitations
- ACYD temporal val (2019–2021); test years ≥2022 untouched.
- Agro-climate benchmark — not operational planting advice.
