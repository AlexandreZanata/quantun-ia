# Results — EXP 081: LargeNanoMLP on ACYD Brazil maize

**Run date:** 2026-07-12  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate

- Params: **1,159,809**
- Train rows: **151,956**
- Val rows: **13,566**
- Logistic val AUC: **0.6983**
- LargeNanoMLP val AUC: **0.8086**
- Advantage: **11.03 pp**
- Elapsed: **15.267s**

## Verdict
**accepted** — val AUC beats logistic by ≥ 2.0 pp.

## Limitations
- ACYD temporal val (2019–2021); test years ≥2022 untouched.
- Maize phenology may differ from soybean season weeks 10–40.
- Agro-climate benchmark — not operational planting advice.
