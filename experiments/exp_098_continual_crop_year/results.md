# Results — EXP 098: Continual crop-year fine-tune (ACYD maize)

**Run date:** 2026-07-12  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)
**Profile:** `publication`

## Validation gate

- Train rows: **151,956** | Val rows: **13,566**
- Train years: **37** | Params: **840,321**
- Joint ResidualNano AUC: **0.8005** (74 epochs)
- Continual year-by-year AUC: **0.7713** (2 epochs/year)
- Backward mean AUC (prior years): **0.7078**
- HistGB (honesty) AUC: **0.8203**
- Continual vs joint: **-2.91 pp** (gate ≥ -1.0)
- Elapsed: **86.412s**

## Verdict
**rejected** — Phase D D-T4 continual crop-year fine-tune.

## Limitations
- Naive fine-tune without EWC/replay (honest lower bound).
- Year column rebuilt from raw ACYD into processed/continual_v1.
- Agro research benchmark — not operational planting advice.
