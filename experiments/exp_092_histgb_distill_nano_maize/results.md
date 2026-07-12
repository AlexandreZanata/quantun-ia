# Results — EXP 092: HistGB → ResidualNano distillation (ACYD maize)

**Run date:** 2026-07-12  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)
**Profile:** `publication`

## Validation gate

- Train rows: **151,956** | Val rows: **13,566**
- Student params: **840,321** | distill α: **1.0**
- HistGB teacher val AUC: **0.8178**
- Hard ResidualNano val AUC: **0.8075**
- Distill ResidualNano val AUC: **0.8130**
- Δ vs teacher: **-0.48 pp** (gate ≥ −1.0 pp)
- Δ vs hard control: **0.55 pp**
- Elapsed: **21.734s**

## Verdict
**accepted** — Phase D H-N3 soft-label distillation vs HistGB teacher.

## Limitations
- Soft BCE (not temperature KL); single seed; temporal val only.
- Agro-climate benchmark — not operational planting advice.
