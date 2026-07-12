# Results — EXP 093: Projected quantum kernel ridge head (ACYD maize)

**Run date:** 2026-07-12  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)
**Profile:** `publication`

## Validation gate

- Train rows: **15,000** | Val rows: **13,566**
- Projection dim: **12** (1-local X/Y/Z on 4q)
- LogisticRegression (raw) AUC: **0.6972**
- LogisticRegression (φ projections) AUC: **0.5082**
- KernelRidge RBF on φ AUC: **0.5307**
- Nyström→logistic AUC: **0.5193**
- HistGB (honesty) AUC: **0.7866**
- KernelRidge vs logistic: **-16.65 pp** (gate ≥ +0.5)
- Feature extract: **77.244s** | Elapsed: **100.118s**

## Verdict
**rejected** — Phase B H-Q2.6 projected quantum kernel ridge head.

## Limitations
- Analytic default.qubit projections (infinite-shot), not hardware shots.
- Soft PQK (1-local projections + classical RBF), not full fidelity kernel.
- Train rows capped for per-row QNode wall-time on RTX 4060.
- Agro research benchmark — not operational planting advice.
