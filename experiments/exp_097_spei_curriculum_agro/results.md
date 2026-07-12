# Results — EXP 097: SPEI-proxy curriculum (ACYD maize)

**Run date:** 2026-07-12  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)
**Profile:** `publication`

## Validation gate

- Train rows: **151,956** | Val rows: **13,566**
- ResidualNano params: **840,321**
- Random staged curriculum AUC: **0.7942**
- SPEI easy→hard curriculum AUC: **0.8025**
- HistGB (honesty) AUC: **0.8203**
- SPEI vs random: **0.83 pp** (gate ≥ +0.5)
- Elapsed: **16.327s**

## Verdict
**accepted** — Phase D D-T3 SPEI-proxy curriculum.

## Limitations
- SPEI is precipitation-mean order proxy (feature index 9), not full SPEI.
- Matched staged+refine epoch budget vs random permutation curriculum.
- Agro research benchmark — not operational planting advice.
