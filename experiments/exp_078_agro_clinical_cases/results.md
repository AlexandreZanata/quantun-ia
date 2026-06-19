# Results — EXP 078: Agro Risk Lab Human Validation

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Model:** `exp_060` LargeNanoMLP · `acyd_soy_brazil_v1` · seed 42

## Summary

| Metric | Value | Gate |
|--------|-------|------|
| Spearman ρ (rank vs risk %) | **0.9762** | ≥ 0.85 |
| Max risk — low tier (L01–L04) | **39.84%** | — |
| Min risk — high tier (H01–H04) | **62.91%** | — |
| Separation (min H − max L) | **+23.076 pp** | > 0 |
| Strict monotonic (ε=1e-4) | **False** | informational |
| Elapsed | **0.3s** | — |

## Verdict
**accepted** — C4 model ordering vs agronomically expected soybean scenarios.

## Case-by-case results

| ID | Expected rank | Tier | Model risk % | Band | Title |
|----|---------------|------|--------------|------|-------|
| **L01** | 1 | very_low | **19.03%** | low | Favorable season — Lucas do Rio Verde (MT) |
| **L02** | 2 | very_low | **29.81%** | low | Normal pampa season — Passo Fundo (RS) |
| **L03** | 3 | low | **32.73%** | low | Paraná soybean belt — Londrina (PR) |
| **L04** | 4 | low | **39.84%** | moderate | GO frontier — Rio Verde with minor dry spells |
| **H01** | 5 | high | **64.29%** | moderate | La Niña drought — Barreiras (BA) |
| **H02** | 6 | high | **62.91%** | moderate | Heat wave — Uberaba (MG) |
| **H03** | 7 | very_high | **70.21%** | high | Compound drought + heat — Bom Jesus (PI) |
| **H04** | 8 | very_high | **77.76%** | high | Severe MATOPIBA stress — Balsas (MA) |

## Limitations

- ACYD research benchmark — not official ZARC / not insurance advice.
- Self-ranking v1; external agronomist panel deferred.
- Dashboard: `dashboard/pages/06_agro_risk_lab.py`.
