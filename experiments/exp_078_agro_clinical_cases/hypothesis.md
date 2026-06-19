# Hypothesis — EXP 078: Agro Risk Lab Human Validation (8 Brazilian cases)

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Model:** `exp_060` LargeNanoMLP on `acyd_soy_brazil_v1` (C4 anchor)  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Question

When we score **eight fixed Brazilian soybean municipality scenarios** (favorable pampa → severe
MATOPIBA drought/heat), does the C4 nano model **rank low-yield risk** in agronomically
sensible order — without exposing humans to raw 37-dim climate features?

## Agro cases (literature / extension-service backed)

Cases ordered by **expected P(low yield)** (low → high). Rationale follows Brazilian soybean
climate-risk patterns: La Niña drought in BA/MA, heat during reproductive stage, NDVI canopy
stress (EMBRAPA / INMET agro-climate monitoring context).

| ID | Profile (summary) | Expected tier | Expected rank |
|----|-------------------|---------------|---------------|
| **L01** | MT benchmark — good rain, moderate heat, high NDVI | very_low | 1 |
| **L02** | RS pampa — balanced season | very_low | 2 |
| **L03** | PR traditional belt — mild stress | low | 3 |
| **L04** | GO cerrado — minor dry spells | low | 4 |
| **H01** | BA La Niña drought | high | 5 |
| **H02** | MG heat wave + high VPD | high | 6 |
| **H03** | PI compound drought + heat | very_high | 7 |
| **H04** | MA severe MATOPIBA stress | very_high | 8 |

## What I expect to happen

- Model **P(low yield)** monotonically non-decreasing with expected rank (Spearman ρ ≥ **0.85**).
- All **L01–L04** below all **H01–H04**: `min(H) − max(L) > 0` on probabilities.
- Dashboard Agro Risk Lab page surfaces same cases with top-3 climate drivers.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Spearman ρ (rank vs risk %) | ≥ **0.85** |
| Low vs high separation | min(H prob) − max(L prob) > **0** |
| Cases | **8** hand-crafted Brazilian scenarios |

## Known limitations

- Research benchmark — not official ZARC / not insurance advice.
- Self-ranking acceptable for v1 (no external agronomist panel yet).
- Absolute probabilities depend on ACYD label definition (below state-year median yield).
