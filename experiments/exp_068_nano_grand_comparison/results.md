# Results — EXP 068: Nano Grand Comparison (C1–C4 Synthesis)

**Run date:** 2026-07-12  
**Hardware:** CPU synthesis (aggregates RTX 4060 publication runs)

## Artifacts

- JSON leaderboard: `dist/leaderboards/nano_grand_comparison.json`
- LaTeX table: `paper/tables/grand_comparison.tex`
- Recipes: **9** · Domains: **4**
- Elapsed: **0.007s**

## Quantum recipe wins (≥ +0.5 pp)

- `compound_stress_acyd`: **0** domain wins
- `entangle_schedule`: **0** domain wins
- `gv_alr_head`: **0** domain wins
- `large_nano_vs_conventional`: **0** domain wins
- `large_nano_vs_logistic`: **0** domain wins
- `noise_reg`: **2** domain wins
- `qnn_head_4q`: **0** domain wins
- `quantum_warmstart`: **1** domain wins
- `reupload_ladder_acyd`: **1** domain wins

## Verdict
**Hypothesis confirmed** — no quantum recipe wins on ≥3/4 domains simultaneously.

## Limitations
- Curated single-seed publication metrics; GoBug QNN head pending exp_071.
- Synthesis only — does not re-train models.
