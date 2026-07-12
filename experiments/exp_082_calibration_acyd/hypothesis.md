# Hypothesis — EXP 082: Isotonic calibration on ACYD C4 (agro)

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Model:** `exp_060` LargeNanoMLP (`large_nano_mlp_acyd_soy`) on `acyd_soy_brazil_v1`  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Question

Can **isotonic calibration** on the temporal ACYD **val** split improve **ECE** (and not
worsen Brier) while preserving **agro case ranking** (Spearman ρ on 8 Brazilian scenarios)?

## What I expect to happen

- C4 raw probabilities are useful for ranking (exp_078 ρ≥0.85) but may be miscalibrated for
  P(low yield) displayed in Agro Risk Lab.
- Isotonic regression on a stratified val fit slice is monotonic → Spearman ρ (raw vs
  calibrated) on agro cases stays ≥ **0.85**.
- ECE on held-out val slice drops to ≤ **0.08**; Brier does not increase; ROC-AUC Δ ≥ **−0.005**
  (≤ 0.5 pp discrimination loss — isotonic can flatten ties).

## Success criteria

| Metric | Gate |
|--------|------|
| ECE after | ≤ **0.08** and `<` ECE before |
| Brier after | ≤ Brier before |
| ROC-AUC Δ | ≥ **−0.005** |
| Spearman ρ (agro cases) | ≥ **0.85** |
| Artifact | `artifacts/exp_082/.../calibration_isotonic.json` |

## Known limitations

- Calibrator fit on temporal val only (2019–2021); test ≥2022 is secondary reporting only.
- Research probabilities — not ZARC / insurance advice.
- Platt scaling deferred; isotonic chosen for monotonicity.
