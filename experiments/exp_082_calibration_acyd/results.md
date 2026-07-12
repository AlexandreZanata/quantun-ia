# Results — EXP 082: Isotonic Calibration (ACYD C4)

**Run date:** 2026-07-12  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Model:** `exp_060` LargeNanoMLP · `acyd_soy_brazil_v1` · seed 42

## Summary

| Metric | Value | Gate |
|--------|-------|------|
| Val rows | **5830** | — |
| Negatives | **3229** | — |
| ECE before | **0.0538** | — |
| ECE after | **0.0355** | ≤ 0.08 · < before |
| Brier before | **0.2282** | — |
| Brier after | **0.2237** | ≤ before |
| ROC-AUC before | **0.6789** | — |
| ROC-AUC after | **0.6767** | Δ ≥ −0.005 |
| Spearman ρ (agro) | **0.9820** | ≥ 0.85 |
| Elapsed | **0.5s** | — |

## Verdict
**accepted** — isotonic calibration on temporal ACYD val for Agro Risk Lab probabilities.

## Artifact

- `/data/dev/projects/webstorm/quantun-ia/artifacts/exp_082/large_nano_mlp_acyd_soy_brazil_v1/seed_42/calibration_isotonic.json`

## Limitations

- Temporal val fit only; not operational ZARC / insurance advice.
- Isotonic is monotone — ranking preservation is expected by construction.
