# Hypothesis — EXP 025: Pima Diabetes Generalization

**Date:** 2026-06-18  
**Author:** Quantum ML Lab  

## Pre-registration

- **Policy:** Publication-profile runs require OSF pre-registration before external citation.
- **Status:** Pending author filing (manual step).
- **OSF URL:** _Paste `https://osf.io/...` after registration._

## What I expect to happen

On Pima Indians Diabetes (OpenML id=37, 768 samples, 8 features, 30% holdout), a
parameter-matched `hybrid_sandwich` (4 qubits, 2 re-upload layers) will achieve holdout
accuracy **within 2 pp** of logistic regression (generalization parity vs exp_024 on breast
cancer), **OR** we report inconclusive / classical-preferred if QML trails by >2 pp.

## Why I expect this

exp_024 showed hybrid QML **parity** with logistic regression on Wisconsin Breast Cancer
(Δ=−0.5 pp, 30 seeds). Pima is a second canonical tabular benchmark with similar sample
size and feature count — if the nano architecture generalizes, parity should hold; if not,
we document honest cross-dataset limits.

## What would prove me wrong

- `hybrid_sandwich` exceeds logistic regression by ≥3 pp with Holm-significant Wilcoxon
- `hybrid_sandwich` trails logistic regression by >2 pp on mean holdout (QML not generalizing)
- Holdout accuracy outside [0.50, 1.00] on full Pima dataset (pipeline bug)

## Metrics I will measure

- [x] Holdout accuracy (CI profile: 2 seeds; publication: 30 seeds)
- [x] Paired Wilcoxon: hybrid_sandwich vs logistic_regression, xgboost_shallow
- [x] Cohen's d on paired seed deltas
- [x] Parameter counts (hybrid vs matched classical)

## Known limitations

- Not a clinical deployment claim — research benchmark only
- Simulator-only quantum execution
- OpenML fetch required on first run (`data/raw/openml/` cache)
