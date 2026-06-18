# Hypothesis — EXP 024: QuantumNano-BC Flagship Nano Model

**Date:** 2026-06-18  
**Author:** Quantum ML Lab  
**Pre-registration:** See [Pre-registration](#pre-registration) below.

## Pre-registration

- **Policy:** Publication-profile runs require OSF pre-registration before external citation.
- **Status:** Pending author filing (manual step before arXiv v1).
- **OSF URL:** _Paste `https://osf.io/...` after registration._

## What I expect to happen

On Wisconsin Breast Cancer (full 569 samples, 30% stratified holdout), a parameter-matched
`hybrid_sandwich` (4 qubits, 2 re-upload layers) will achieve holdout accuracy **within 2 pp**
of logistic regression (parity claim), **OR** exceed logistic regression by **≥3 pp** with
Holm-significant paired Wilcoxon (advantage claim).

Shallow XGBoost and parameter-matched classical MLP provide additional strong baselines; the
perceptron anchors the linear floor.

## Why I expect this

exp_011 and exp_022 show classical models are competitive on this UCI task; hybrid QML may
match but is unlikely to dominate logistic regression or XGBoost at nano parameter budget.
Full-dataset evaluation (no subsampling) and 30 seeds improve power over exp_022 (10 seeds).

## What would prove me wrong

- `hybrid_sandwich` exceeds logistic regression by ≥3 pp with Holm-significant Wilcoxon
- `hybrid_sandwich` trails logistic regression by >2 pp on mean holdout (classical ships as product)
- Any model reports holdout accuracy outside [0.50, 1.00] on full breast cancer (pipeline bug)

## Metrics I will measure

- [x] Holdout accuracy (30 seeds, bootstrap 95% CI)
- [x] Paired Wilcoxon: hybrid_sandwich vs logistic_regression, classical_matched, xgboost_shallow
- [x] Cohen's d on paired seed deltas
- [x] Parameter counts (hybrid vs matched classical)
- [x] Checkpoint + `model_cards/quantum_nano_bc.md` for flagship artifact

## Success criteria (shippable product)

- `qml-train --model hybrid_sandwich --dataset breast_cancer --profile publication` works out of the box
- `model_cards/quantum_nano_bc.md` documents limitations and expected metrics
- MicroQML Bench v1 lists this task as flagship real-world anchor

## Known limitations

- Not a clinical deployment claim — research benchmark only
- Simulator-only quantum execution (no hardware noise model)
