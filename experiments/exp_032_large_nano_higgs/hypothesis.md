# Hypothesis — EXP 032: LargeNanoMLP on HIGGS (805K train)

**Date:** 2026-06-18  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

A **~1.2M-parameter** `LargeNanoMLP` trained with mini-batches on the **HIGGS v1**
train split (805K rows, 28 features) will achieve **validation ROC-AUC at least 1 pp
above** a logistic regression baseline fit on the same scaled features.

## Why I expect this

- HIGGS is a structured tabular signal-vs-background problem where deep nonlinear
  models outperform linear baselines in published benchmarks.
- 805K rows with dropout 0.3 and weight decay 1e-4 provides enough regularization
  for ~1.14M parameters on RTX 4060 (batch 2048, no full-tensor OOM).
- Train-only `StandardScaler` avoids leakage; val split (172.5K) is held out for
  model selection only — test split untouched in this experiment.

## What would prove me wrong

- Val ROC-AUC ≤ logistic + 1 pp after epoch budget
- OOM on RTX 4060 at batch 2048 (would require architecture or batch reduction)
- Train accuracy >> val AUC with flat val curve → overfitting / leakage

## Metrics I will measure

- [ ] Parameter count (target ≥ 1,000,000, logged to `experiments.jsonl`)
- [ ] Logistic val ROC-AUC (baseline)
- [ ] LargeNanoMLP val ROC-AUC (primary)
- [ ] Δ AUC in percentage points (nano − logistic)
- [ ] Wall-clock elapsed on RTX 4060

## Success criteria

- `n_params` ≥ **1,000,000**
- Val AUC advantage ≥ **+1.0 pp** vs logistic on same val split
- Training completes without OOM on RTX 4060
- `make check-real` stays green after merge

## Known limitations

- HIGGS is physics tabular — not clinical; validates **machinery** before Synthea
  nano training (`exp_033+`).
- Test split not evaluated in this gate (val-only selection per Phase L protocol).
- Classical-only — hybrid QNN head deferred to later ablation.
