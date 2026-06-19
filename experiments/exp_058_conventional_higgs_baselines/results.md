# Results — EXP 058: Conventional tabular baselines vs LargeNanoMLP (HIGGS)

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate (publication — primary)

- Train rows: **805,000** | Val rows: **172,500**
- Best conventional val ROC-AUC: **0.8429** (sklearn MLPClassifier)
- LargeNanoMLP val ROC-AUC: **0.8358** (shipped exp_032 checkpoint)
- Advantage: **−0.71 pp** (gate ≥ **+0.5 pp**)
- Elapsed: **686 s** (~11 min; sklearn MLP dominates wall time)

| Model | Val ROC-AUC | Val accuracy | Train (s) |
|-------|-------------|--------------|-----------|
| MLPClassifier (sklearn, 2048-512-64) | **0.8429** | 0.7607 | 679.2 |
| LargeNanoMLP (quantun-ia) | 0.8358 | 0.7533 | 0.2 |
| HistGradientBoosting (sklearn) | 0.8097 | 0.7303 | 3.3 |
| XGBoost shallow (xgboost) | 0.7773 | 0.7036 | 2.2 |
| LogisticRegression (sklearn) | 0.6849 | 0.6424 | 0.8 |

## CI slice (50K train / 10K val — smoke reference)

| Model | Val ROC-AUC |
|-------|-------------|
| LargeNanoMLP | **0.8328** |
| HistGradientBoosting | 0.7915 |
| sklearn MLP | 0.7747 |
| XGBoost | 0.7600 |
| Logistic | 0.6788 |

On the CI slice LargeNanoMLP leads by **+4.14 pp** vs best conventional; on full data
sklearn MLP overtakes after 679 s CPU training.

## Verdict

**rejected (publication)** — sklearn `MLPClassifier` with matched topology beats the shipped
LargeNanoMLP checkpoint by **+0.71 pp** on full val. Gate requires LargeNanoMLP ≥ best
conventional + 0.5 pp.

**Partial (CI)** — LargeNanoMLP wins on the 50K-row slice (+4.14 pp vs HistGradientBoosting).

## Interpretation

- LargeNanoMLP still crushes **logistic regression** (+15.1 pp on full val, consistent with exp_032).
- **HistGradientBoosting** and **XGBoost** (shallow defaults) do not beat our checkpoint.
- **Matched sklearn MLP** on 805K rows is the strongest conventional competitor — likely due to
  longer effective optimization on CPU despite fewer PyTorch-specific regularization tricks.
- Practical recommendation: retrain LargeNanoMLP with publication epochs if sklearn MLP parity
  is insufficient; do not claim universal superiority over all conventional stacks.

## Limitations

- Single seed (42); val split only (aligned with exp_032).
- LargeNanoMLP evaluated from frozen shipped checkpoint; baselines retrained each run.
- sklearn MLP `max_iter=12` did not fully converge (ConvergenceWarning) yet still won.
- Compare absolute metrics to literature only with protocol alignment (`docs/baselines.md`).
