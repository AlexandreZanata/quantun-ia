# Results — EXP 039: Regularized LargeNanoMLP on Synthea CV

**Run date:** 2026-06-18  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Regularization

- Dropout: **0.5** (exp_034: 0.3)
- Weight decay: **0.001** (exp_034: 1e-4)
- Learning rate: **0.0005** (exp_034: 0.001)

## Validation gate

- Params: **1,165,953**
- Train rows: **700,000**
- Val rows: **150,000**
- Logistic val AUC: **0.7949**
- Regularized nano val AUC: **0.7889**
- Δ vs logistic: **-0.60 pp**
- exp_034 nano reference: **0.7867**
- Δ vs exp_034: **+0.22 pp**
- Elapsed: **76.801s**

## Verdict
**accepted** — beat logistic stretch goal: **no**.

## Limitations
- Synthea synthetic EHR — research prototype, not clinical deployment.
- Test split not used for model selection in this gate.
