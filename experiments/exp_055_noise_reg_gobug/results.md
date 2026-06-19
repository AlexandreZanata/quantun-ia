# Results — EXP 055: Depolarizing noise on GoBug hybrid QNN

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)
**Eval split:** temporal test (latest sha-order 15%)

## Validation gate

- Depolarizing p: **0.03**
- Train rows: **27,172** · Val: **5,822** · Test: **5,824**
- Noiseless test PR-AUC: **0.3231**
- Noisy test PR-AUC: **0.3281**
- Advantage: **+0.50 pp**
- Elapsed: **189.063s**

## Verdict
**honest negative** — noisy vs noiseless hybrid on temporal test (gate ≥ 0.5 pp).
