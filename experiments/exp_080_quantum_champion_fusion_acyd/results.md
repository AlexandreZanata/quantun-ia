# Results — EXP 080: Quantum champion fusion on ACYD (C4)

**Run date:** 2026-07-12  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gates (ROC-AUC)

- Train rows: **50,107** · Val rows: **5,830**
- Trainable head params: **289**
- Schedule: **3** warm-start + **5** GV-ALR · depolarizing p=**0.03** (train-only)
- Classical C4: **0.6777**
- Best hybrid baseline: **0.6771**
- Champion (noiseless eval): **0.6709**
- vs classical: **-0.68 pp**
- vs best hybrid: **-0.62 pp**
- Elapsed: **305.016s**

## Verdict
**honest negative** — parity vs C4 (≥ -1.0 pp) and lift vs best frozen hybrid (≥ 0.5 pp).

## Limitations
- Train-time noise on `default.mixed`; eval copies weights to noiseless head.
- Entangle / angle-encoding / re-upload depth curriculum excluded from fusion.
