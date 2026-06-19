# Results — EXP 075: GV-ALR on frozen hybrid QNN head (NIHR C2 replication)

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate (PR-AUC)

- Fixed LR: **8** epochs · val PR-AUC **0.2392** · **15.735s**
- GV-ALR: **5** epochs · val PR-AUC **0.2369** · **9.091s**
- Δ PR-AUC: **-0.24 pp**
- Epoch fraction: **5/8**
- Wall-time ratio: **0.58**

## Verdict
**accepted** — |Δ PR-AUC| ≤ 0.3 pp and adaptive epochs ≤ 70% of fixed.

## Limitations
- PennyLane QNN sim on CPU; frozen C2 backbone from exp_069.
- Val PR-AUC only; efficiency gate mirrors exp_054.
