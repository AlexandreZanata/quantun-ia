# Results — EXP 086: Residual-skip QNN vs plain QNN (ACYD maize)

**Run date:** 2026-07-12  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)
**Profile:** `publication`

## Validation gate

- Train rows: **50,000** | Val rows: **13,566**
- Classical distill ResidualNano AUC: **0.8129**
- Plain QNN AUC: **0.8120** (Δ classical -0.09 pp | trainable 289)
- Residual-skip QNN AUC: **0.8127** (Δ classical -0.02 pp | trainable 354)
- Residual vs plain: **0.07 pp** (gate ≥ 0.5)
- Parity floor: ≥ classical − 1.0 pp
- Elapsed: **32.185s**

## Verdict
**rejected** — Phase B H-Q2.1 residual QNN skip.

## Limitations
- Frozen distill backbone; PennyLane TorchLayer on CPU.
- Hybrid fine-tune row budget may be capped for QNN wall-time on 4060.
- Agro research benchmark — not operational planting advice.
