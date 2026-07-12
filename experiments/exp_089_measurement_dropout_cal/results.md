# Results — EXP 089: Measurement-dropout QNN calibration (ACYD maize)

**Run date:** 2026-07-12  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)
**Profile:** `publication`

## Validation gate

- Train rows: **30,000** | Val rows: **13,566**
- Classical distill ResidualNano AUC: **0.8130**
- Plain QNN: AUC **0.8125** | ECE **0.0242** | trainable 289
- Measurement-dropout QNN (p=0.2, MC=16): AUC **0.8117** | ECE **0.0374** | trainable 289
- ECE relative improvement: **-54.8%** (gate ≥ 20%)
- AUC Δ (dropout − plain): **-0.08 pp** (gate ≥ -0.5)
- Elapsed: **15.532s**

## Verdict
**rejected** — Phase B H-Q2.4 measurement-dropout calibration.

## Limitations
- Frozen distill backbone; PennyLane TorchLayer on CPU.
- Hybrid fine-tune row budget capped for QNN wall-time on 4060.
- Agro research benchmark — not operational planting advice.
