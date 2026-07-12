# Results — EXP 091: Circuit-cut effective 6q head (ACYD maize)

**Run date:** 2026-07-12  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)
**Profile:** `publication`

## Validation gate

- Train rows: **30,000** | Val rows: **13,566**
- Classical bottleneck head AUC: **0.8129** (trainable 65)
- Plain 4q re-upload AUC: **0.8128** (trainable 289)
- Circuit-cut 2×4q (effective 6q) AUC: **0.8125** (trainable 447)
- Cut vs classical: **-0.03 pp** (gate ≥ -1.0)
- Cut vs plain 4q: **-0.03 pp**
- Elapsed: **22.193s**

## Verdict
**accepted** — Phase B H-Q2.5 circuit-cut 6q effective head.

## Limitations
- Soft overlapping-fragment cut (not full tomography reconstruction).
- Frozen distill backbone; PennyLane TorchLayer on CPU.
- Hybrid fine-tune row budget capped for QNN wall-time on 4060.
- Agro research benchmark — not operational planting advice.
