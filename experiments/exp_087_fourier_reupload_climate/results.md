# Results — EXP 087: Fourier re-upload vs flat angle head (ACYD maize)

**Run date:** 2026-07-12  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)
**Profile:** `publication`

## Validation gate

- Train rows: **30,000** | Val rows: **13,566**
- Classical distill ResidualNano AUC: **0.8130**
- Rung wins (Fourier > flat): **0/3** (gate ≥ 2)
- Deepest flat Δ classical: **-0.03 pp**
- Deepest Fourier Δ classical: **-0.24 pp**
- Parity floor: ≥ classical − 1.0 pp
- Trainable (deepest flat / Fourier): 301 / 369
- Elapsed: **42.603s**

## Per-rung AUC

| Layers | Flat | Fourier | Δ pp | Result |
|--------|------|---------|------|--------|
| 1 | 0.8127 | 0.8121 | -0.07 | lose |
| 2 | 0.8128 | 0.8106 | -0.22 | lose |
| 3 | 0.8128 | 0.8107 | -0.21 | lose |

## Verdict
**rejected** — Phase B H-Q2.2 Fourier climate re-upload.

## Limitations
- Frozen distill backbone; PennyLane TorchLayer on CPU.
- Hybrid fine-tune row budget capped for QNN wall-time on 4060.
- Agro research benchmark — not operational planting advice.
