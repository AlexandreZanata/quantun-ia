# Results — EXP 067: Re-upload climate feature-block ladder (ACYD / H-Q11)

**Run date:** 2026-07-12  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)
**Dataset:** `acyd_soy_brazil_v1`

## Validation gate (ROC-AUC)

- Wins: **2/3** (gate ≥ **2**)
- Per-rung advantage gate: **≥ 0.3 pp**

| Rung | Slice | Dim | Curriculum | Fixed | Δ pp | Won |
|------|-------|-----|------------|-------|------|-----|
| temp_only | [13:21] | 8 | 0.5660 | 0.5551 | +1.08 | yes |
| temp_precip | [9:21] | 12 | 0.5638 | 0.5713 | -0.75 | no |
| full_37 | [0:37] | 37 | 0.6402 | 0.5981 | +4.21 | yes |

- Elapsed: **59.766s**

## Verdict
**accepted** — re-upload depth curriculum vs fixed max depth on climate feature blocks.
