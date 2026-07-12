# Results — EXP 090 Multi-crop joint ResidualNano

**Profile:** `publication`  
**Verdict:** rejected  
**Train:** soy=50,107 + maize=151,956  
**Val:** maize=13,566 · soy=5,830  
**Elapsed:** 22.7s

| Model | Val ROC-AUC | Notes |
|-------|-------------|-------|
| HistGB maize (honesty) | 0.8203 | 37-d native |
| Maize-solo ResidualNano | 0.8073 | 840,833 params |
| Joint ResidualNano (maize val) | 0.7938 | 840,833 params |
| Joint ResidualNano (soy val) | 0.6758 | secondary |

- Δ joint − solo = **-1.34 pp** (need ≥ -0.5)

## Interpretation

Joint multi-crop training hurt maize ranking beyond −0.5 pp — do not claim cross-crop climate transfer without crop-specific heads.

## Limitations

- Hard labels; crop indicator after StandardScaler.
- Single seed; existing temporal splits (no year in processed parquet).
