# Open Datasets — quantun-ia (public)

This folder holds **reproducible, license-clear** datasets for large-scale nano model training (Phase L).

## Principles

- **Licenses must allow redistribution** (or we ship DVC pointers + download scripts only).
- **No PHI** — synthetic (Synthea) or non-clinical bootstrap (HIGGS) only in this phase.
- **Raw features only** — scaling happens after train/val/test split in training code (same as breast cancer).
- **Bulk artifacts** live under `*/processed/` and may be tracked with DVC — not committed as multi-GB blobs in git.

## Datasets

| ID | Path | Rows (target) | Features | License | Status |
|----|------|---------------|----------|---------|--------|
| `higgs_v1` | `higgs/processed/v1/` | 1,150,000 | 28 | CC0 (UCI) | `ready` (DVC: `processed/v1.dvc`) |
| `synthea_cv_risk_v1` | `synthea_cv_risk/processed/v1/` | 1,000,000 | 40 | MIT (Synthea) | `ready` (DVC: `processed/v1.dvc`) |
| `nihr_cv_synthetic_v1` | `nihr_cv_synthetic/processed/v1/` | 100,000 | 13 | CC0 (Zenodo) | `ready` (DVC: `processed/v1.dvc`) |
| `code_defects_gobug_v1` | `code_defects_gobug/processed/v1/` | ~39,000 | 23 | IEEE/go-bug-collector | `ready` (DVC: `processed/v1.dvc`) |
| `acyd_soy_brazil_v1` | `acyd_soy_brazil/processed/v1/` | TBD | 37 | CC BY stack (ACYD) | `ready=false` until built |

See `manifest.json` for checksums and split counts once built.

## Schema

All Phase L tabular sets follow `schemas/tabular_binary_v1.json`:

- `feature_0 … feature_{N-1}` — float32, no NaN after export
- `label` — int `{0, 1}`

## Reproduce

```bash
# Bootstrap (Tier A)
make data-open-higgs

# L2 gate — manifest + checksums + schema + DVC pointer
make data-open-verify

# Clinical-aligned (Tier B)
make data-open-synthea-cv
make data-open-verify

# NIHR synthetic CV (realistic prevalence — exp_044)
make data-open-nihr-cv
make data-open-verify

# ACYD Brazil soybean (agro-climate — exp_060)
make data-open-acyd-soy
make data-open-verify DATASET=acyd_soy_brazil_v1
```

## Citations

- **HIGGS:** Baldi, P., Sadowski, P., Whiteson, D. (2014). Searching for Exotic Particles in High-Energy Physics with Deep Learning. *Nature Communications*. UCI ML Repository.
- **Synthea:** Walonoski, J. et al. Synthea: An approach, method, and software mechanism for generating synthetic patients and the standard medical data of a comprehensive lifetime journey. *JAMIA* (2018).
