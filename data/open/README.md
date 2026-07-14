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
| `acyd_soy_brazil_v1` | `acyd_soy_brazil/processed/v1/` | ~62K | 37 | CC BY stack (ACYD) | `ready` |
| `acyd_maize_brazil_v1` | `acyd_maize_brazil/processed/v1/` | ~179K | 37 | CC BY stack (ACYD) | `ready` |

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

# ACYD Brazil maize (agro-climate — exp_081 / C4b)
make data-open-acyd-maize
make data-open-verify DATASET=acyd_maize_brazil_v1
```

## Multi-crop joint protocol (Phase C / exp_090)

Soy and maize share the same **37** climate/soil features and temporal cutoffs
(train ≤ 2018, val 2019–2021, test ≥ 2022). For joint training:

1. Concatenate train rows from `acyd_soy_brazil_v1` + `acyd_maize_brazil_v1`.
2. Fit `StandardScaler` on the **joint** climate features only (no year leakage; years are not in processed parquet).
3. Append a crop indicator after scaling (`0=soy`, `1=maize`) → **38-d** input.
4. Evaluate maize-only and soy-only temporal val separately.

Loader: `src/data/multicrop_acyd.py`.

## CY-Bench maize sample (Phase C / exp_095)

External panel from AgML **CY-Bench** (EUPL-1.2). Full Zenodo archive is large
(~6 GB); this lab uses the official `sample_data` US designed-feature tables.

```bash
make data-open-cybench-maize
make data-open-verify DATASET=cybench_maize_us_v1
```

- Dataset ID: `cybench_maize_us_v1`
- Features: AgML designed climate/RS/soil columns (excludes `adm_id`, `year`,
  `yield`, and yield-lag / `yield_trend` columns to avoid label leakage)
- Label: low-yield binary (yield ≤ train-period median, years ≤ 2011)
- Splits: train ≤ 2011 · val 2012–2015 · test ≥ 2016

## Cycle v3 — open image packs (Phase G / exp_101+)

Nano I2I / T2I training uses **license-clear** image corpora under `images/`.

| Pack | Path | Role | Priority | Status |
|------|------|------|----------|--------|
| `cifar10` | `images/cifar10/` | I2I / class-cond FID floor | P0 | ✅ ready (`cifar10_v1`) |
| `fashion_mnist` | `images/fashion_mnist/` | CI smoke | P0 | ✅ ready (`fashion_mnist_v1`) |
| `flowers102` | `images/flowers102/` | fine-detail class-cond | P0 | ✅ ready (`flowers102_v1`) |
| `flickr8k` | `images/flickr8k/` | T2I captions (G-T3) | P0 | ✅ ready (`flickr8k_captions_v1`) |
| pokemon-blip | n/a | was P1 toy | — | ❌ gated/DMCA — skipped |
| LAION-Aesthetic micro ≤50k | TBD | optional hard T2I | P2 gated | G-T7 |

```bash
make data-open-images-smoke
make data-open-images-splits
make data-open-images-captions
make data-open-caption-splits
make exp-101-publication
make exp-103-publication
# writes GENERATION.md + processed/*/stats.json (+ pairs.parquet for captions)
```

**Rules:** split train/val/test **before** resize/normalize; document licenses in `images/GENERATION.md`; do not commit multi-GB raw blobs — gitignore `images/*/raw/`.

## Citations

- **HIGGS:** Baldi, P., Sadowski, P., Whiteson, D. (2014). Searching for Exotic Particles in High-Energy Physics with Deep Learning. *Nature Communications*. UCI ML Repository.
- **Synthea:** Walonoski, J. et al. Synthea: An approach, method, and software mechanism for generating synthetic patients and the standard medical data of a comprehensive lifetime journey. *JAMIA* (2018).
- **CY-Bench:** Kallenberg et al. (2026). CY-Bench: a comprehensive benchmark dataset for sub-national crop yield forecasting. *ESSD* / Zenodo DOI [10.5281/zenodo.11502142](https://doi.org/10.5281/zenodo.11502142). Sample tables via [WUR-AI/sample_data](https://github.com/WUR-AI/sample_data).
- **CIFAR-10:** Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images.
- **Fashion-MNIST:** Xiao, H., Rasul, K., Vollgraf, R. (2017). Fashion-MNIST.
- **Oxford Flowers-102:** Nilsback, M., Zisserman, A. (2008). Automated flower classification over a large number of classes.
- **Flickr8k:** Hodosh, M., Young, P., Hockenmaier, J. (2013). Framing image description as a ranking task. *JAIR*.
