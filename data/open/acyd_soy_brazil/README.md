# ACYD Brazil Soybean — quantun-ia (Phase L agro-climate)

Open agro-climatic tabular dataset built from [ACYD](https://huggingface.co/datasets/notadib/ACYD)
(HuggingFace) for LargeNanoMLP training (exp_060 / anchor C4).

## License stack

| Component | License |
|-----------|---------|
| ACYD pipeline | MIT ([GitHub](https://github.com/Neehan/amazing-crop-yield-datasets)) |
| AgERA5 / ERA5 | Copernicus CC BY 4.0 |
| IBGE yields | CC BY 3.0 |
| HYDE cropland | CC BY 4.0 |

Verify redistribution before publishing processed parquet outside this repo.

## Label

**`below_state_median`:** `label=1` when municipal `soybean_yield` is below the
**state-year median** (same `admin_level_1` + `year`).

## Split

**Temporal by crop year** (no shuffle):

| Split | Years |
|-------|-------|
| train | ≤ 2018 |
| val | 2019–2021 |
| test | ≥ 2022 |

## Reproduce

```bash
# Recommended (uses .venv/bin/python)
make data-open-acyd-soy
make data-open-acyd-dvc
make data-open-verify DATASET=acyd_soy_brazil_v1

# Manual equivalent
.venv/bin/python scripts/download_acyd_brazil.py --crop soybean
.venv/bin/python scripts/build_open_acyd_soy.py

# Smoke (1 chunk only — not for publication)
.venv/bin/python scripts/download_acyd_brazil.py --max-feature-chunks 1
.venv/bin/python scripts/build_open_acyd_soy.py --max-feature-chunks 1 --skip-manifest
```

Use `make data-open-acyd-soy` or `.venv/bin/python scripts/...` — plain `python` may fail
with `ModuleNotFoundError: src` unless the project is installed (`pip install -e .`).

## Citation

ACYD dataset and processing pipeline — see HuggingFace model card `notadib/ACYD`.
