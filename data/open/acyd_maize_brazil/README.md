# ACYD Brazil Maize — quantun-ia (C4b agro-climate)

Open agro-climatic tabular dataset built from [ACYD](https://huggingface.co/datasets/notadib/ACYD)
(HuggingFace) corn yield table for LargeNanoMLP training (exp_081 / anchor C4b).

## License stack

| Component | License |
|-----------|---------|
| ACYD pipeline | MIT ([GitHub](https://github.com/Neehan/amazing-crop-yield-datasets)) |
| AgERA5 / ERA5 | Copernicus CC BY 4.0 |
| IBGE yields | CC BY 3.0 |
| HYDE cropland | CC BY 4.0 |

Verify redistribution before publishing processed parquet outside this repo.

## Label

**`below_state_median`:** `label=1` when municipal `corn_yield` is below the
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
make data-open-acyd-maize
make data-open-verify DATASET=acyd_maize_brazil_v1

# Manual equivalent (reuses soy climate feature chunks when present)
.venv/bin/python scripts/download_acyd_brazil.py --crop maize
.venv/bin/python scripts/build_open_acyd_maize.py
```

## Citation

ACYD dataset and processing pipeline — see HuggingFace model card `notadib/ACYD`.
