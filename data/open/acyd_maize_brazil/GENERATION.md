# Generation rules — acyd_maize_brazil_v1

**Hypothesis for data:** Municipality-level maize (corn) yield in Brazil is predictable from
pre-season and in-season climate/soil covariates; temporal split prevents future-year leakage.

## Join keys

`country`, `admin_level_1` (state), `admin_level_2` (municipality), `year`

## Yield source

ACYD HuggingFace file `brazil/final/crop/crop_corn_yield.csv` (`corn_yield` column).
Dataset id uses **maize** naming for product consistency (`acyd_maize_brazil_v1`).

## Features (37)

| Index | Description |
|-------|-------------|
| 0–1 | latitude, longitude |
| 2 | log1p(area_harvested) |
| 3–8 | SoilGrids surface (0–5 cm): organic C, pH, clay, sand, CEC, bulk density |
| 9–36 | Season weeks 10–40 aggregates (mean, std, min, max) × 7 weather variables |

Climate feature chunks are shared with soybean (municipality×year panel).

## Exclusions

- No municipality ID in features (only lat/lon)
- No yield, production, or area_planted as direct features (area_harvested only as log scale)
- Rows with NaN/Inf dropped after feature extraction

## Builder

`scripts/build_open_acyd_maize.py` → `src/data/open_acyd.py`
