"""Unit tests for ACYD Brazil open dataset builder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.open_acyd import (
    N_FEATURES,
    build_binary_labels_below_state_median,
    extract_acyd_feature_matrix,
    join_yield_features,
    temporal_year_split,
)


def _synthetic_yield() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "country": ["Brazil", "Brazil", "Brazil", "Brazil"],
            "admin_level_1": ["SP", "SP", "MG", "MG"],
            "admin_level_2": ["Campinas", "Ribeirao", "Uberaba", "Uberlandia"],
            "year": [2018, 2019, 2018, 2022],
            "soybean_yield": [3000.0, 2000.0, 3200.0, 2800.0],
            "area_harvested": [100.0, 80.0, 120.0, 90.0],
            "area_planted": [100.0, 80.0, 120.0, 90.0],
            "production": [300.0, 160.0, 384.0, 252.0],
        }
    )


def _synthetic_features() -> pd.DataFrame:
    row = {
        "country": "Brazil",
        "admin_level_1": "SP",
        "admin_level_2": "Campinas",
        "year": 2018,
        "latitude": -22.9,
        "longitude": -47.1,
        "organic_carbon_0_5cm": 1.2,
        "ph_h2o_0_5cm": 5.5,
        "clay_0_5cm": 30.0,
        "sand_0_5cm": 40.0,
        "cec_0_5cm": 10.0,
        "bulk_density_0_5cm": 1.3,
    }
    for week in range(1, 53):
        row[f"precipitation_week_{week}"] = float(week)
        row[f"t2m_min_week_{week}"] = 290.0 + week * 0.1
        row[f"t2m_max_week_{week}"] = 300.0 + week * 0.1
        row[f"solar_radiation_week_{week}"] = 1e7 + week
        row[f"lai_high_week_{week}"] = 0.5
        row[f"ndvi_week_{week}"] = 0.6
        row[f"vapor_pressure_deficit_week_{week}"] = 1.0
    rows = []
    for spec in _synthetic_yield().to_dict("records"):
        item = dict(row)
        item.update(
            {
                "admin_level_1": spec["admin_level_1"],
                "admin_level_2": spec["admin_level_2"],
                "year": spec["year"],
            }
        )
        rows.append(item)
    return pd.DataFrame(rows)


def test_binary_label_below_state_median():
    frame = _synthetic_yield()
    frame = pd.concat(
        [
            frame,
            pd.DataFrame(
                [
                    {
                        "country": "Brazil",
                        "admin_level_1": "SP",
                        "admin_level_2": "Sorocaba",
                        "year": 2019,
                        "soybean_yield": 3000.0,
                        "area_harvested": 50.0,
                        "area_planted": 50.0,
                        "production": 150.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    labels = build_binary_labels_below_state_median(frame)
    assert labels[1] == 1
    assert labels[4] == 0


def test_extract_acyd_feature_matrix_shape():
    merged = join_yield_features(_synthetic_yield(), _synthetic_features())
    matrix = extract_acyd_feature_matrix(merged)
    assert matrix.shape == (len(merged), N_FEATURES)
    assert matrix.dtype == np.float32
    assert not np.isnan(matrix).any()


def test_temporal_year_split_masks():
    years = np.array([2017, 2018, 2019, 2020, 2022])
    train, val, test = temporal_year_split(years)
    assert train.tolist() == [True, True, False, False, False]
    assert val.tolist() == [False, False, True, True, False]
    assert test.tolist() == [False, False, False, False, True]


def test_join_yield_features_inner():
    merged = join_yield_features(_synthetic_yield(), _synthetic_features())
    assert len(merged) == 4
    assert "latitude" in merged.columns
