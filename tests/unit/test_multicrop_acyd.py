"""Unit tests for multi-crop ACYD joint loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.multicrop_acyd import (
    CROP_MAIZE,
    CROP_SOY,
    N_CLIMATE_FEATURES,
    load_multicrop_acyd_splits,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    not (ROOT / "data/open/acyd_maize_brazil/processed/v1/train.parquet").is_file(),
    reason="ACYD maize processed missing",
)
@pytest.mark.skipif(
    not (ROOT / "data/open/acyd_soy_brazil/processed/v1/train.parquet").is_file(),
    reason="ACYD soy processed missing",
)
def test_multicrop_loader_shapes_and_crop_bit():
    splits = load_multicrop_acyd_splits(
        ROOT,
        n_train_rows_maize=40,
        n_train_rows_soy=40,
        n_val_rows_maize=20,
        n_val_rows_soy=20,
        random_state=0,
        include_crop_indicator=True,
    )
    assert splits.x_train.shape[1] == N_CLIMATE_FEATURES + 1
    assert splits.x_val_maize.shape[1] == N_CLIMATE_FEATURES + 1
    assert splits.n_soy_train == 40
    assert splits.n_maize_train == 40
    assert set(np.unique(splits.crop_train).tolist()) == {CROP_SOY, CROP_MAIZE}
    # Crop bit after scaling stays in {0,1}
    assert set(np.unique(splits.x_val_maize[:, -1]).tolist()) == {CROP_MAIZE}
    assert set(np.unique(splits.x_val_soy[:, -1]).tolist()) == {CROP_SOY}


def test_multicrop_without_crop_indicator_dim():
    if not (ROOT / "data/open/acyd_maize_brazil/processed/v1/train.parquet").is_file():
        pytest.skip("ACYD missing")
    splits = load_multicrop_acyd_splits(
        ROOT,
        n_train_rows_maize=30,
        n_train_rows_soy=30,
        n_val_rows_maize=15,
        n_val_rows_soy=15,
        include_crop_indicator=False,
    )
    assert splits.x_train.shape[1] == N_CLIMATE_FEATURES
