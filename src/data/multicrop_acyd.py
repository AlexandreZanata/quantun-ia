"""Multi-crop ACYD joint splits — soy + maize shared 37-d features + crop indicator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.open_manifest import get_dataset, load_manifest
from src.data.open_parquet import load_open_parquet_splits

SOY_DATASET_ID = "acyd_soy_brazil_v1"
MAIZE_DATASET_ID = "acyd_maize_brazil_v1"
CROP_SOY = 0.0
CROP_MAIZE = 1.0
N_CLIMATE_FEATURES = 37


@dataclass(frozen=True)
class MulticropAcydSplits:
    """Joint train arrays plus crop-specific val/test for evaluation."""

    x_train: np.ndarray
    y_train: np.ndarray
    crop_train: np.ndarray
    x_val_maize: np.ndarray
    y_val_maize: np.ndarray
    x_val_soy: np.ndarray
    y_val_soy: np.ndarray
    x_train_maize_solo: np.ndarray
    y_train_maize_solo: np.ndarray
    scaler: StandardScaler
    n_soy_train: int
    n_maize_train: int


def _cap_rows(
    x: np.ndarray,
    y: np.ndarray,
    n_rows: int | None,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_rows is None or n_rows <= 0 or n_rows >= len(y):
        return x, y
    idx = np.arange(len(y))
    selected, _ = train_test_split(
        idx,
        train_size=int(n_rows),
        stratify=y,
        random_state=random_state,
    )
    return x[selected], y[selected]


def _append_crop_indicator(x_scaled: np.ndarray, crop_value: float) -> np.ndarray:
    indicator = np.full((x_scaled.shape[0], 1), crop_value, dtype=np.float32)
    return np.concatenate([x_scaled, indicator], axis=1).astype(np.float32)


def load_multicrop_acyd_splits(
    root: Path,
    *,
    n_train_rows_maize: int | None = None,
    n_train_rows_soy: int | None = None,
    n_val_rows_maize: int | None = None,
    n_val_rows_soy: int | None = None,
    random_state: int = 42,
    include_crop_indicator: bool = True,
) -> MulticropAcydSplits:
    """Load soy+maize, scale climate features on joint train, append crop bit after scale.

    Crop indicator is **not** passed through ``StandardScaler`` (stays in {0, 1}).
    Maize-solo train uses the same joint climate scaler for a fair head-to-head.
    """
    manifest = load_manifest(root / "data" / "open" / "manifest.json")

    def _raw(dataset_id: str, split: str) -> tuple[np.ndarray, np.ndarray]:
        dataset = get_dataset(manifest, dataset_id)
        if not dataset.get("ready"):
            msg = f"{dataset_id} is not ready"
            raise ValueError(msg)
        processed = root / "data" / "open" / dataset["path"]
        frame = pd.read_parquet(processed / dataset["files"][split])
        n_features = int(dataset["n_features"])
        if n_features != N_CLIMATE_FEATURES:
            msg = f"{dataset_id} expected {N_CLIMATE_FEATURES} features, got {n_features}"
            raise ValueError(msg)
        cols = [f"feature_{i}" for i in range(n_features)]
        return (
            frame[cols].to_numpy(dtype=np.float32),
            frame["label"].to_numpy(dtype=np.float32),
        )

    x_soy_tr, y_soy_tr = _raw(SOY_DATASET_ID, "train")
    x_soy_va, y_soy_va = _raw(SOY_DATASET_ID, "val")
    x_mz_tr, y_mz_tr = _raw(MAIZE_DATASET_ID, "train")
    x_mz_va, y_mz_va = _raw(MAIZE_DATASET_ID, "val")

    x_soy_tr, y_soy_tr = _cap_rows(x_soy_tr, y_soy_tr, n_train_rows_soy, random_state=random_state)
    x_mz_tr, y_mz_tr = _cap_rows(
        x_mz_tr, y_mz_tr, n_train_rows_maize, random_state=random_state + 1
    )
    x_soy_va, y_soy_va = _cap_rows(
        x_soy_va, y_soy_va, n_val_rows_soy, random_state=random_state + 2
    )
    x_mz_va, y_mz_va = _cap_rows(
        x_mz_va, y_mz_va, n_val_rows_maize, random_state=random_state + 3
    )

    x_joint_raw = np.concatenate([x_soy_tr, x_mz_tr], axis=0)
    y_joint = np.concatenate([y_soy_tr, y_mz_tr], axis=0)
    crop_train = np.concatenate(
        [
            np.full(len(y_soy_tr), CROP_SOY, dtype=np.float32),
            np.full(len(y_mz_tr), CROP_MAIZE, dtype=np.float32),
        ],
        axis=0,
    )

    scaler = StandardScaler()
    x_joint_scaled = scaler.fit_transform(x_joint_raw).astype(np.float32)
    x_mz_tr_scaled = scaler.transform(x_mz_tr).astype(np.float32)
    x_mz_va_scaled = scaler.transform(x_mz_va).astype(np.float32)
    x_soy_va_scaled = scaler.transform(x_soy_va).astype(np.float32)

    if include_crop_indicator:
        # Rebuild joint with per-crop indicators matching concatenation order
        x_soy_part = _append_crop_indicator(
            scaler.transform(x_soy_tr).astype(np.float32), CROP_SOY
        )
        x_mz_part = _append_crop_indicator(x_mz_tr_scaled, CROP_MAIZE)
        x_train = np.concatenate([x_soy_part, x_mz_part], axis=0)
        x_val_maize = _append_crop_indicator(x_mz_va_scaled, CROP_MAIZE)
        x_val_soy = _append_crop_indicator(x_soy_va_scaled, CROP_SOY)
        x_train_maize_solo = _append_crop_indicator(x_mz_tr_scaled, CROP_MAIZE)
    else:
        x_train = x_joint_scaled
        x_val_maize = x_mz_va_scaled
        x_val_soy = x_soy_va_scaled
        x_train_maize_solo = x_mz_tr_scaled

    return MulticropAcydSplits(
        x_train=x_train,
        y_train=y_joint,
        crop_train=crop_train,
        x_val_maize=x_val_maize,
        y_val_maize=y_mz_va,
        x_val_soy=x_val_soy,
        y_val_soy=y_soy_va,
        x_train_maize_solo=x_train_maize_solo,
        y_train_maize_solo=y_mz_tr,
        scaler=scaler,
        n_soy_train=len(y_soy_tr),
        n_maize_train=len(y_mz_tr),
    )


def maize_solo_open_splits(
    root: Path,
    *,
    n_train_rows: int | None = None,
    n_val_rows: int | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convenience: standard maize open splits (37-d) for HistGB honesty floor."""
    x_tr, y_tr, x_va, y_va, _xt, _yt, _sc = load_open_parquet_splits(
        MAIZE_DATASET_ID,
        root,
        n_train_rows=n_train_rows,
        n_val_rows=n_val_rows,
        random_state=random_state,
    )
    return x_tr, y_tr, x_va, y_va
