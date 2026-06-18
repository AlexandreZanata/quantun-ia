"""Load Phase L open parquet splits with train-only scaling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.open_manifest import get_dataset, load_manifest


def load_open_parquet_splits(
    dataset_id: str,
    root: Path,
    *,
    n_train_rows: int | None = None,
    n_val_rows: int | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Load train/val/test arrays from an open dataset with scaler fit on train only."""
    manifest = load_manifest(root / "data" / "open" / "manifest.json")
    dataset = get_dataset(manifest, dataset_id)
    if not dataset.get("ready"):
        msg = f"{dataset_id} is not ready in manifest.json"
        raise ValueError(msg)

    processed = root / "data" / "open" / dataset["path"]
    feature_cols = [f"feature_{i}" for i in range(int(dataset["n_features"]))]

    def _load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
        frame = pd.read_parquet(processed / dataset["files"][name])
        features = frame[feature_cols].to_numpy(dtype=np.float32)
        labels = frame["label"].to_numpy(dtype=np.float32)
        return features, labels

    x_train, y_train = _load_split("train")
    x_val, y_val = _load_split("val")
    x_test, y_test = _load_split("test")

    if n_train_rows is not None and n_train_rows < len(y_train):
        idx = np.arange(len(y_train))
        selected, _ = train_test_split(
            idx,
            train_size=n_train_rows,
            stratify=y_train,
            random_state=random_state,
        )
        x_train, y_train = x_train[selected], y_train[selected]

    if n_val_rows is not None and n_val_rows < len(y_val):
        idx = np.arange(len(y_val))
        selected, _ = train_test_split(
            idx,
            train_size=n_val_rows,
            stratify=y_val,
            random_state=random_state,
        )
        x_val, y_val = x_val[selected], y_val[selected]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    return x_train, y_train, x_val, y_val, x_test, y_test, scaler
