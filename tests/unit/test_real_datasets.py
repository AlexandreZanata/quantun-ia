"""Unit tests for real-world dataset loaders."""

import numpy as np

from src.data.dataset_registry import get_dataset, prepare_dataset
from src.data.real_datasets import load_breast_cancer_raw
from src.data.scaling import scale_train_test


def test_load_breast_cancer_binary():
    X, y, meta = load_breast_cancer_raw()
    assert X.shape[1] == 30
    assert set(np.unique(y)) <= {0.0, 1.0}
    assert meta["name"] == "breast_cancer"


def test_get_dataset_registry_breast_cancer():
    X, y, meta = get_dataset("breast_cancer")
    assert X.ndim == 2
    assert meta["source"] == "sklearn"


def test_load_pima_diabetes_raw(monkeypatch):
    import src.data.real_datasets as rd

    fake_x = np.random.default_rng(0).normal(size=(768, 8)).astype(np.float32)
    fake_y = np.zeros(768, dtype=np.float32)
    fake_y[::2] = 1.0

    def _fake_loader():
        return fake_x, fake_y, {"name": "pima_diabetes", "n_features": 8, "source": "openml:37"}

    monkeypatch.setattr(rd, "load_pima_diabetes_raw", _fake_loader)
    X, y, meta = get_dataset("pima_diabetes")
    assert X.shape == (768, 8)
    assert meta["name"] == "pima_diabetes"


def test_prepare_pima_diabetes_splits_and_scales(monkeypatch):
    import src.data.real_datasets as rd

    rng = np.random.default_rng(42)
    fake_x = rng.normal(size=(768, 8)).astype(np.float32)
    fake_y = (fake_x[:, 0] > 0).astype(np.float32)
    monkeypatch.setattr(
        rd,
        "load_pima_diabetes_raw",
        lambda: (fake_x, fake_y, {"name": "pima_diabetes", "n_features": 8, "source": "openml:37"}),
    )
    X_train, X_test, y_train, y_test, meta = prepare_dataset(
        "pima_diabetes",
        random_state=42,
        test_size=0.3,
    )
    assert len(X_train) + len(X_test) == 768
    assert meta["scaled"] is True


def test_scaler_fit_on_train_only_no_leakage():
    X, y, _ = load_breast_cancer_raw()
    X_train, X_test = X[:400], X[400:]
    X_train_s, X_test_s, scaler = scale_train_test(X_train, X_test)

    assert X_train_s.shape == X_train.shape
    assert np.isclose(X_train_s.mean(axis=0), 0, atol=1e-5).all()

    leaked = scaler.transform(X_test).astype(np.float32)
    np.testing.assert_allclose(X_test_s, leaked)


def test_prepare_dataset_splits_and_scales():
    X_train, X_test, y_train, y_test, meta = prepare_dataset(
        "breast_cancer",
        random_state=42,
        test_size=0.3,
    )
    assert len(X_train) + len(X_test) == len(load_breast_cancer_raw()[0])
    assert meta["scaled"] is True
    assert set(np.unique(y_train)) <= {0.0, 1.0}
