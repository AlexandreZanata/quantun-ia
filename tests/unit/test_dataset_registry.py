"""Unit tests for dataset registry."""

from src.data.dataset_registry import get_dataset, prepare_dataset


def test_get_dataset_synthetic_circles():
    X, y, meta = get_dataset("circles", random_state=7, n_samples=80, noise=0.2)
    assert X.shape == (80, 2)
    assert meta["name"] == "circles"


def test_prepare_dataset_wine_binary():
    X_train, X_test, y_train, y_test, meta = prepare_dataset("wine_binary", random_state=1)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert meta["scaled"] is True
