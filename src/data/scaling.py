"""Feature scaling helpers — fit on train only to prevent leakage."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def fit_standard_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_with_scaler(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(X).astype(np.float32)


def scale_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = fit_standard_scaler(X_train)
    return transform_with_scaler(scaler, X_train), transform_with_scaler(scaler, X_test), scaler


def fit_pca(X_train: np.ndarray, n_components: int, random_state: int = 42) -> PCA:
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_train)
    return pca


def transform_with_pca(pca: PCA, X: np.ndarray) -> np.ndarray:
    return pca.transform(X).astype(np.float32)


def pca_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    pca = fit_pca(X_train, n_components, random_state=random_state)
    return transform_with_pca(pca, X_train), transform_with_pca(pca, X_test), pca
