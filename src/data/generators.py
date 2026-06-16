"""Synthetic dataset generators for experiments."""

import numpy as np
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.preprocessing import StandardScaler


def make_binary_classification(
    n_samples: int = 200,
    dataset: str = "moons",
    noise: float = 0.1,
    random_state: int = 42,
):
    if dataset == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset == "circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, random_state=random_state)
    else:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=random_state,
        )

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    y = y.astype(np.float32)
    return X, y, scaler
