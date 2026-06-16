import numpy as np


def poison_dataset(X, y, poison_rate=0.1, seed=42):
    """
    Flip labels for `poison_rate` fraction of the dataset.
    Returns X_poisoned, y_poisoned, poison_mask (for analysis).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    n_poison = int(n * poison_rate)
    poison_idx = rng.choice(n, size=n_poison, replace=False)

    y_poisoned = y.copy()
    y_poisoned[poison_idx] = 1 - y_poisoned[poison_idx]

    poison_mask = np.zeros(n, dtype=bool)
    poison_mask[poison_idx] = True

    return X, y_poisoned, poison_mask


def measure_robustness(model_results: dict) -> dict:
    """
    Given a dict {poison_rate: accuracy}, compute:
    - drop per poison rate
    - degradation relative to clean baseline
    """
    rates = sorted(model_results.keys())
    base = model_results[0.0]
    results = {}
    for r in rates:
        drop = base - model_results[r]
        results[r] = {"accuracy": model_results[r], "drop": drop}
    return results
