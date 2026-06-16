"""Train/test splits — applied before preprocessing to avoid leakage."""

from sklearn.model_selection import train_test_split


def split_train_test(X, y, test_size: float = 0.3, random_state: int = 42):
    """Split data before any label poisoning or curriculum ordering."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
