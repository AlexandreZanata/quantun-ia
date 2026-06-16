"""Shared test fixtures."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_binary_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 2)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    return X, y


@pytest.fixture
def temp_log_file(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "experiments.jsonl"
        monkeypatch.setattr("src.training.metrics.LOGS_PATH", log_path)
        yield log_path
