"""Shared test fixtures."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def prefer_local_cuda():
    """Use NVIDIA GPU for classical training when available (local RTX 4060)."""
    if torch.cuda.is_available() and os.environ.get("QML_DEVICE") is None:
        os.environ["QML_DEVICE"] = "cuda"


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
