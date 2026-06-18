"""Unit tests for model checkpoint persistence."""

import json

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.classical.mlp import ClassicalNet
from src.training.checkpoints import (
    checkpoint_path,
    load_checkpoint_bundle,
    load_scaler,
    save_best_checkpoint,
    save_checkpoint,
    save_scaler,
)


def test_save_checkpoint_writes_files(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path)
    model = ClassicalNet(hidden=8)
    directory = checkpoint_path("exp_test", "classical_8", seed=42)
    path = save_checkpoint(
        model,
        directory,
        config={"lr": 0.01},
        metadata={"test_accuracy": 0.8},
    )
    assert path.exists()
    assert (directory / "config.json").exists()
    payload = json.loads((directory / "config.json").read_text())
    assert payload["config"]["lr"] == 0.01
    assert payload["metadata"]["test_accuracy"] == 0.8


def test_save_best_checkpoint_only_on_improvement(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path)
    model = ClassicalNet(hidden=8)
    best, path = save_best_checkpoint(
        model,
        "exp_test",
        "classical_8",
        42,
        0.7,
        best_metric=None,
        config={"epochs": 5},
    )
    assert best == 0.7
    assert path is not None

    best2, path2 = save_best_checkpoint(
        model,
        "exp_test",
        "classical_8",
        42,
        0.6,
        best_metric=best,
        config={"epochs": 5},
    )
    assert best2 == 0.7
    assert path2 is None


def test_save_and_load_scaler_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path)
    directory = checkpoint_path("exp_test", "hybrid_sandwich_breast_cancer", seed=42)
    scaler = StandardScaler()
    scaler.fit(np.random.randn(50, 30).astype(np.float32))
    save_scaler(scaler, directory)
    loaded = load_scaler(directory)
    sample = np.random.randn(2, 30).astype(np.float32)
    np.testing.assert_allclose(loaded.transform(sample), scaler.transform(sample))


def test_load_checkpoint_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path)
    model = ClassicalNet(hidden=8, input_dim=30)
    directory = checkpoint_path("exp_ship", "hybrid_sandwich_breast_cancer", seed=7)
    save_checkpoint(
        model,
        directory,
        config={"model_name": "hybrid_sandwich", "dataset": "breast_cancer", "input_dim": 30},
        metadata={"holdout_accuracy": 0.9},
    )
    bundle = load_checkpoint_bundle("exp_ship", "hybrid_sandwich", "breast_cancer", seed=7)
    assert bundle.config["input_dim"] == 30
    restored = ClassicalNet(hidden=8, input_dim=30)
    restored.load_state_dict(bundle.state_dict)
    x = torch.randn(1, 30)
    with torch.no_grad():
        out = restored(x)
        assert out.shape == () or out.shape == (1,)
