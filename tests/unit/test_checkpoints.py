"""Unit tests for model checkpoint persistence."""

import json

from src.classical.mlp import ClassicalNet
from src.training.checkpoints import checkpoint_path, save_best_checkpoint, save_checkpoint


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
