"""Unit tests for quantum warm-start schedule."""

import torch

from src.quantum.hybrid_model import HybridSandwich
from src.training.quantum_warmstart import (
    WarmStartConfig,
    split_warmstart_epochs,
    train_hybrid_warmstart,
)


def test_split_warmstart_epochs_sums_to_total():
    classical, quantum = split_warmstart_epochs(10, 0.7)
    assert classical + quantum == 10
    assert classical == 7
    assert quantum == 3


def test_split_warmstart_epochs_minimum_one_each():
    classical, quantum = split_warmstart_epochs(3, 0.9)
    assert classical + quantum == 3
    assert classical >= 1
    assert quantum >= 1


def test_warmstart_config_rejects_invalid_fraction():
    try:
        WarmStartConfig(classical_fraction=1.0, total_epochs=10)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_hybrid_sandwich_bypasses_quantum_when_disabled():
    model = HybridSandwich(input_dim=4, n_qubits=2, n_layers=1, reupload=False)
    x = torch.randn(3, 4)
    model.set_quantum_enabled(True)
    out_q = model(x)
    model.set_quantum_enabled(False)
    out_c = model(x)
    assert out_q.shape == (3,)
    assert out_c.shape == (3,)


def test_train_hybrid_warmstart_runs_two_phases(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")
    model = HybridSandwich(input_dim=4, n_qubits=2, n_layers=1, reupload=False)
    x = torch.randn(32, 4)
    y = torch.randint(0, 2, (32,)).float()
    config = WarmStartConfig(classical_fraction=0.7, total_epochs=4)
    classical, quantum = train_hybrid_warmstart(
        model,
        x,
        y,
        "exp_test",
        "warmstart_smoke",
        config=config,
        lr=0.05,
        batch_size=16,
        seed=42,
    )
    assert classical + quantum == 4
    assert model._quantum_enabled is True
