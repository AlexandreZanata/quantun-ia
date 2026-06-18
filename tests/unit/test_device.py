"""Unit tests for training device resolution."""

import torch

from src.training.device import model_requires_cpu, resolve_device


def test_resolve_device_cpu_explicit(monkeypatch):
    monkeypatch.delenv("QML_DEVICE", raising=False)
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_auto_returns_cpu_without_cuda(monkeypatch):
    monkeypatch.setenv("QML_DEVICE", "auto")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device().type == "cpu"


def test_model_requires_cpu_for_hybrid_sandwich():
    from src.quantum.hybrid_model import HybridSandwich

    model = HybridSandwich(input_dim=8, n_qubits=4, n_layers=2)
    assert model_requires_cpu(model) is True


def test_resolve_device_auto_uses_cpu_for_quantum_model(monkeypatch):
    from src.quantum.hybrid_model import HybridSandwich

    monkeypatch.setenv("QML_DEVICE", "auto")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = HybridSandwich(input_dim=8, n_qubits=4, n_layers=2)
    assert resolve_device(model=model).type == "cpu"
