"""Unit tests for hybrid NAS helpers."""

from src.quantum.hybrid_model import ClassicalFirst, HybridSandwich, QuantumFirst
from src.training.hpo import build_hybrid_from_params


def test_build_hybrid_from_params_sandwich():
    model, lr = build_hybrid_from_params(
        {
            "architecture": "hybrid_sandwich",
            "n_qubits": 4,
            "n_layers": 2,
            "learning_rate": 0.03,
            "reupload": True,
        }
    )
    assert isinstance(model, HybridSandwich)
    assert lr == 0.03


def test_build_hybrid_from_params_quantum_first():
    model, lr = build_hybrid_from_params(
        {"architecture": "quantum_first", "n_qubits": 3, "n_layers": 1, "learning_rate": 0.01}
    )
    assert isinstance(model, QuantumFirst)
    assert lr == 0.01


def test_build_hybrid_from_params_classical_first_no_reupload():
    model, _ = build_hybrid_from_params(
        {"architecture": "classical_first", "n_qubits": 4, "n_layers": 2, "reupload": False}
    )
    assert isinstance(model, ClassicalFirst)
