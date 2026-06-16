"""Factory for building QNN models from experiment config."""

from __future__ import annotations

from src.quantum.qnn_basic import QuantumNetBasic
from src.quantum.qnn_reupload import QuantumNetReupload


def build_qnn(cfg: dict, *, input_dim: int = 2):
    """Build a QNN from shared config keys: qnn_type, n_qubits, n_layers."""
    qnn_type = cfg.get("qnn_type", "basic")
    n_qubits = cfg.get("n_qubits", 4)
    n_layers = cfg.get("n_layers", 1)

    if qnn_type == "reupload":
        return QuantumNetReupload(n_qubits=n_qubits, n_layers=n_layers, input_dim=input_dim)
    return QuantumNetBasic(n_qubits=n_qubits, n_layers=n_layers, input_dim=input_dim)
