"""Unit tests for QNN factory."""

from src.quantum.qnn_basic import QuantumNetBasic
from src.quantum.qnn_factory import build_qnn
from src.quantum.qnn_reupload import QuantumNetReupload


def test_build_qnn_basic_default():
    model = build_qnn({"n_qubits": 4, "n_layers": 1})
    assert isinstance(model, QuantumNetBasic)


def test_build_qnn_reupload_from_config():
    model = build_qnn({"qnn_type": "reupload", "n_qubits": 4, "n_layers": 3})
    assert isinstance(model, QuantumNetReupload)
