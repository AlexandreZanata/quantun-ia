"""Unit tests for PennyLane device resolution."""

from __future__ import annotations

import pytest

from src.quantum.pennylane_device import DEFAULT_QML_DEVICE, QmlDeviceError, resolve_qml_device
from src.quantum.qnn_basic import QuantumNetBasic, make_qnn_basic


def test_resolve_qml_device_default():
    dev = resolve_qml_device(4, DEFAULT_QML_DEVICE)
    assert len(dev.wires) == 4


def test_make_qnn_basic_accepts_custom_device():
    layer = make_qnn_basic(4, 2, qml_device=DEFAULT_QML_DEVICE)
    assert layer is not None


def test_quantum_net_basic_custom_device_forward():
    import torch

    model = QuantumNetBasic(n_qubits=4, n_layers=2, input_dim=4, qml_device=DEFAULT_QML_DEVICE)
    x = torch.randn(3, 4)
    out = model(x)
    assert out.shape == (3,)


def test_resolve_qml_device_invalid_raises():
    with pytest.raises(QmlDeviceError, match="unavailable"):
        resolve_qml_device(4, "not.a.real.pennylane.device")
