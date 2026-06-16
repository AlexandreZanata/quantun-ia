"""Unit tests for gradient variance measurement."""

import torch

from src.quantum.qnn_basic import QuantumNetBasic
from src.training.gradients import measure_gradient_variance


def test_measure_gradient_variance_finite():
    def factory(n_qubits):
        return QuantumNetBasic(n_qubits=n_qubits, n_layers=1, input_dim=2)

    variances = measure_gradient_variance(factory, [2, 4], n_samples=3)
    assert 2 in variances and 4 in variances
    for v in variances.values():
        assert not torch.isnan(torch.tensor(v))
        assert v >= 0.0
