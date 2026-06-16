"""Unit tests for gradient variance measurement."""

import torch

from src.quantum.qnn_basic import QuantumNetBasic
from src.training.gradients import measure_gradient_variance


def test_measure_gradient_variance_finite():
    def factory(n_qubits):
        return QuantumNetBasic(n_qubits=n_qubits, n_layers=1, input_dim=2)

    variances = measure_gradient_variance(factory, [2, 4], n_samples=3)
    assert 2 in variances and 4 in variances
    for stats in variances.values():
        assert not torch.isnan(torch.tensor(stats["mean"]))
        assert stats["mean"] >= 0.0
        assert stats["ci_low"] <= stats["mean"] <= stats["ci_high"]


def test_measure_gradient_variance_parameter_shift_batch_one():
    def factory(n_qubits):
        return QuantumNetBasic(n_qubits=n_qubits, n_layers=2, input_dim=2)

    variances = measure_gradient_variance(
        factory, [2], n_samples=2, batch_size=1, use_parameter_shift=True
    )
    assert 2 in variances
    assert variances[2]["n_samples"] >= 0
