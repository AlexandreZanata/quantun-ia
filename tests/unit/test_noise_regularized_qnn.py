"""Unit tests for depolarizing-noise hybrid QNN."""

import torch

from src.quantum.hybrid_model import HybridSandwich
from src.quantum.noise_regularized_qnn import NoiseRegularizedHybridSandwich


def test_noise_regularized_hybrid_forward():
    x = torch.randn(8, 12)
    model = NoiseRegularizedHybridSandwich(
        input_dim=12,
        n_qubits=4,
        n_layers=1,
        reupload=True,
        depolarizing_p=0.02,
    )
    out = model(x)
    assert out.shape == (8,)


def test_zero_noise_matches_hybrid_sandwich_shape():
    x = torch.randn(4, 6)
    noisy = NoiseRegularizedHybridSandwich(input_dim=6, n_qubits=4, n_layers=1, depolarizing_p=0.0)
    baseline = HybridSandwich(input_dim=6, n_qubits=4, n_layers=1)
    assert noisy(x).shape == baseline(x).shape
