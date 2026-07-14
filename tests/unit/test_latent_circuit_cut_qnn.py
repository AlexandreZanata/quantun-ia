"""Unit tests for latent circuit-cut 6q head."""

import torch

from src.quantum.latent_circuit_cut_qnn import LatentCircuitCutQNN


def test_latent_circuit_cut_forward_shape():
    model = LatentCircuitCutQNN(8, hidden=32, n_layers=1)
    z = torch.randn(4, 8)
    t = torch.randint(0, 10, (4,))
    out = model(z, t)
    assert out.shape == z.shape
    assert model.n_effective_qubits == 6
    assert model.count_parameters() > 0


def test_latent_circuit_cut_fragments_exist():
    model = LatentCircuitCutQNN(8, hidden=16, n_layers=1)
    assert hasattr(model, "qlayer_a")
    assert hasattr(model, "qlayer_b")
