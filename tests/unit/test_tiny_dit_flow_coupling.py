"""Unit tests for TinyDiT flow couplings."""

import torch

from src.classical.tiny_dit import ClassicalAffineCoupling, TinyDiT, UnitaryFlowCoupling


def test_classical_coupling_shape():
    m = ClassicalAffineCoupling(64)
    x = torch.randn(2, 8, 64)
    y = m(x)
    assert y.shape == x.shape


def test_unitary_coupling_preserves_shape():
    m = UnitaryFlowCoupling(64, n_layers=2)
    x = torch.randn(2, 8, 64)
    y = m(x)
    assert y.shape == x.shape


def test_tiny_dit_forward_classical_and_unitary():
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 10, (2,))
    for kind in ("classical", "unitary"):
        model = TinyDiT(dim=32, depth=2, n_heads=4, coupling=kind, time_dim=64)
        out = model(x, t)
        assert out.shape == x.shape
        assert model.count_parameters() > 0
