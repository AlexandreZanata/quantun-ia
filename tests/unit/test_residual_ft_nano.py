"""Unit tests for ResidualNano / NarrowDeepNano / FT-lite shapes."""

from __future__ import annotations

import torch

from src.classical.ft_lite_nano import FTLiteNano
from src.classical.narrow_deep_nano import NarrowDeepNano
from src.classical.residual_nano_mlp import ResidualNanoMLP
from src.training.trainer import count_parameters


def test_residual_nano_forward_shape():
    model = ResidualNanoMLP(input_dim=37, hidden=512, n_blocks=3, dropout=0.2)
    x = torch.randn(8, 37)
    y = model(x)
    assert y.shape == (8,)
    assert torch.isfinite(y).all()
    assert 0.0 <= float(y.detach().min()) and float(y.detach().max()) <= 1.0


def test_narrow_deep_nano_forward_shape():
    model = NarrowDeepNano(input_dim=37, width=512, depth=3, bottleneck=64, dropout=0.2)
    x = torch.randn(4, 37)
    y = model(x)
    assert y.shape == (4,)
    assert count_parameters(model) > 100_000


def test_ft_lite_nano_forward_shape():
    model = FTLiteNano(input_dim=37, d_token=32, n_heads=4, n_layers=2, dropout=0.1)
    x = torch.randn(5, 37)
    y = model(x)
    assert y.shape == (5,)
    assert count_parameters(model) < 2_000_000
