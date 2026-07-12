"""Unit tests for ResidualNanoHybrid shapes and freeze."""

from __future__ import annotations

import torch

from src.classical.residual_nano_mlp import ResidualNanoMLP
from src.quantum.residual_nano_hybrid import ResidualNanoHybrid
from src.training.trainer import count_parameters


def test_residual_nano_hybrid_plain_forward():
    model = ResidualNanoHybrid(input_dim=37, residual_skip=False, n_qubits=4, n_layers=1)
    model.freeze_backbone()
    x = torch.randn(3, 37)
    y = model(x)
    assert y.shape == (3,)
    assert count_parameters(model) > 0


def test_residual_nano_hybrid_skip_loads_backbone():
    donor = ResidualNanoMLP(input_dim=37, hidden=64, n_blocks=1, bottleneck=16, dropout=0.0)
    hybrid = ResidualNanoHybrid(
        input_dim=37,
        hidden=64,
        n_blocks=1,
        bottleneck=16,
        dropout=0.0,
        n_qubits=2,
        n_layers=1,
        residual_skip=True,
        backbone_device="cpu",
    )
    n = hybrid.load_frozen_backbone_from_residual_nano(donor.state_dict())
    assert n > 0
    hybrid.freeze_backbone()
    trainable = sum(p.numel() for p in hybrid.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in hybrid.parameters() if not p.requires_grad)
    assert trainable > 0
    assert frozen > trainable
    y = hybrid(torch.randn(2, 37))
    assert y.shape == (2,)
