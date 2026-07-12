"""Unit tests for MeasurementDropoutHybrid shapes and masking."""

from __future__ import annotations

import torch

from src.classical.residual_nano_mlp import ResidualNanoMLP
from src.quantum.measurement_dropout_hybrid import MeasurementDropoutHybrid


def test_measurement_dropout_hybrid_plain_forward():
    model = MeasurementDropoutHybrid(
        input_dim=37,
        measurement_dropout=0.0,
        n_qubits=4,
        n_layers=1,
        backbone_device="cpu",
    )
    model.freeze_backbone()
    model.eval()
    y = model(torch.randn(3, 37))
    assert y.shape == (3,)


def test_measurement_dropout_hybrid_mc_and_freeze():
    donor = ResidualNanoMLP(input_dim=37, hidden=64, n_blocks=1, bottleneck=16, dropout=0.0)
    hybrid = MeasurementDropoutHybrid(
        input_dim=37,
        hidden=64,
        n_blocks=1,
        bottleneck=16,
        dropout=0.0,
        n_qubits=2,
        n_layers=1,
        measurement_dropout=0.3,
        backbone_device="cpu",
    )
    n = hybrid.load_frozen_backbone_from_residual_nano(donor.state_dict())
    assert n > 0
    hybrid.freeze_backbone()
    trainable = sum(p.numel() for p in hybrid.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in hybrid.parameters() if not p.requires_grad)
    assert trainable > 0
    assert frozen > trainable
    y = hybrid.forward_mc(torch.randn(2, 37), n_samples=3)
    assert y.shape == (2,)
