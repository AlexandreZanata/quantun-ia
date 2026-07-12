"""Unit tests for FourierReuploadHybrid shapes and freeze."""

from __future__ import annotations

import torch

from src.classical.residual_nano_mlp import ResidualNanoMLP
from src.quantum.fourier_reupload_hybrid import FourierAngleMap, FourierReuploadHybrid
from src.training.trainer import count_parameters


def test_fourier_angle_map_shape():
    layer = FourierAngleMap(input_dim=16, n_qubits=4, n_frequencies=2)
    out = layer(torch.randn(3, 16))
    assert out.shape == (3, 4)


def test_fourier_reupload_hybrid_flat_forward():
    model = FourierReuploadHybrid(
        input_dim=37,
        encoding="flat",
        n_qubits=4,
        n_layers=1,
        backbone_device="cpu",
    )
    model.freeze_backbone()
    y = model(torch.randn(3, 37))
    assert y.shape == (3,)
    assert count_parameters(model) > 0


def test_fourier_reupload_hybrid_loads_backbone():
    donor = ResidualNanoMLP(input_dim=37, hidden=64, n_blocks=1, bottleneck=16, dropout=0.0)
    hybrid = FourierReuploadHybrid(
        input_dim=37,
        hidden=64,
        n_blocks=1,
        bottleneck=16,
        dropout=0.0,
        n_qubits=2,
        n_layers=1,
        encoding="fourier",
        n_frequencies=2,
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
