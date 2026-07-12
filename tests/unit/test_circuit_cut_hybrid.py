"""Unit tests for circuit-cut hybrid shapes and freeze."""

from __future__ import annotations

import torch

from src.classical.residual_nano_mlp import ResidualNanoMLP
from src.quantum.circuit_cut_hybrid import CircuitCutSixQubitHybrid, ClassicalBottleneckHead


def test_circuit_cut_hybrid_forward():
    model = CircuitCutSixQubitHybrid(
        input_dim=37,
        n_layers=1,
        backbone_device="cpu",
    )
    model.freeze_backbone()
    y = model(torch.randn(3, 37))
    assert y.shape == (3,)


def test_circuit_cut_loads_backbone():
    donor = ResidualNanoMLP(input_dim=37, hidden=64, n_blocks=1, bottleneck=16, dropout=0.0)
    hybrid = CircuitCutSixQubitHybrid(
        input_dim=37,
        hidden=64,
        n_blocks=1,
        bottleneck=16,
        dropout=0.0,
        n_layers=1,
        backbone_device="cpu",
    )
    n = hybrid.load_frozen_backbone_from_residual_nano(donor.state_dict())
    assert n > 0
    hybrid.freeze_backbone()
    trainable = sum(p.numel() for p in hybrid.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in hybrid.parameters() if not p.requires_grad)
    assert trainable > 0
    assert frozen > trainable


def test_classical_bottleneck_head_forward():
    model = ClassicalBottleneckHead(input_dim=37, backbone_device="cpu")
    model.freeze_backbone()
    y = model(torch.randn(2, 37))
    assert y.shape == (2,)


def test_classical_bottleneck_head_cuda_head_cpu_feats():
    """Trainer may place Linear on CUDA while frozen backbone returns CPU feats."""
    if not torch.cuda.is_available():
        return
    model = ClassicalBottleneckHead(input_dim=37, backbone_device="cuda")
    model.freeze_backbone()
    model.head = model.head.to("cuda")
    y = model(torch.randn(2, 37))
    assert y.shape == (2,)
    assert y.device.type == "cuda"
