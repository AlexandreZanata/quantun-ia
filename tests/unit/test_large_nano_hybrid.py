"""Unit tests for LargeNanoHybrid — frozen backbone + QNN head."""

from __future__ import annotations

import torch

from src.classical.large_nano_mlp import LargeNanoMLP
from src.quantum.large_nano_hybrid import LargeNanoHybrid
from src.training.device import model_requires_cpu


def test_large_nano_hybrid_forward_shape():
    model = LargeNanoHybrid(input_dim=28, hidden1=32, hidden2=16, hidden3=8, n_qubits=4, n_layers=2)
    x = torch.randn(4, 28)
    out = model(x)
    assert out.shape == (4,)


def test_load_frozen_backbone_from_large_nano():
    classical = LargeNanoMLP(input_dim=28, hidden1=32, hidden2=16, hidden3=8, dropout=0.0)
    hybrid = LargeNanoHybrid(input_dim=28, hidden1=32, hidden2=16, hidden3=8, n_qubits=4, n_layers=2)
    n_loaded = hybrid.load_frozen_backbone_from_large_nano(classical.state_dict())
    assert n_loaded == 6
    hybrid.freeze_backbone()
    assert all(not p.requires_grad for p in hybrid.backbone.parameters())
    assert any(p.requires_grad for p in hybrid.head_proj.parameters())


def test_model_requires_cpu_for_large_nano_hybrid():
    model = LargeNanoHybrid(input_dim=28, hidden1=32, hidden2=16, hidden3=8)
    assert model_requires_cpu(model) is True


def test_to_keeps_backbone_on_cuda_when_available(monkeypatch):
    if not torch.cuda.is_available():
        return
    model = LargeNanoHybrid(
        input_dim=28,
        hidden1=32,
        hidden2=16,
        hidden3=8,
        backbone_device="cuda",
    )
    model.to(torch.device("cpu"))
    assert next(model.backbone.parameters()).device.type == "cuda"


def test_large_nano_hybrid_depolarizing_forward_and_export():
    noisy = LargeNanoHybrid(
        input_dim=8,
        hidden1=16,
        hidden2=8,
        hidden3=4,
        n_qubits=2,
        n_layers=1,
        depolarizing_p=0.03,
        reupload=True,
    )
    x = torch.randn(3, 8)
    out = noisy(x)
    assert out.shape == (3,)
    head = noisy.export_noiseless_head_state()
    assert any(k.startswith("qlayer.") for k in head)
    noiseless = LargeNanoHybrid(
        input_dim=8,
        hidden1=16,
        hidden2=8,
        hidden3=4,
        n_qubits=2,
        n_layers=1,
        depolarizing_p=0.0,
        reupload=True,
    )
    noiseless.load_state_dict(head, strict=False)
    out2 = noiseless(x)
    assert out2.shape == (3,)


def test_set_qlayer_trainable_toggles_requires_grad():
    model = LargeNanoHybrid(
        input_dim=8,
        hidden1=16,
        hidden2=8,
        hidden3=4,
        n_qubits=2,
        n_layers=1,
    )
    model.set_qlayer_trainable(enabled=False)
    assert all(not p.requires_grad for p in model.qlayer.parameters())
    model.set_qlayer_trainable(enabled=True)
    assert any(p.requires_grad for p in model.qlayer.parameters())

