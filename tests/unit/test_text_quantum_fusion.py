"""Unit tests for text–quantum token fusion."""

import torch

from src.classical.tiny_dit import TinyDiT
from src.quantum.text_quantum_fusion import ClassicalTextTokenFusion, QuantumTextTokenFusion


def test_classical_and_quantum_fusion_shapes():
    clip = torch.randn(3, 512)
    c = ClassicalTextTokenFusion(512, 32, hidden=64)
    q = QuantumTextTokenFusion(512, 32, n_qubits=4, n_layers=1)
    assert c(clip).shape == (3, 32)
    assert q(clip).shape == (3, 32)


def test_tiny_dit_cross_attn_with_fusion():
    fusion = ClassicalTextTokenFusion(512, 32, hidden=64)
    model = TinyDiT(dim=32, depth=2, n_heads=4, time_dim=64, text_dim=32, use_cross_attn=True)
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 8, (2,))
    text = fusion(torch.randn(2, 512))
    out = model(x, t, text)
    assert out.shape == x.shape
    assert model.cross_attn is not None
