"""Unit tests for Transformer → QNN fusion model."""

import torch

from src.quantum.transformer_qnn_fusion import TransformerQNNFusion


def test_transformer_qnn_fusion_forward_shape():
    model = TransformerQNNFusion(input_dim=4, d_model=8, n_qubits=4, n_layers=1)
    x = torch.randn(5, 12, 4)
    out = model(x)
    assert out.shape == (5,)
