"""Transformer encoder fused with variational quantum classifier for sequences."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.classical.transformer_mini import PositionalEncoding
from src.quantum.hybrid_model import make_quantum_layer
from src.training.base_model import TrainableMixin


class SequenceTransformerEncoder(nn.Module):
    """Transformer encoder returning a pooled sequence embedding."""

    def __init__(self, input_dim: int = 4, d_model: int = 16, nhead: int = 2, num_layers: int = 1):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=32, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.embed(x)
        x = self.pos(x)
        return self.encoder(x).mean(dim=1)


class TransformerQNNFusion(TrainableMixin, nn.Module):
    """Transformer-mini encoder → linear qubit projection → QNN decision head."""

    def __init__(
        self,
        input_dim: int = 4,
        d_model: int = 16,
        n_qubits: int = 4,
        n_layers: int = 2,
        *,
        reupload: bool = False,
    ):
        super().__init__()
        self.encoder = SequenceTransformerEncoder(input_dim=input_dim, d_model=d_model)
        self.to_qubits = nn.Linear(d_model, n_qubits)
        self.qlayer = make_quantum_layer(n_qubits, n_layers, reupload=reupload)
        self.post = nn.Sequential(nn.Linear(n_qubits, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        features = self.encoder.encode(x)
        q_in = torch.tanh(self.to_qubits(features))
        q_out = self.qlayer(q_in)
        return self.post(q_out).squeeze()
