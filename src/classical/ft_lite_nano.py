"""FT-lite — feature tokenizer + tiny Transformer encoder for tabular nano."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.training.base_model import TrainableMixin


class FTLiteNano(TrainableMixin, nn.Module):
    """Per-feature linear tokenizer → CLS + TransformerEncoder → sigmoid head.

    Keeps parameter count ≤ ~2M for 37 agro features on RTX 4060.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        d_token: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        ff_mult: int = 2,
    ) -> None:
        super().__init__()
        if d_token % n_heads != 0:
            raise ValueError(f"d_token={d_token} must be divisible by n_heads={n_heads}")
        self.input_dim = input_dim
        self.d_token = d_token
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout_rate = dropout

        self.tokenizer = nn.Linear(1, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, 1),
            nn.Sigmoid(),
        )
        self._scale = 1.0 / math.sqrt(d_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) → tokens (batch, n_features, d_token)
        tokens = self.tokenizer(x.unsqueeze(-1)) * self._scale
        batch = x.shape[0]
        cls = self.cls_token.expand(batch, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(seq)
        return self.head(encoded[:, 0, :]).squeeze(-1)
