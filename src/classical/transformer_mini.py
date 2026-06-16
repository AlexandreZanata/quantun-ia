"""Track 3: Nano transformer for short-sequence binary classification."""

import math

import torch
import torch.nn as nn

from src.training.base_model import TrainableMixin


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerMini(TrainableMixin, nn.Module):
    """Minimal transformer for binary classification."""

    def __init__(self, input_dim: int = 2, d_model: int = 16, nhead: int = 2, num_layers: int = 1):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=32, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.embed(x)
        x = self.pos(x)
        x = self.encoder(x).mean(dim=1)
        return self.head(x).squeeze()
