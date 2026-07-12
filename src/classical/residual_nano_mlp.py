"""Residual MLP nano for tabular agro — skip connections within hidden width."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.training.base_model import TrainableMixin


class _ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.fc = nn.Linear(width, width)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.act(self.fc(x)))


class ResidualNanoMLP(TrainableMixin, nn.Module):
    """Project to hidden width, stack residual blocks, then bottleneck head."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden: int = 512,
        n_blocks: int = 3,
        bottleneck: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.n_blocks = n_blocks
        self.bottleneck = bottleneck
        self.dropout_rate = dropout
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([_ResidualBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden, bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)
