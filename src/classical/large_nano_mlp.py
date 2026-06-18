"""Large-scale classical MLP for Phase L open tabular datasets (~1.2M params)."""

from __future__ import annotations

import torch.nn as nn

from src.training.base_model import TrainableMixin


class LargeNanoMLP(TrainableMixin, nn.Module):
    """Deep MLP backbone for million-row tabular nano training on RTX 4060."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden1: int = 2048,
        hidden2: int = 512,
        hidden3: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.dropout_rate = dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
