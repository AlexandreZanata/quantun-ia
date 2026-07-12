"""Narrow-deep MLP nano — depth over width for VRAM-friendly tabular training."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.training.base_model import TrainableMixin


class NarrowDeepNano(TrainableMixin, nn.Module):
    """Repeated equal-width layers then bottleneck → sigmoid."""

    def __init__(
        self,
        input_dim: int,
        *,
        width: int = 512,
        depth: int = 3,
        bottleneck: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.width = width
        self.depth = depth
        self.bottleneck = bottleneck
        self.dropout_rate = dropout

        layers: list[nn.Module] = [
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(max(depth - 1, 0)):
            layers.extend(
                [
                    nn.Linear(width, width),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.extend(
            [
                nn.Linear(width, bottleneck),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck, 1),
                nn.Sigmoid(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
