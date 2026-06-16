"""Multi-layer perceptron baseline for binary classification."""

import torch.nn as nn

from src.training.base_model import TrainableMixin


class ClassicalNet(TrainableMixin, nn.Module):
    def __init__(self, input_dim: int = 2, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze()
