"""Track 1: Classical perceptron from scratch."""

import torch
import torch.nn as nn

from src.training.base_model import TrainableMixin


class Perceptron(TrainableMixin, nn.Module):
    def __init__(self, input_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze()
