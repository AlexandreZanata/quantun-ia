"""Mini RNN for binary classification."""

import torch.nn as nn

from src.training.base_model import TrainableMixin


class RNNMini(TrainableMixin, nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 16, num_layers: int = 1):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze()
