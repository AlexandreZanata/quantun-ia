"""Unit tests for batched validation metrics."""

from __future__ import annotations

import torch

from src.classical.large_nano_mlp import LargeNanoMLP
from src.training.batched_trainer import evaluate_with_auc, evaluate_with_auc_batched


def test_evaluate_with_auc_batched_matches_full():
    torch.manual_seed(0)
    model = LargeNanoMLP(input_dim=8, hidden1=32, hidden2=16, hidden3=8)
    x = torch.randn(200, 8)
    y = (torch.rand(200) > 0.5).float()

    full = evaluate_with_auc(model, x, y)
    batched = evaluate_with_auc_batched(model, x, y, batch_size=32)

    assert abs(full["roc_auc"] - batched["roc_auc"]) < 1e-5
    assert abs(full["accuracy"] - batched["accuracy"]) < 1e-5
