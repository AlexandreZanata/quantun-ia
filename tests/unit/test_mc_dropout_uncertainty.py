"""Unit tests for MC-dropout uncertainty helper."""

from __future__ import annotations

import torch

from src.application.mc_dropout_uncertainty import enable_mc_dropout, mc_dropout_predict
from src.classical.residual_nano_mlp import ResidualNanoMLP


def test_mc_dropout_predict_std_positive():
    model = ResidualNanoMLP(input_dim=8, hidden=16, n_blocks=1, bottleneck=4, dropout=0.5)
    x = torch.randn(1, 8)
    unc = mc_dropout_predict(model, x, n_samples=8, seed=0)
    assert unc.n_samples == 8
    assert unc.method == "mc_dropout"
    assert 0.0 <= unc.mean_probability <= 1.0
    assert unc.std_probability >= 0.0


def test_enable_mc_dropout_keeps_dropout_train():
    model = ResidualNanoMLP(input_dim=4, hidden=8, n_blocks=1, bottleneck=4, dropout=0.3)
    enable_mc_dropout(model)
    drops = [m for m in model.modules() if m.__class__.__name__ == "Dropout"]
    assert drops
    assert all(m.training for m in drops)
