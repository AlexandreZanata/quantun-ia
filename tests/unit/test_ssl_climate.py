"""Unit tests for masked climate SSL helpers."""

from __future__ import annotations

import torch

from src.classical.residual_nano_mlp import ResidualNanoMLP
from src.training.ssl_climate import (
    ResidualNanoSSL,
    copy_encoder_to_residual_nano,
    mask_weather_features,
    weather_feature_indices,
)


def test_weather_feature_indices():
    idx = weather_feature_indices(37)
    assert idx[0] == 9
    assert idx[-1] == 36
    assert len(idx) == 28


def test_mask_weather_features_shape():
    x = torch.randn(8, 37)
    masked, mask = mask_weather_features(x, mask_ratio=0.4)
    assert masked.shape == x.shape
    assert mask.shape == x.shape
    assert mask[:, :9].sum() == 0
    assert mask.any()
    assert torch.allclose(masked[~mask], x[~mask])


def test_copy_encoder_to_residual_nano():
    ssl = ResidualNanoSSL(37, hidden=64, n_blocks=1, dropout=0.0)
    supervised = ResidualNanoMLP(37, hidden=64, n_blocks=1, bottleneck=16, dropout=0.0)
    n = copy_encoder_to_residual_nano(ssl, supervised)
    assert n > 0
    x = torch.randn(3, 37)
    # After copy, stem outputs should match encode path prefix.
    with torch.no_grad():
        h_ssl = ssl.encode(x)
        h_sup = supervised.stem(x)
        for block in supervised.blocks:
            h_sup = block(h_sup)
        assert torch.allclose(h_ssl, h_sup)
