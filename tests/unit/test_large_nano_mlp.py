"""Unit tests for LargeNanoMLP architecture."""

from src.classical.large_nano_mlp import LargeNanoMLP
from src.training.trainer import count_parameters


def test_large_nano_mlp_param_count_higgs_dim():
    model = LargeNanoMLP(input_dim=28)
    n_params = count_parameters(model)
    assert 1_100_000 <= n_params <= 1_200_000


def test_large_nano_mlp_param_count_synthea_dim():
    model = LargeNanoMLP(input_dim=40)
    n_params = count_parameters(model)
    assert 1_150_000 <= n_params <= 1_250_000


def test_large_nano_mlp_forward_shape():
    import torch

    model = LargeNanoMLP(input_dim=28)
    x = torch.randn(8, 28)
    out = model(x)
    assert out.shape == (8,)
