"""Unit tests for nanomodel registry."""

import pytest
import torch

from src.application.model_registry import (
    build_model,
    forward_smoke,
    list_models,
    validate_model_dataset_pair,
)
from src.application.nanotrainer_config import load_nanotrainer_config


def test_list_models_includes_core_entries():
    names = list_models()
    assert "perceptron" in names
    assert "transformer_qnn_fusion" in names


def test_build_perceptron_tabular():
    model, lr = build_model("perceptron", input_dim=30)
    x = torch.randn(4, 30)
    out = model(x)
    assert out.shape == (4,)
    assert lr == 0.02


def test_build_residual_nano_distill():
    model, lr = build_model("residual_nano_distill", input_dim=37)
    out = model(torch.randn(3, 37))
    assert out.shape == (3,)
    assert lr == 0.001
    assert model.hidden == 512


def test_build_transformer_fusion_sequence():
    model, lr = build_model(
        "transformer_qnn_fusion",
        input_dim=0,
        seq_len=12,
        input_dim_seq=4,
    )
    out = model(torch.randn(2, 12, 4))
    assert out.shape == (2,)


def test_incompatible_pair_raises():
    with pytest.raises(ValueError, match="incompatible"):
        validate_model_dataset_pair("perceptron", "sequential_phase")


def test_disallowed_same_kind_pair_raises():
    with pytest.raises(ValueError, match="Disallowed"):
        validate_model_dataset_pair("perceptron", "mnist_binary")


@pytest.mark.parametrize("name", list_models())
def test_forward_smoke_all_models(name):
    mc = load_nanotrainer_config().get("models", {}).get(name, {})
    kind = mc.get("kind", "tabular")
    if kind == "sequence":
        model, _ = build_model(name, input_dim=0, seq_len=12, input_dim_seq=4)
        forward_smoke(model, kind="sequence", input_dim=0, seq_len=12, input_dim_seq=4)
    else:
        model, _ = build_model(name, input_dim=16)
        forward_smoke(model, kind="tabular", input_dim=16)
