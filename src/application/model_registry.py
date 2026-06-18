"""Canonical nanomodel factory for the Nano Trainer app."""

from __future__ import annotations

from typing import Any

import torch

from src.application.nanotrainer_config import load_nanotrainer_config, model_kind
from src.classical.large_nano_mlp import LargeNanoMLP
from src.classical.mlp import ClassicalNet
from src.classical.perceptron import Perceptron
from src.classical.transformer_mini import TransformerMini
from src.quantum.hybrid_model import HybridSandwich
from src.quantum.qnn_amplitude import QuantumNetAmplitude
from src.quantum.qnn_basic import QuantumNetBasic
from src.quantum.transformer_qnn_fusion import TransformerQNNFusion

TABULAR_KIND = "tabular"
SEQUENCE_KIND = "sequence"


def list_models() -> list[str]:
    return sorted(load_nanotrainer_config().get("models", {}).keys())


def _model_config(name: str) -> dict[str, Any]:
    cfg = load_nanotrainer_config()
    mc = cfg.get("models", {}).get(name)
    if mc is None:
        raise ValueError(f"Unknown nanomodel: {name}")
    return mc


def validate_model_dataset_pair(model_name: str, dataset: str) -> None:
    cfg = load_nanotrainer_config()
    from src.application.nanotrainer_config import dataset_kind, is_allowed_pair

    mk = model_kind(cfg, model_name)
    dk = dataset_kind(cfg, dataset)
    if mk != dk:
        raise ValueError(
            f"Model kind '{mk}' incompatible with dataset kind '{dk}' "
            f"({model_name} × {dataset})"
        )
    if not is_allowed_pair(cfg, model_name, dataset):
        raise ValueError(f"Disallowed pair: {model_name} × {dataset}")


def build_model(
    name: str,
    *,
    input_dim: int,
    seq_len: int | None = None,
    input_dim_seq: int | None = None,
) -> tuple[Any, float]:
    """Build model and default learning rate for tabular or sequence data."""
    mc = _model_config(name)
    lr = float(mc.get("learning_rate", 0.01))
    kind = mc.get("kind", TABULAR_KIND)

    if kind == SEQUENCE_KIND:
        if seq_len is None or input_dim_seq is None:
            raise ValueError(f"Sequence model {name} requires seq_len and input_dim_seq")
        if name == "transformer_mini":
            model = TransformerMini(input_dim=input_dim_seq, d_model=mc.get("d_model", 16))
        elif name == "transformer_qnn_fusion":
            model = TransformerQNNFusion(
                input_dim=input_dim_seq,
                d_model=mc.get("d_model", 16),
                n_qubits=mc.get("n_qubits", 4),
                n_layers=mc.get("n_layers", 2),
            )
        else:
            raise ValueError(f"Unknown sequence model: {name}")
        return model, lr

    if kind != TABULAR_KIND:
        raise ValueError(f"Unknown model kind: {kind}")

    if name == "perceptron":
        return Perceptron(input_dim=input_dim), lr
    if name == "classical_mlp":
        return ClassicalNet(hidden=mc.get("hidden", 16), input_dim=input_dim), lr
    if name == "quantum_angle":
        return (
            QuantumNetBasic(
                n_qubits=mc.get("n_qubits", 4),
                n_layers=mc.get("n_layers", 2),
                input_dim=input_dim,
            ),
            lr,
        )
    if name == "quantum_amplitude":
        return (
            QuantumNetAmplitude(
                n_qubits=mc.get("n_qubits", 4),
                n_layers=mc.get("n_layers", 2),
                input_dim=input_dim,
            ),
            lr,
        )
    if name == "hybrid_sandwich":
        return (
            HybridSandwich(
                input_dim=input_dim,
                n_qubits=mc.get("n_qubits", 4),
                n_layers=mc.get("n_layers", 2),
                reupload=bool(mc.get("reupload", False)),
            ),
            lr,
        )
    if name == "large_nano_mlp":
        return (
            LargeNanoMLP(
                input_dim=input_dim,
                hidden1=int(mc.get("hidden1", 2048)),
                hidden2=int(mc.get("hidden2", 512)),
                hidden3=int(mc.get("hidden3", 64)),
                dropout=float(mc.get("dropout", 0.3)),
            ),
            lr,
        )
    raise ValueError(f"Unknown tabular model: {name}")


def forward_smoke(model: Any, *, kind: str, input_dim: int, seq_len: int = 8, input_dim_seq: int = 2) -> None:
    """Single-batch forward pass to warm circuits and validate shapes."""
    model.eval()
    with torch.no_grad():
        if kind == SEQUENCE_KIND:
            x = torch.randn(2, seq_len, input_dim_seq)
        else:
            x = torch.randn(2, input_dim)
        out = model(x)
        if out.shape != (2,):
            raise ValueError(f"Unexpected output shape: {out.shape}")
