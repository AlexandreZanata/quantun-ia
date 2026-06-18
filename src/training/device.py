"""Training device resolution (CPU/CUDA auto-detect)."""

from __future__ import annotations

import os

import torch
from torch import nn


def model_requires_cpu(model: nn.Module) -> bool:
    """PennyLane TorchLayer simulators must stay on CPU even when CUDA is available."""
    try:
        from pennylane.qnn import TorchLayer
    except ImportError:
        return False
    return any(isinstance(module, TorchLayer) for module in model.modules())


def resolve_device(prefer: str | None = None, *, model: nn.Module | None = None) -> torch.device:
    """
    Resolve torch device from preference or ``QML_DEVICE`` env var.

    Values: ``auto`` (default), ``cpu``, ``cuda``.
    Quantum PennyLane simulators default to CPU; classical models benefit from CUDA.
    """
    if model is not None and model_requires_cpu(model):
        return torch.device("cpu")

    choice = (prefer or os.environ.get("QML_DEVICE", "auto")).lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
