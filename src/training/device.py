"""Training device resolution (CPU/CUDA auto-detect)."""

from __future__ import annotations

import os

import torch


def resolve_device(prefer: str | None = None) -> torch.device:
    """
    Resolve torch device from preference or ``QML_DEVICE`` env var.

    Values: ``auto`` (default), ``cpu``, ``cuda``.
    Quantum PennyLane simulators default to CPU; classical models benefit from CUDA.
    """
    choice = (prefer or os.environ.get("QML_DEVICE", "auto")).lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
