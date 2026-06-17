"""Shared PennyLane circuit utilities for research-grade QNN training."""

from __future__ import annotations

# PennyLane: parameter-shift is preferred for deep circuits but incompatible
# with batched TorchLayer training (PennyLane #4462). Use backprop for training,
# parameter-shift for single-sample gradient diagnostics (exp_006).
PARAMETER_SHIFT_MIN_LAYERS = 3


def qnode_diff_method(
    n_layers: int,
    *,
    for_training: bool = True,
    qml_device: str | None = None,
) -> str:
    """
    Gradient method for QNodes.
    Training uses backprop (batched). Diagnostics use parameter-shift when deep.
    Lightning simulators use adjoint — backprop is unsupported on those devices.
    """
    if for_training and qml_device and "lightning" in qml_device:
        return "adjoint"
    if for_training:
        return "backprop"
    if n_layers >= PARAMETER_SHIFT_MIN_LAYERS:
        return "parameter-shift"
    return "backprop"
