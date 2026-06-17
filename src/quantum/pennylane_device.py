"""PennyLane device resolution for configurable QML backends."""

from __future__ import annotations

import pennylane as qml

DEFAULT_QML_DEVICE = "default.qubit"


class QmlDeviceError(Exception):
    """Raised when a requested PennyLane device cannot be constructed."""


def resolve_qml_device(n_wires: int, device_name: str | None = None) -> qml.Device:
    """Create a PennyLane device for ``n_wires`` qubits."""
    name = (device_name or DEFAULT_QML_DEVICE).strip()
    try:
        return qml.device(name, wires=n_wires)
    except Exception as exc:
        raise QmlDeviceError(f"PennyLane device '{name}' unavailable: {exc}") from exc
