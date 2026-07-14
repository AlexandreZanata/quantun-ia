"""Classical vs amplitude-encoded 4x4 patch bottlenecks (Phase J / H-Q3.2)."""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn

from src.quantum.circuit_utils import qnode_diff_method
from src.quantum.pennylane_device import DEFAULT_QML_DEVICE, resolve_qml_device


def make_amplitude_pauli_layer(n_qubits: int, n_layers: int, *, qml_device: str | None = None):
    """AmplitudeEmbedding + variational layers → PauliZ vector (length n_qubits)."""
    device_name = qml_device or DEFAULT_QML_DEVICE
    dev = resolve_qml_device(n_qubits, device_name)
    diff_method = qnode_diff_method(n_layers, qml_device=device_name)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 2)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class ClassicalPatchBottleneck(nn.Module):
    """Linear bottleneck matched to amp width (patch_dim → amp_dim → patch_dim)."""

    def __init__(self, patch_dim: int = 48, bottleneck: int = 16) -> None:
        super().__init__()
        self.enc = nn.Linear(patch_dim, bottleneck)
        self.dec = nn.Linear(bottleneck, patch_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.enc(x))
        return self.dec(z)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PatchAmplitudeBottleneck(nn.Module):
    """Project patch → unit-norm amp features → 4q QNN → decode to patch."""

    def __init__(
        self,
        patch_dim: int = 48,
        *,
        n_qubits: int = 4,
        n_layers: int = 2,
        qml_device: str | None = None,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.amp_dim = 2**n_qubits
        self.pre = nn.Linear(patch_dim, self.amp_dim)
        self.qlayer = make_amplitude_pauli_layer(n_qubits, n_layers, qml_device=qml_device)
        self.code = nn.Linear(n_qubits, self.amp_dim)
        self.dec = nn.Linear(self.amp_dim, patch_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PennyLane on CPU
        x_cpu = x if x.device.type == "cpu" else x.cpu()
        amp = self.pre(x_cpu)
        amp = amp / amp.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        q_out = self.qlayer(amp)
        code = torch.tanh(self.code(q_out))
        out = self.dec(code)
        return out.to(x.device) if x.device.type != "cpu" else out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
