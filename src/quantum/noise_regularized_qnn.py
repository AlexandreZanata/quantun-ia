"""PennyLane QNN layers with optional depolarizing noise for regularization."""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn

from src.quantum.circuit_utils import qnode_diff_method
from src.training.base_model import TrainableMixin


def _apply_depolarizing_noise(noise_p: float, n_qubits: int) -> None:
    if noise_p <= 0.0:
        return
    for wire in range(n_qubits):
        qml.DepolarizingChannel(noise_p, wires=wire)


def make_noisy_quantum_layer(
    n_qubits: int,
    n_layers: int,
    *,
    reupload: bool = False,
    depolarizing_p: float = 0.0,
):
    """Quantum layer with optional depolarizing noise after each variational block."""
    device_name = "default.mixed" if depolarizing_p > 0.0 else "default.qubit"
    dev = qml.device(device_name, wires=n_qubits)
    diff_method = qnode_diff_method(n_layers)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        if reupload:
            for layer in range(n_layers):
                qml.AngleEmbedding(inputs, wires=range(n_qubits))
                qml.StronglyEntanglingLayers(weights[layer : layer + 1], wires=range(n_qubits))
                _apply_depolarizing_noise(depolarizing_p, n_qubits)
        else:
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            for layer in range(n_layers):
                qml.StronglyEntanglingLayers(weights[layer : layer + 1], wires=range(n_qubits))
                _apply_depolarizing_noise(depolarizing_p, n_qubits)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class NoiseRegularizedHybridSandwich(TrainableMixin, nn.Module):
    """HybridSandwich with depolarizing noise in the QNN forward pass during training."""

    def __init__(
        self,
        input_dim: int,
        n_qubits: int = 4,
        n_layers: int = 2,
        *,
        reupload: bool = False,
        depolarizing_p: float = 0.03,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.reupload = reupload
        self.depolarizing_p = depolarizing_p
        self.pre = nn.Sequential(nn.Linear(input_dim, n_qubits), nn.Tanh())
        self.qlayer = make_noisy_quantum_layer(
            n_qubits,
            n_layers,
            reupload=reupload,
            depolarizing_p=depolarizing_p,
        )
        self.post = nn.Sequential(nn.Linear(n_qubits, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = self.qlayer(x)
        return self.post(x).squeeze()
