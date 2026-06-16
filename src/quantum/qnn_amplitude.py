"""QNN with amplitude encoding (unit-norm vectors required)."""

import pennylane as qml
import torch
import torch.nn as nn

from src.training.base_model import TrainableMixin


def make_amplitude_circuit(n_qubits: int, n_layers: int):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    weight_shapes = {"weights": (n_layers, n_qubits, 2)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class QuantumNetAmplitude(TrainableMixin, nn.Module):
    """Variational classifier with amplitude encoding."""

    def __init__(self, n_qubits: int = 2, n_layers: int = 2, input_dim: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.amp_dim = 2**n_qubits
        self.qlayer = make_amplitude_circuit(n_qubits, n_layers)
        self.post = nn.Linear(1, 1)

    def forward(self, x):
        if x.shape[1] < self.amp_dim:
            pad = torch.zeros(x.shape[0], self.amp_dim - x.shape[1], device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        x = x[:, : self.amp_dim]
        norm = x.norm(dim=1, keepdim=True).clamp(min=1e-8)
        x = x / norm
        out = self.qlayer(x).unsqueeze(1)
        return torch.sigmoid(self.post(out)).squeeze()
