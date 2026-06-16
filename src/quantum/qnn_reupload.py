"""Variational QNN with data re-uploading — re-embeds inputs at every layer."""

import pennylane as qml
import torch
import torch.nn as nn

from src.quantum.circuit_utils import qnode_diff_method
from src.training.base_model import TrainableMixin


def make_qnn_reupload(n_qubits: int = 4, n_layers: int = 2):
    dev = qml.device("default.qubit", wires=n_qubits)
    diff_method = qnode_diff_method(n_layers)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        for layer in range(n_layers):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    weight_shapes = {"weights": (n_layers, n_qubits, 2)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class QuantumNetReupload(TrainableMixin, nn.Module):
    """Angle-encoding QNN that re-uploads classical features each variational layer."""

    def __init__(self, n_qubits: int = 4, n_layers: int = 2, input_dim: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.pre = nn.Linear(input_dim, n_qubits) if input_dim != n_qubits else nn.Identity()
        self.qlayer = make_qnn_reupload(n_qubits, n_layers)
        self.post = nn.Linear(1, 1)

    def forward(self, x):
        x = self.pre(x) if not isinstance(self.pre, nn.Identity) else x
        out = self.qlayer(x).unsqueeze(1)
        return torch.sigmoid(self.post(out)).squeeze()
