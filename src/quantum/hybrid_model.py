import pennylane as qml
import torch
import torch.nn as nn

from src.training.base_model import TrainableMixin


def make_quantum_layer(n_qubits, n_layers):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class HybridSandwich(TrainableMixin, nn.Module):
    """Classical -> Quantum -> Classical"""

    def __init__(self, input_dim, n_qubits=4, n_layers=2):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(input_dim, n_qubits), nn.Tanh())
        self.qlayer = make_quantum_layer(n_qubits, n_layers)
        self.post = nn.Sequential(nn.Linear(n_qubits, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.pre(x)
        x = self.qlayer(x)
        return self.post(x).squeeze()


class QuantumFirst(TrainableMixin, nn.Module):
    """Quantum -> Classical"""

    def __init__(self, input_dim=2, n_qubits=4, n_layers=2, output_dim=1):
        super().__init__()
        self.n_qubits = n_qubits
        self.input_proj = nn.Linear(input_dim, n_qubits)
        self.qlayer = make_quantum_layer(n_qubits, n_layers)
        self.post = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = torch.tanh(self.input_proj(x))
        x = self.qlayer(x)
        return self.post(x).squeeze()


class ClassicalFirst(TrainableMixin, nn.Module):
    """Classical -> Quantum (quantum as final decision layer)"""

    def __init__(self, input_dim=2, n_qubits=4, n_layers=2):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(input_dim, n_qubits), nn.Tanh())
        self.qlayer = make_quantum_layer(n_qubits, n_layers)
        self.post = nn.Linear(n_qubits, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.qlayer(x)
        return torch.sigmoid(self.post(x)).squeeze()
