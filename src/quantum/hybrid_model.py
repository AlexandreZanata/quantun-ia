import pennylane as qml
import torch
import torch.nn as nn

from src.quantum.circuit_utils import qnode_diff_method
from src.training.base_model import TrainableMixin


def make_quantum_layer(n_qubits: int, n_layers: int, *, reupload: bool = False):
    dev = qml.device("default.qubit", wires=n_qubits)
    diff_method = qnode_diff_method(n_layers)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        if reupload:
            for layer in range(n_layers):
                qml.AngleEmbedding(inputs, wires=range(n_qubits))
                qml.StronglyEntanglingLayers(weights[layer : layer + 1], wires=range(n_qubits))
        else:
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class HybridSandwich(TrainableMixin, nn.Module):
    """Classical -> Quantum -> Classical"""

    def __init__(self, input_dim, n_qubits=4, n_layers=2, *, reupload: bool = False):
        super().__init__()
        self.n_qubits = n_qubits
        self.pre = nn.Sequential(nn.Linear(input_dim, n_qubits), nn.Tanh())
        self.qlayer = make_quantum_layer(n_qubits, n_layers, reupload=reupload)
        self.post = nn.Sequential(nn.Linear(n_qubits, 1), nn.Sigmoid())
        self._quantum_enabled = True

    def set_quantum_enabled(self, enabled: bool) -> None:
        """Bypass PennyLane block during classical warm-start phase."""
        self._quantum_enabled = enabled

    def forward(self, x):
        x = self.pre(x)
        if self._quantum_enabled:
            x = self.qlayer(x)
        return self.post(x).squeeze()


class QuantumFirst(TrainableMixin, nn.Module):
    """Quantum -> Classical"""

    def __init__(
        self, input_dim=2, n_qubits=4, n_layers=2, output_dim=1, *, reupload: bool = False
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.input_proj = nn.Linear(input_dim, n_qubits)
        self.qlayer = make_quantum_layer(n_qubits, n_layers, reupload=reupload)
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

    def __init__(self, input_dim=2, n_qubits=4, n_layers=2, *, reupload: bool = False):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(input_dim, n_qubits), nn.Tanh())
        self.qlayer = make_quantum_layer(n_qubits, n_layers, reupload=reupload)
        self.post = nn.Linear(n_qubits, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.qlayer(x)
        return torch.sigmoid(self.post(x)).squeeze()
