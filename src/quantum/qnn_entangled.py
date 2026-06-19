import pennylane as qml
import torch
import torch.nn as nn

from src.quantum.circuit_utils import qnode_diff_method
from src.training.base_model import TrainableMixin


def _apply_entanglement(entanglement: str, n_qubits: int) -> None:
    if entanglement == "chain":
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    elif entanglement == "chain_half":
        for i in range(0, n_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])
    elif entanglement == "ring":
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])


def make_entangled_circuit(
    n_qubits: int,
    n_layers: int,
    entanglement: str = "chain",
    *,
    reupload: bool = False,
):
    dev = qml.device("default.qubit", wires=n_qubits)
    diff_method = qnode_diff_method(n_layers)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        if reupload:
            for layer_idx in range(n_layers):
                qml.AngleEmbedding(inputs, wires=range(n_qubits))
                for i in range(n_qubits):
                    qml.RY(weights[layer_idx, i, 0], wires=i)
                    qml.RZ(weights[layer_idx, i, 1], wires=i)
                if entanglement != "none":
                    _apply_entanglement(entanglement, n_qubits)
        else:
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            for layer_idx in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[layer_idx, i, 0], wires=i)
                    qml.RZ(weights[layer_idx, i, 1], wires=i)
                if entanglement != "none":
                    _apply_entanglement(entanglement, n_qubits)

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 2)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class QuantumNetEntangled(TrainableMixin, nn.Module):
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        entanglement: str = "chain",
        input_dim: int = 2,
        *,
        reupload: bool = False,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement = entanglement
        self.reupload = reupload
        self.pre = nn.Linear(input_dim, n_qubits) if input_dim != n_qubits else nn.Identity()
        self.qlayer = make_entangled_circuit(n_qubits, n_layers, entanglement, reupload=reupload)
        self.post = nn.Linear(n_qubits, 1)

    def set_entanglement(self, entanglement: str) -> None:
        """Swap CNOT topology; preserve variational weights when shapes match."""
        state = self.qlayer.state_dict()
        self.entanglement = entanglement
        self.qlayer = make_entangled_circuit(
            self.n_qubits,
            self.n_layers,
            entanglement,
            reupload=self.reupload,
        )
        self.qlayer.load_state_dict(state)

    def forward(self, x):
        x = self.pre(x) if not isinstance(self.pre, nn.Identity) else x
        x = x[:, : self.n_qubits]
        out = self.qlayer(x)
        return torch.sigmoid(self.post(out)).squeeze()
