"""Pauli / classical-shadow feature maps for tabular QML (H-Q2.3).

Builds a fixed 4-qubit angle-encoded circuit and returns a bank of Pauli
expectation values (analytic ``default.qubit`` = infinite-shot classical shadow
limit for those observables). Features are then consumed by classical nanos.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pennylane as qml
import torch


def build_pauli_observables(n_qubits: int, n_features: int = 64) -> list[qml.operation.Operator]:
    """Deterministic Pauli bank: all 1-local then 2-local XY/Z products, truncated."""
    if n_qubits < 1:
        msg = "n_qubits must be >= 1"
        raise ValueError(msg)
    if n_features < 1:
        msg = "n_features must be >= 1"
        raise ValueError(msg)

    observables: list[qml.operation.Operator] = []
    singles = (qml.PauliX, qml.PauliY, qml.PauliZ)
    for wire in range(n_qubits):
        for pauli in singles:
            observables.append(pauli(wire))
            if len(observables) >= n_features:
                return observables[:n_features]

    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            for left in singles:
                for right in singles:
                    observables.append(left(i) @ right(j))
                    if len(observables) >= n_features:
                        return observables[:n_features]

    if len(observables) < n_features:
        msg = f"only {len(observables)} Pauli observables for n_qubits={n_qubits}; need {n_features}"
        raise ValueError(msg)
    return observables[:n_features]


def _seeded_weights(n_layers: int, n_qubits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low=0.0, high=2.0 * np.pi, size=(n_layers, n_qubits, 3)).astype(np.float64)


def _seeded_projection(input_dim: int, n_qubits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 17)
    matrix = rng.normal(size=(input_dim, n_qubits)).astype(np.float64)
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return matrix / norms


class PauliShadowFeatureEncoder:
    """Project tabular rows → n_qubits angles → 64 Pauli expectation features."""

    def __init__(
        self,
        input_dim: int,
        *,
        n_qubits: int = 4,
        n_layers: int = 1,
        n_features: int = 64,
        seed: int = 42,
        device_name: str = "default.qubit",
    ) -> None:
        self.input_dim = int(input_dim)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.n_features = int(n_features)
        self.seed = int(seed)
        self.device_name = device_name
        self.projection = _seeded_projection(self.input_dim, self.n_qubits, self.seed)
        self.weights = _seeded_weights(self.n_layers, self.n_qubits, self.seed)
        self.observables = build_pauli_observables(self.n_qubits, self.n_features)
        self._qnode = self._build_qnode()

    def _build_qnode(self):
        dev = qml.device(self.device_name, wires=self.n_qubits)
        weights = self.weights
        observables = self.observables
        n_qubits = self.n_qubits
        n_layers = self.n_layers

        @qml.qnode(dev, interface="numpy")
        def circuit(angles: np.ndarray) -> Sequence[float]:
            qml.AngleEmbedding(angles, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights[:n_layers], wires=range(n_qubits))
            return [qml.expval(obs) for obs in observables]

        return circuit

    def project_angles(self, x: np.ndarray) -> np.ndarray:
        """Map (n, input_dim) → (n, n_qubits) angles in (-pi, pi) via tanh."""
        features = np.asarray(x, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            msg = f"expected shape (n, {self.input_dim}), got {features.shape}"
            raise ValueError(msg)
        projected = features @ self.projection
        return np.pi * np.tanh(projected)

    def transform_row(self, row: np.ndarray) -> np.ndarray:
        angles = self.project_angles(np.asarray(row, dtype=np.float64).reshape(1, -1))[0]
        values = np.asarray(self._qnode(angles), dtype=np.float64)
        return values.reshape(-1)

    def transform(
        self,
        x: np.ndarray,
        *,
        progress_every: int = 0,
    ) -> np.ndarray:
        """Transform (n, input_dim) → (n, n_features) Pauli/shadow features."""
        angles = self.project_angles(x)
        out = np.empty((angles.shape[0], self.n_features), dtype=np.float64)
        for i, row_angles in enumerate(angles):
            out[i] = np.asarray(self._qnode(row_angles), dtype=np.float64)
            if progress_every > 0 and (i + 1) % progress_every == 0:
                print(f"  shadow features {i + 1}/{len(angles)}", flush=True)
        return out

    def transform_torch(self, x: torch.Tensor) -> torch.Tensor:
        array = x.detach().cpu().numpy()
        return torch.tensor(self.transform(array), dtype=torch.float32)
