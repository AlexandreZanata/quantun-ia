"""Projected quantum kernel (PQK) features for tabular QML (H-Q2.6).

Maps rows → 4q angle embedding → 1-local Pauli expectations (projections),
then classical RBF KernelRidge / Nyström+linear heads on those projections.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pennylane as qml
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def _seeded_weights(n_layers: int, n_qubits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low=0.0, high=2.0 * np.pi, size=(n_layers, n_qubits, 3)).astype(np.float64)


def _seeded_projection(input_dim: int, n_qubits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 17)
    matrix = rng.normal(size=(input_dim, n_qubits)).astype(np.float64)
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return matrix / norms


def build_local_pauli_observables(n_qubits: int) -> list[qml.operation.Operator]:
    """1-local X/Y/Z per wire — Huang-style projected observables."""
    if n_qubits < 1:
        msg = "n_qubits must be >= 1"
        raise ValueError(msg)
    observables: list[qml.operation.Operator] = []
    for wire in range(n_qubits):
        observables.extend([qml.PauliX(wire), qml.PauliY(wire), qml.PauliZ(wire)])
    return observables


class ProjectedQuantumFeatureEncoder:
    """Project tabular rows → n_qubits angles → 1-local Pauli projections."""

    def __init__(
        self,
        input_dim: int,
        *,
        n_qubits: int = 4,
        n_layers: int = 1,
        seed: int = 42,
        device_name: str = "default.qubit",
    ) -> None:
        self.input_dim = int(input_dim)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.seed = int(seed)
        self.device_name = device_name
        self.projection = _seeded_projection(self.input_dim, self.n_qubits, self.seed)
        self.weights = _seeded_weights(self.n_layers, self.n_qubits, self.seed)
        self.observables = build_local_pauli_observables(self.n_qubits)
        self.n_features = len(self.observables)
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
        features = np.asarray(x, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            msg = f"expected shape (n, {self.input_dim}), got {features.shape}"
            raise ValueError(msg)
        projected = features @ self.projection
        return np.pi * np.tanh(projected)

    def transform(
        self,
        x: np.ndarray,
        *,
        progress_every: int = 0,
    ) -> np.ndarray:
        """Transform (n, input_dim) → (n, 3 * n_qubits) projected features."""
        angles = self.project_angles(x)
        out = np.empty((angles.shape[0], self.n_features), dtype=np.float64)
        for i, row_angles in enumerate(angles):
            out[i] = np.asarray(self._qnode(row_angles), dtype=np.float64)
            if progress_every > 0 and (i + 1) % progress_every == 0:
                print(f"  PQK projections {i + 1}/{len(angles)}", flush=True)
        return out


def fit_kernel_ridge_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    *,
    alpha: float = 1.0,
    gamma: float | None = None,
) -> np.ndarray:
    """Fit RBF KernelRidge on projected features; return val decision scores."""
    model = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
    model.fit(x_train, y_train.astype(np.float64))
    return np.asarray(model.predict(x_val), dtype=np.float64)


def fit_nystroem_logistic_proba(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    *,
    n_components: int = 256,
    gamma: float | None = None,
    seed: int = 42,
    max_iter: int = 500,
) -> np.ndarray:
    """Nyström RBF PQK features → logistic (linear head); return val P(y=1)."""
    n_comp = min(int(n_components), len(x_train))
    pipe = Pipeline(
        [
            (
                "nystroem",
                Nystroem(
                    kernel="rbf",
                    gamma=gamma,
                    n_components=n_comp,
                    random_state=seed,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=max_iter, random_state=seed),
            ),
        ]
    )
    pipe.fit(x_train, y_train)
    return np.asarray(pipe.predict_proba(x_val)[:, 1], dtype=np.float64)
