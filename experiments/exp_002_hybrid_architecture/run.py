"""
EXP 002 — Hybrid Architecture
Write your hypothesis in hypothesis.md BEFORE running this script.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.data.generators import make_binary_classification
from src.quantum.hybrid_model import ClassicalFirst, HybridSandwich, QuantumFirst
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_002_hybrid_architecture"
EXP_ID = "exp_002"


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    log_event("info", "experiment run started", exp_id=EXP_ID)

    X, y, _ = make_binary_classification(
        n_samples=cfg["n_samples"],
        dataset=cfg["dataset"],
        noise=cfg["noise"],
        random_state=cfg["random_state"],
    )
    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    models = [
        (HybridSandwich(input_dim=2, n_qubits=4, n_layers=2), "hybrid_sandwich"),
        (QuantumFirst(input_dim=2, n_qubits=4, n_layers=2), "quantum_first"),
        (ClassicalFirst(input_dim=2, n_qubits=4, n_layers=2), "classical_first"),
    ]

    for model, name in models:
        model.train(
            X_t,
            y_t,
            exp_id=EXP_ID,
            model_name=name,
            epochs=cfg["epochs"],
            lr=cfg["learning_rate"],
        )

    log_event("info", "experiment run finished", exp_id=EXP_ID)
