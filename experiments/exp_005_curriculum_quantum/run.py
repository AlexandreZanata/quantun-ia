"""
EXP 005 — Quantum Curriculum Learning
Write your hypothesis in hypothesis.md BEFORE running this script.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.data.generators import make_binary_classification
from src.quantum.qnn_basic import QuantumNetBasic
from src.training.config import load_experiment_config
from src.training.curriculum import sort_by_difficulty
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_005_curriculum_quantum"
EXP_ID = "exp_005"


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

    for method in cfg["methods"]:
        X_sorted, y_sorted = sort_by_difficulty(X, y, method=method)
        model = QuantumNetBasic(n_qubits=4, n_layers=2, input_dim=2)
        model.train(
            torch.tensor(X_sorted),
            torch.tensor(y_sorted),
            exp_id=EXP_ID,
            model_name=f"curriculum_{method}",
            epochs=cfg["epochs"],
            lr=cfg["learning_rate"],
        )

    log_event("info", "experiment run finished", exp_id=EXP_ID)
