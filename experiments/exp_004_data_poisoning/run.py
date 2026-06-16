"""
EXP 004 — Dataset Poisoning
Write your hypothesis in hypothesis.md BEFORE running this script.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.classical.mlp import ClassicalNet
from src.data.generators import make_binary_classification
from src.data.poisoning import measure_robustness, poison_dataset
from src.quantum.qnn_basic import QuantumNetBasic
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_004_data_poisoning"
EXP_ID = "exp_004"


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
    classical_results = {}
    quantum_results = {}

    for rate in cfg["poison_rates"]:
        _, y_poisoned, _ = poison_dataset(X, y, poison_rate=rate)
        X_t = torch.tensor(X)
        y_t = torch.tensor(y_poisoned)

        classical = ClassicalNet(hidden=16)
        classical.train(
            X_t,
            y_t,
            exp_id=EXP_ID,
            model_name=f"classical_poison_{int(rate * 100)}",
            epochs=30,
            lr=cfg["learning_rate"],
        )
        classical_results[rate] = classical.evaluate(X_t, y_t)["accuracy"]

        quantum = QuantumNetBasic(n_qubits=4, n_layers=2, input_dim=2)
        quantum.train(
            X_t,
            y_t,
            exp_id=EXP_ID,
            model_name=f"quantum_poison_{int(rate * 100)}",
            epochs=30,
            lr=cfg["learning_rate"],
        )
        quantum_results[rate] = quantum.evaluate(X_t, y_t)["accuracy"]

    log_event(
        "info",
        "poisoning robustness summary",
        exp_id=EXP_ID,
        classical=measure_robustness(classical_results),
        quantum=measure_robustness(quantum_results),
    )
    log_event("info", "experiment run finished", exp_id=EXP_ID)
