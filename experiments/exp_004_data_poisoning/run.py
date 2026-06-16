"""
EXP 004 — Dataset Poisoning
Train on poisoned labels; evaluate on clean holdout test set.
Compare angle vs amplitude encoding under label noise.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.classical.mlp import ClassicalNet
from src.data.generators import make_binary_classification
from src.data.poisoning import measure_robustness, poison_dataset
from src.data.splits import split_train_test
from src.quantum.qnn_amplitude import QuantumNetAmplitude
from src.quantum.qnn_basic import QuantumNetBasic
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_004_data_poisoning"
EXP_ID = "exp_004"

QUANTUM_MODELS = {
    "angle": lambda: QuantumNetBasic(n_qubits=4, n_layers=2, input_dim=2),
    "amplitude": lambda: QuantumNetAmplitude(n_qubits=2, n_layers=2, input_dim=2),
}


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
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=cfg["test_size"], random_state=cfg["random_state"]
    )
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)

    classical_results = {}
    quantum_results = {enc: {} for enc in cfg["encodings"]}

    for rate in cfg["poison_rates"]:
        _, y_train_poisoned, _ = poison_dataset(X_train, y_train, poison_rate=rate)
        X_train_t = torch.tensor(X_train)
        y_train_t = torch.tensor(y_train_poisoned)

        classical = ClassicalNet(hidden=16)
        classical.train(
            X_train_t,
            y_train_t,
            exp_id=EXP_ID,
            model_name=f"classical_poison_{int(rate * 100)}",
            epochs=30,
            lr=cfg["learning_rate"],
            X_test=X_test_t,
            y_test=y_test_t,
        )
        classical_results[rate] = classical.evaluate(X_test_t, y_test_t)["accuracy"]

        for encoding in cfg["encodings"]:
            quantum = QUANTUM_MODELS[encoding]()
            quantum.train(
                X_train_t,
                y_train_t,
                exp_id=EXP_ID,
                model_name=f"quantum_{encoding}_poison_{int(rate * 100)}",
                epochs=30,
                lr=cfg["learning_rate"],
                X_test=X_test_t,
                y_test=y_test_t,
            )
            quantum_results[encoding][rate] = quantum.evaluate(X_test_t, y_test_t)["accuracy"]

    log_event(
        "info",
        "poisoning robustness summary",
        exp_id=EXP_ID,
        classical=measure_robustness(classical_results),
        quantum_angle=measure_robustness(quantum_results["angle"]),
        quantum_amplitude=measure_robustness(quantum_results["amplitude"]),
    )
    log_event("info", "experiment run finished", exp_id=EXP_ID)
