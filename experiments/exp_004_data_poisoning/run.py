"""
EXP 004 — Dataset Poisoning
Train on poisoned labels; evaluate on clean holdout test set.
Re-upload vs amplitude encoding with Holm-corrected Wilcoxon tests.
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
from src.quantum.qnn_factory import build_qnn
from src.training.config import load_experiment_config
from src.training.holdout import compare_conditions_batch, summarize_multi_seed
from src.training.protocol import log_experiment_protocol
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_004_data_poisoning"
EXP_ID = "exp_004"


def build_quantum(encoding: str, cfg: dict):
    model_cfg = cfg.get("model_configs", {}).get(encoding, {})
    lr = model_cfg.get("learning_rate", cfg["learning_rate"])
    if encoding == "reupload":
        qcfg = {**cfg, **model_cfg, "qnn_type": "reupload"}
        return build_qnn(qcfg), lr
    if encoding == "amplitude":
        n_qubits = model_cfg.get("n_qubits", 4)
        n_layers = model_cfg.get("n_layers", 2)
        return QuantumNetAmplitude(n_qubits=n_qubits, n_layers=n_layers, input_dim=2), lr
    raise ValueError(f"Unknown encoding: {encoding}")


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    seeds = cfg.get("seeds", [cfg["random_state"]])
    log_experiment_protocol(EXP_ID, cfg)
    log_event("info", "experiment run started", exp_id=EXP_ID, seeds=seeds)

    classical_by_rate: dict[float, list[float]] = {r: [] for r in cfg["poison_rates"]}
    quantum_by_encoding: dict[str, dict[float, list[float]]] = {
        enc: {r: [] for r in cfg["poison_rates"]} for enc in cfg["encodings"]
    }

    for seed in seeds:
        X, y, _ = make_binary_classification(
            n_samples=cfg["n_samples"],
            dataset=cfg["dataset"],
            noise=cfg["noise"],
            random_state=seed,
        )
        X_train, X_test, y_train, y_test = split_train_test(
            X, y, test_size=cfg["test_size"], random_state=seed
        )
        X_test_t = torch.tensor(X_test)
        y_test_t = torch.tensor(y_test)

        for rate in cfg["poison_rates"]:
            _, y_train_poisoned, _ = poison_dataset(X_train, y_train, poison_rate=rate)
            X_train_t = torch.tensor(X_train)
            y_train_t = torch.tensor(y_train_poisoned)

            classical = ClassicalNet(hidden=16)
            classical.train(
                X_train_t,
                y_train_t,
                exp_id=EXP_ID,
                model_name=f"classical_poison_{int(rate * 100)}_seed{seed}",
                epochs=cfg["epochs"],
                lr=cfg["learning_rate"],
                X_test=X_test_t,
                y_test=y_test_t,
            )
            classical_by_rate[rate].append(classical.evaluate(X_test_t, y_test_t)["accuracy"])

            for encoding in cfg["encodings"]:
                model, lr = build_quantum(encoding, cfg)
                model.train(
                    X_train_t,
                    y_train_t,
                    exp_id=EXP_ID,
                    model_name=f"quantum_{encoding}_poison_{int(rate * 100)}_seed{seed}",
                    epochs=cfg["epochs"],
                    lr=lr,
                    X_test=X_test_t,
                    y_test=y_test_t,
                )
                acc = model.evaluate(X_test_t, y_test_t)["accuracy"]
                quantum_by_encoding[encoding][rate].append(acc)

    summary_at_rates = {}
    for rate in [0.0, 0.3]:
        summary_at_rates[f"classical_poison_{int(rate * 100)}"] = classical_by_rate[rate]
        for encoding in cfg["encodings"]:
            key = f"quantum_{encoding}_poison_{int(rate * 100)}"
            summary_at_rates[key] = quantum_by_encoding[encoding][rate]

    summarize_multi_seed(EXP_ID, summary_at_rates)

    compare_conditions_batch(
        EXP_ID,
        [
            {
                "label_a": "classical_poison_0",
                "label_b": "quantum_reupload_poison_0",
                "condition_a": classical_by_rate[0.0],
                "condition_b": quantum_by_encoding["reupload"][0.0],
            },
            {
                "label_a": "classical_poison_30",
                "label_b": "quantum_reupload_poison_30",
                "condition_a": classical_by_rate[0.3],
                "condition_b": quantum_by_encoding["reupload"][0.3],
            },
            {
                "label_a": "quantum_amplitude_poison_0",
                "label_b": "quantum_reupload_poison_0",
                "condition_a": quantum_by_encoding["amplitude"][0.0],
                "condition_b": quantum_by_encoding["reupload"][0.0],
            },
        ],
    )

    classical_results = {r: sum(v) / len(v) for r, v in classical_by_rate.items()}
    quantum_results = {
        enc: {r: sum(v) / len(v) for r, v in rates.items()}
        for enc, rates in quantum_by_encoding.items()
    }

    log_event(
        "info",
        "poisoning robustness summary",
        exp_id=EXP_ID,
        classical=measure_robustness(classical_results),
        quantum_reupload=measure_robustness(quantum_results["reupload"]),
        quantum_amplitude=measure_robustness(quantum_results["amplitude"]),
    )
    log_event("info", "experiment run finished", exp_id=EXP_ID)
