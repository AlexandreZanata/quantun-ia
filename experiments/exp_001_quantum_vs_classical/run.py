"""
EXP 001 — Quantum vs Classical
Holdout evaluation on 30% test split; repeated across 10 seeds with bootstrap CI.
Includes data re-uploading QNN (exp_008 follow-up).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.classical.mlp import ClassicalNet
from src.data.generators import make_binary_classification
from src.data.splits import split_train_test
from src.quantum.qnn_basic import QuantumNetBasic
from src.quantum.qnn_reupload import QuantumNetReupload
from src.training.config import load_experiment_config
from src.training.holdout import compare_conditions, summarize_multi_seed, train_with_holdout
from src.training.protocol import log_experiment_protocol
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_001_quantum_vs_classical"
EXP_ID = "exp_001"


def build_model(name: str, cfg: dict):
    model_cfg = cfg.get("model_configs", {}).get(name, {})
    lr = model_cfg.get("learning_rate", cfg["learning_rate"])

    if name == "classical_8":
        return ClassicalNet(hidden=8), lr
    if name == "classical_32":
        return ClassicalNet(hidden=32), lr
    if name == "quantum_4q_2l":
        return (
            QuantumNetBasic(
                n_qubits=model_cfg.get("n_qubits", 4),
                n_layers=model_cfg.get("n_layers", 1),
                input_dim=2,
            ),
            lr,
        )
    if name == "quantum_6q_3l":
        return (
            QuantumNetBasic(
                n_qubits=model_cfg.get("n_qubits", 6),
                n_layers=model_cfg.get("n_layers", 3),
                input_dim=2,
            ),
            lr,
        )
    if name == "quantum_reupload_4q_3l":
        return (
            QuantumNetReupload(
                n_qubits=model_cfg.get("n_qubits", 4),
                n_layers=model_cfg.get("n_layers", 3),
                input_dim=2,
            ),
            lr,
        )
    raise ValueError(f"Unknown model: {name}")


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    seeds = cfg.get("seeds", [cfg["random_state"]])
    model_names = cfg.get("models", [])
    log_experiment_protocol(EXP_ID, cfg)
    log_event("info", "experiment run started", exp_id=EXP_ID, seeds=seeds)

    results_by_model: dict[str, list[float]] = {name: [] for name in model_names}

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

        for name in model_names:
            model, lr = build_model(name, cfg)
            metrics = train_with_holdout(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                exp_id=EXP_ID,
                model_name=f"{name}_seed{seed}",
                epochs=cfg["epochs"],
                lr=lr,
            )
            results_by_model[name].append(metrics["accuracy"])

    summarize_multi_seed(EXP_ID, results_by_model)

    if "classical_32" in results_by_model and "quantum_4q_2l" in results_by_model:
        compare_conditions(
            EXP_ID,
            results_by_model["classical_32"],
            results_by_model["quantum_4q_2l"],
            "classical_32",
            "quantum_4q_2l",
        )
    if "quantum_reupload_4q_3l" in results_by_model and "quantum_4q_2l" in results_by_model:
        compare_conditions(
            EXP_ID,
            results_by_model["quantum_reupload_4q_3l"],
            results_by_model["quantum_4q_2l"],
            "quantum_reupload_4q_3l",
            "quantum_4q_2l",
        )
    if "classical_32" in results_by_model and "quantum_reupload_4q_3l" in results_by_model:
        compare_conditions(
            EXP_ID,
            results_by_model["classical_32"],
            results_by_model["quantum_reupload_4q_3l"],
            "classical_32",
            "quantum_reupload_4q_3l",
        )

    log_event("info", "experiment run finished", exp_id=EXP_ID)
