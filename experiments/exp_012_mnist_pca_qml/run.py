"""
EXP 012 — MNIST PCA QML: angle vs amplitude encoding.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset_registry import prepare_dataset
from src.quantum.qnn_amplitude import QuantumNetAmplitude
from src.quantum.qnn_basic import QuantumNetBasic
from src.training.config import load_experiment_config
from src.training.holdout import compare_conditions_batch, summarize_multi_seed, train_with_holdout
from src.training.protocol import log_experiment_protocol
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_012_mnist_pca_qml"
EXP_ID = "exp_012"


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    seeds = cfg.get("seeds", [cfg["random_state"]])
    n_qubits = cfg.get("n_qubits", 4)
    n_layers = cfg.get("n_layers", 2)
    n_components = cfg.get("n_components", 8)
    log_experiment_protocol(EXP_ID, cfg)
    log_event("info", "experiment run started", exp_id=EXP_ID, seeds=seeds)

    results_by_model: dict[str, list[float]] = {"quantum_angle": [], "quantum_amplitude": []}

    for seed in seeds:
        X_train, X_test, y_train, y_test, meta = prepare_dataset(
            "mnist_binary",
            random_state=seed,
            test_size=cfg["test_size"],
            n_samples=cfg.get("n_samples", 500),
            n_components=n_components,
        )
        input_dim = X_train.shape[1]
        log_event(
            "info",
            "pca prepared",
            exp_id=EXP_ID,
            seed=seed,
            n_components=n_components,
            input_dim=input_dim,
            pca_explained=sum(meta.get("pca").explained_variance_ratio_) if meta.get("pca") else None,
        )

        models = {
            "quantum_angle": (
                QuantumNetBasic(n_qubits=n_qubits, n_layers=n_layers, input_dim=input_dim),
                cfg.get("model_configs", {}).get("quantum_angle", {}).get("learning_rate", 0.02),
            ),
            "quantum_amplitude": (
                QuantumNetAmplitude(n_qubits=n_qubits, n_layers=n_layers, input_dim=input_dim),
                cfg.get("model_configs", {}).get("quantum_amplitude", {}).get("learning_rate", 0.02),
            ),
        }

        for name, (model, lr) in models.items():
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
                seed=seed,
                profile=cfg.get("profile"),
            )
            results_by_model[name].append(metrics["accuracy"])

    summarize_multi_seed(EXP_ID, results_by_model)
    compare_conditions_batch(
        EXP_ID,
        [
            {
                "label_a": "quantum_amplitude",
                "label_b": "quantum_angle",
                "condition_a": results_by_model["quantum_amplitude"],
                "condition_b": results_by_model["quantum_angle"],
            }
        ],
    )
    log_event("info", "experiment run finished", exp_id=EXP_ID)
