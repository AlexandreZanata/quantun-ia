"""
EXP 013 — Augmentation robustness on noisy circles.
Compares baseline QNN training vs Gaussian-augmented training data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.augmentation import add_gaussian_noise
from src.data.dataset_registry import prepare_dataset
from src.quantum.qnn_basic import QuantumNetBasic
from src.training.config import load_experiment_config
from src.training.holdout import compare_conditions_batch, summarize_multi_seed, train_with_holdout
from src.training.protocol import log_experiment_protocol
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_013_augmentation_robustness"
EXP_ID = "exp_013"


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    seeds = cfg.get("seeds", [cfg["random_state"]])
    augment_sigma = cfg.get("augment_sigma", 0.15)
    log_experiment_protocol(EXP_ID, cfg)
    log_event("info", "experiment run started", exp_id=EXP_ID, seeds=seeds)

    results_by_model: dict[str, list[float]] = {"baseline": [], "augmented": []}

    for seed in seeds:
        X_train, X_test, y_train, y_test, _ = prepare_dataset(
            cfg.get("dataset", "circles"),
            random_state=seed,
            test_size=cfg["test_size"],
            n_samples=cfg["n_samples"],
            noise=cfg.get("noise", 0.2),
        )

        qnn_cfg = cfg.get("model_configs", {}).get("quantum_baseline", {})
        n_qubits = qnn_cfg.get("n_qubits", cfg.get("n_qubits", 4))
        n_layers = qnn_cfg.get("n_layers", cfg.get("n_layers", 2))
        lr = qnn_cfg.get("learning_rate", cfg["learning_rate"])

        baseline_model = QuantumNetBasic(n_qubits=n_qubits, n_layers=n_layers, input_dim=2)
        metrics_base = train_with_holdout(
            baseline_model,
            X_train,
            y_train,
            X_test,
            y_test,
            exp_id=EXP_ID,
            model_name=f"baseline_seed{seed}",
            epochs=cfg["epochs"],
            lr=lr,
            seed=seed,
            profile=cfg.get("profile"),
        )
        results_by_model["baseline"].append(metrics_base["accuracy"])

        X_train_aug = add_gaussian_noise(X_train, sigma=augment_sigma, seed=seed)
        aug_model = QuantumNetBasic(n_qubits=n_qubits, n_layers=n_layers, input_dim=2)
        metrics_aug = train_with_holdout(
            aug_model,
            X_train_aug,
            y_train,
            X_test,
            y_test,
            exp_id=EXP_ID,
            model_name=f"augmented_seed{seed}",
            epochs=cfg["epochs"],
            lr=lr,
            seed=seed,
            profile=cfg.get("profile"),
        )
        results_by_model["augmented"].append(metrics_aug["accuracy"])

    summarize_multi_seed(EXP_ID, results_by_model)
    compare_conditions_batch(
        EXP_ID,
        [
            {
                "label_a": "augmented",
                "label_b": "baseline",
                "condition_a": results_by_model["augmented"],
                "condition_b": results_by_model["baseline"],
            }
        ],
    )
    log_event("info", "experiment run finished", exp_id=EXP_ID)
