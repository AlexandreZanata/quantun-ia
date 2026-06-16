"""
EXP 002 — Hybrid Architecture
Holdout evaluation on 30% test split; repeated across 3 seeds.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.generators import make_binary_classification
from src.data.splits import split_train_test
from src.quantum.hybrid_model import ClassicalFirst, HybridSandwich, QuantumFirst
from src.training.config import load_experiment_config
from src.training.holdout import summarize_multi_seed, train_with_holdout
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_002_hybrid_architecture"
EXP_ID = "exp_002"

MODEL_BUILDERS = {
    "hybrid_sandwich": lambda: HybridSandwich(input_dim=2, n_qubits=4, n_layers=2),
    "quantum_first": lambda: QuantumFirst(input_dim=2, n_qubits=4, n_layers=2),
    "classical_first": lambda: ClassicalFirst(input_dim=2, n_qubits=4, n_layers=2),
}


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    seeds = cfg.get("seeds", [cfg["random_state"]])
    log_event("info", "experiment run started", exp_id=EXP_ID, seeds=seeds)

    results_by_model: dict[str, list[float]] = {name: [] for name in MODEL_BUILDERS}

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

        for name, builder in MODEL_BUILDERS.items():
            metrics = train_with_holdout(
                builder(),
                X_train,
                y_train,
                X_test,
                y_test,
                exp_id=EXP_ID,
                model_name=f"{name}_seed{seed}",
                epochs=cfg["epochs"],
                lr=cfg["learning_rate"],
            )
            results_by_model[name].append(metrics["accuracy"])

    summarize_multi_seed(EXP_ID, results_by_model)
    log_event("info", "experiment run finished", exp_id=EXP_ID)
