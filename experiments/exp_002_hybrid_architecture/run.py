"""
EXP 002 — Hybrid Architecture
Holdout evaluation; 10 seeds, bootstrap CI, Wilcoxon vs best hybrid.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.generators import make_binary_classification
from src.data.splits import split_train_test
from src.quantum.hybrid_model import ClassicalFirst, HybridSandwich, QuantumFirst
from src.training.config import load_experiment_config
from src.training.holdout import compare_conditions, summarize_multi_seed, train_with_holdout
from src.training.protocol import log_experiment_protocol
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_002_hybrid_architecture"
EXP_ID = "exp_002"

ARCHITECTURES = {
    "HybridSandwich": ("hybrid_sandwich", HybridSandwich),
    "QuantumFirst": ("quantum_first", QuantumFirst),
    "ClassicalFirst": ("classical_first", ClassicalFirst),
}


CLS_BY_KEY = {key: cls for key, cls in ARCHITECTURES.values()}


def build_hybrid(arch_key: str, cfg: dict):
    model_cfg = cfg.get("model_configs", {}).get(arch_key, {})
    n_qubits = model_cfg.get("n_qubits", 4)
    n_layers = model_cfg.get("n_layers", 2)
    lr = model_cfg.get("learning_rate", cfg["learning_rate"])
    cls = CLS_BY_KEY.get(arch_key)
    if cls is None:
        raise ValueError(f"Unknown architecture key: {arch_key}")
    return cls(input_dim=2, n_qubits=n_qubits, n_layers=n_layers), lr


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    seeds = cfg.get("seeds", [cfg["random_state"]])
    arch_entries = [
        ARCHITECTURES[a] for a in cfg.get("architectures", list(ARCHITECTURES)) if a in ARCHITECTURES
    ]
    arch_names = [key for key, _ in arch_entries]
    log_experiment_protocol(EXP_ID, cfg)
    log_event("info", "experiment run started", exp_id=EXP_ID, seeds=seeds)

    results_by_model: dict[str, list[float]] = {name: [] for name in arch_names}

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

        for name in arch_names:
            model, lr = build_hybrid(name, cfg)
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
    if "hybrid_sandwich" in results_by_model and "classical_first" in results_by_model:
        compare_conditions(
            EXP_ID,
            results_by_model["hybrid_sandwich"],
            results_by_model["classical_first"],
            "hybrid_sandwich",
            "classical_first",
        )
    if "quantum_first" in results_by_model and "classical_first" in results_by_model:
        compare_conditions(
            EXP_ID,
            results_by_model["quantum_first"],
            results_by_model["classical_first"],
            "quantum_first",
            "classical_first",
        )
    log_event("info", "experiment run finished", exp_id=EXP_ID)
