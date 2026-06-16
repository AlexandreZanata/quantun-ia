"""
EXP 005 — Quantum Curriculum Learning
Multi-seed comparison: margin_batches vs random with paired Wilcoxon test.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.data.generators import make_binary_classification
from src.data.splits import split_train_test
from src.quantum.qnn_basic import QuantumNetBasic
from src.training.config import load_experiment_config
from src.training.curriculum import sort_by_difficulty, train_curriculum_batched
from src.training.holdout import compare_conditions, summarize_multi_seed
from src.training.protocol import log_experiment_protocol, task_learnable
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_005_curriculum_quantum"
EXP_ID = "exp_005"


def run_method(method: str, cfg: dict, X_train, y_train, X_test, y_test, seed: int) -> float:
    n_qubits = cfg.get("n_qubits", 4)
    n_layers = cfg.get("n_layers", 1)
    lr = cfg.get("learning_rate", 0.01)
    model = QuantumNetBasic(n_qubits=n_qubits, n_layers=n_layers, input_dim=2)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)

    if method == "margin_batches":
        result = train_curriculum_batched(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            exp_id=EXP_ID,
            model_name=f"curriculum_margin_batches_seed{seed}",
            n_stages=cfg["curriculum_stages"],
            epochs_per_stage=cfg["epochs_per_stage"],
            lr=lr,
            refine_epochs=cfg.get("refine_epochs", 12),
        )
        return result["test_accuracy"]

    X_sorted, y_sorted = sort_by_difficulty(X_train, y_train, method=method)
    model.train(
        torch.tensor(X_sorted),
        torch.tensor(y_sorted),
        exp_id=EXP_ID,
        model_name=f"curriculum_{method}_seed{seed}",
        epochs=cfg["epochs"],
        lr=lr,
        X_test=X_test_t,
        y_test=y_test_t,
    )
    return model.evaluate(X_test_t, y_test_t)["accuracy"]


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    seeds = cfg.get("seeds", [cfg["random_state"]])
    log_experiment_protocol(EXP_ID, cfg)
    log_event("info", "experiment run started", exp_id=EXP_ID, seeds=seeds)

    results_by_method: dict[str, list[float]] = {m: [] for m in cfg["methods"]}

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

        for method in cfg["methods"]:
            acc = run_method(method, cfg, X_train, y_train, X_test, y_test, seed)
            results_by_method[method].append(acc)
            log_event(
                "info",
                "curriculum seed result",
                exp_id=EXP_ID,
                method=method,
                seed=seed,
                test_accuracy=acc,
            )

    renamed = {f"curriculum_{m}": v for m, v in results_by_method.items()}
    summarize_multi_seed(EXP_ID, renamed)

    if "margin_batches" in results_by_method and "random" in results_by_method:
        compare_conditions(
            EXP_ID,
            results_by_method["margin_batches"],
            results_by_method["random"],
            "margin_batches",
            "random",
        )

    threshold = cfg.get("learnability_threshold", 0.55)
    learnable = task_learnable(results_by_method.get("random", []), threshold)
    log_event(
        "info",
        "curriculum learnability gate",
        exp_id=EXP_ID,
        task_learnable=learnable,
        threshold=threshold,
        random_mean=sum(results_by_method.get("random", [])) / max(len(results_by_method.get("random", [])), 1),
    )

    log_event("info", "experiment run finished", exp_id=EXP_ID)
