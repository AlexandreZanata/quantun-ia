"""
EXP 005 — Quantum Curriculum Learning
margin_batches: staged easy→hard training with held-out test eval.
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
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=cfg["test_size"], random_state=cfg["random_state"]
    )
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)

    for method in cfg["methods"]:
        model = QuantumNetBasic(n_qubits=4, n_layers=2, input_dim=2)

        if method == "margin_batches":
            result = train_curriculum_batched(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                exp_id=EXP_ID,
                model_name="curriculum_margin_batches",
                n_stages=cfg["curriculum_stages"],
                epochs_per_stage=cfg["epochs_per_stage"],
                lr=cfg["learning_rate"],
                refine_epochs=cfg.get("refine_epochs", 12),
            )
            log_event(
                "info",
                "curriculum margin_batches summary",
                exp_id=EXP_ID,
                test_accuracy=result["test_accuracy"],
                stage_holdout=[s["holdout_accuracy"] for s in result["stage_metrics"]],
            )
        else:
            X_sorted, y_sorted = sort_by_difficulty(X_train, y_train, method=method)
            model.train(
                torch.tensor(X_sorted),
                torch.tensor(y_sorted),
                exp_id=EXP_ID,
                model_name=f"curriculum_{method}",
                epochs=cfg["epochs"],
                lr=cfg["learning_rate"],
                X_test=X_test_t,
                y_test=y_test_t,
            )

    log_event("info", "experiment run finished", exp_id=EXP_ID)
