"""
EXP 007 — Quantum Self-Play
Self-play fine-tuning on train pool only; holdout eval with best-checkpoint tracking.
Multi-seed evaluation with bootstrap CI.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time

import torch

from src.data.generators import make_binary_classification
from src.data.splits import split_train_test
from src.quantum.qnn_basic import QuantumNetBasic
from src.training.config import load_experiment_config
from src.training.holdout import compare_conditions, summarize_multi_seed
from src.training.metrics import ExperimentLogger
from src.training.self_play import self_play_train
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_007_self_play"
EXP_ID = "exp_007"


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    seeds = cfg.get("seeds", [cfg["random_state"]])
    log_event("info", "experiment run started", exp_id=EXP_ID, seeds=seeds)

    base_holdouts: list[float] = []
    best_holdouts: list[float] = []

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

        model = QuantumNetBasic(
            n_qubits=cfg.get("n_qubits", 4),
            n_layers=cfg.get("n_layers", 1),
            input_dim=2,
        )
        model.train(
            torch.tensor(X_train),
            torch.tensor(y_train),
            exp_id=EXP_ID,
            model_name=f"self_play_base_seed{seed}",
            epochs=cfg.get("base_epochs", cfg["epochs"]),
            lr=cfg["learning_rate"],
            X_test=torch.tensor(X_test),
            y_test=torch.tensor(y_test),
        )

        t0 = time.time()
        result = self_play_train(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            exp_id=EXP_ID,
            rounds=cfg["rounds"],
            hard_frac=cfg["hard_frac"],
            min_hard=cfg.get("min_hard", 5),
            fine_tune_epochs=cfg.get("fine_tune_epochs", 15),
            lr=cfg["learning_rate"],
            revert_threshold=cfg.get("revert_threshold", 0.05),
        )

        base_holdouts.append(result["base_holdout"])
        best_holdouts.append(result["best_holdout"])

        log = ExperimentLogger(EXP_ID, f"self_play_history_seed{seed}")
        for entry in result["history"]:
            log.log(entry["round"], n_hard=entry["n_hard"], accuracy=entry["acc"])
        log.finish(
            time.time() - t0,
            test_accuracy=result["best_holdout"],
            final_holdout=result["final_holdout"],
            base_holdout=result["base_holdout"],
            eval_set="holdout_test",
            self_play_history=result["history"],
        )

    summarize_multi_seed(
        EXP_ID,
        {"self_play_base": base_holdouts, "self_play_best": best_holdouts},
    )
    compare_conditions(EXP_ID, best_holdouts, base_holdouts, "self_play_best", "self_play_base")

    log_event("info", "experiment run finished", exp_id=EXP_ID)
