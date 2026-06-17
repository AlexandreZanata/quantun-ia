"""
EXP 005 — Quantum Curriculum Learning (re-upload QNN base)
Multi-seed comparison: margin_batches vs random with paired Wilcoxon test.
Skips curriculum when the base QNN does not exceed the learnability threshold.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.data.generators import make_binary_classification
from src.data.splits import split_train_test
from src.quantum.qnn_factory import build_qnn
from src.training.config import load_experiment_config
from src.training.curriculum import (
    curriculum_total_epochs,
    sort_by_difficulty,
    train_curriculum_batched,
)
from src.training.holdout import compare_conditions_batch, summarize_multi_seed
from src.training.protocol import log_applicability_gate, log_experiment_protocol, task_learnable
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_005_curriculum_quantum"
EXP_ID = "exp_005"


def run_random_baseline(cfg: dict, X_train, y_train, X_test, y_test, seed: int) -> float:
    lr = cfg.get("learning_rate", 0.01)
    epochs = curriculum_total_epochs(cfg)
    model = build_qnn(cfg)
    X_sorted, y_sorted = sort_by_difficulty(X_train, y_train, method="random")
    model.train(
        torch.tensor(X_sorted),
        torch.tensor(y_sorted),
        exp_id=EXP_ID,
        model_name=f"curriculum_random_seed{seed}",
        epochs=epochs,
        lr=lr,
        X_test=torch.tensor(X_test),
        y_test=torch.tensor(y_test),
    )
    return model.evaluate(torch.tensor(X_test), torch.tensor(y_test))["accuracy"]


def run_margin_curriculum(cfg: dict, X_train, y_train, X_test, y_test, seed: int) -> float:
    lr = cfg.get("learning_rate", 0.01)
    model = build_qnn(cfg)
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


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    seeds = cfg.get("seeds", [cfg["random_state"]])
    threshold = cfg.get("learnability_threshold", 0.55)
    log_experiment_protocol(EXP_ID, cfg)
    log_event(
        "info",
        "experiment run started",
        exp_id=EXP_ID,
        seeds=seeds,
        qnn_type=cfg.get("qnn_type", "basic"),
    )

    random_holdouts: list[float] = []
    margin_holdouts: list[float] = []

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
        acc = run_random_baseline(cfg, X_train, y_train, X_test, y_test, seed)
        random_holdouts.append(acc)
        log_event(
            "info",
            "curriculum seed result",
            exp_id=EXP_ID,
            method="random",
            seed=seed,
            test_accuracy=acc,
        )

    mean_random = sum(random_holdouts) / len(random_holdouts)
    learnable = task_learnable(random_holdouts, threshold)
    log_applicability_gate(
        EXP_ID,
        "curriculum",
        learnable,
        threshold=threshold,
        mean_holdout=mean_random,
        reason="re-upload QNN random baseline holdout vs learnability threshold",
    )

    if learnable:
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
            acc = run_margin_curriculum(cfg, X_train, y_train, X_test, y_test, seed)
            margin_holdouts.append(acc)
            log_event(
                "info",
                "curriculum seed result",
                exp_id=EXP_ID,
                method="margin_batches",
                seed=seed,
                test_accuracy=acc,
            )
    else:
        log_event(
            "info",
            "curriculum skipped",
            exp_id=EXP_ID,
            reason="task not learnable",
            margin_batches_status="N/A",
        )

    summary: dict[str, list[float]] = {"curriculum_random": random_holdouts}
    if margin_holdouts:
        summary["curriculum_margin_batches"] = margin_holdouts
    summarize_multi_seed(EXP_ID, summary)

    if margin_holdouts:
        compare_conditions_batch(
            EXP_ID,
            [
                {
                    "label_a": "margin_batches",
                    "label_b": "random",
                    "condition_a": margin_holdouts,
                    "condition_b": random_holdouts,
                },
            ],
        )

    log_event("info", "experiment run finished", exp_id=EXP_ID)
