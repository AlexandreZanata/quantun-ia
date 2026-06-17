"""
EXP 007 — Quantum Self-Play (re-upload QNN base)
Self-play fine-tuning on train pool only; holdout eval with best-checkpoint tracking.
Skips self-play rounds when the base QNN does not exceed the learnability threshold.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time

import torch

from src.data.generators import make_binary_classification
from src.data.splits import split_train_test
from src.quantum.qnn_factory import build_qnn
from src.training.config import load_experiment_config
from src.training.holdout import compare_conditions_batch, summarize_multi_seed
from src.training.metrics import ExperimentLogger
from src.training.protocol import log_applicability_gate, log_experiment_protocol, task_learnable
from src.training.self_play import self_play_train
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_007_self_play"
EXP_ID = "exp_007"


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

    base_holdouts: list[float] = []
    best_holdouts: list[float] = []
    trained_models: list[tuple] = []

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

        model = build_qnn(cfg)
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
        base_acc = model.evaluate(torch.tensor(X_test), torch.tensor(y_test))["accuracy"]
        base_holdouts.append(base_acc)
        trained_models.append((model, X_train, y_train, X_test, y_test, seed))

    mean_base = sum(base_holdouts) / len(base_holdouts)
    applicable = task_learnable(base_holdouts, threshold)
    log_applicability_gate(
        EXP_ID,
        "self_play",
        applicable,
        threshold=threshold,
        mean_holdout=mean_base,
        reason="re-upload QNN base holdout vs learnability threshold",
    )

    if applicable:
        for model, X_train, y_train, X_test, y_test, seed in trained_models:
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
    else:
        best_holdouts = list(base_holdouts)
        log_event(
            "info",
            "self-play skipped",
            exp_id=EXP_ID,
            reason="task not learnable",
            self_play_status="N/A",
        )

    summarize_multi_seed(
        EXP_ID,
        {"self_play_base": base_holdouts, "self_play_best": best_holdouts},
    )
    if applicable:
        compare_conditions_batch(
            EXP_ID,
            [
                {
                    "label_a": "self_play_best",
                    "label_b": "self_play_base",
                    "condition_a": best_holdouts,
                    "condition_b": base_holdouts,
                },
            ],
        )

    log_event("info", "experiment run finished", exp_id=EXP_ID)
