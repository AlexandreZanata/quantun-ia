"""
EXP 007 — Quantum Self-Play
Write your hypothesis in hypothesis.md BEFORE running this script.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time

import torch

from src.data.generators import make_binary_classification
from src.quantum.qnn_basic import QuantumNetBasic
from src.training.config import load_experiment_config
from src.training.metrics import ExperimentLogger
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import fine_tune

EXP_KEY = "exp_007_self_play"
EXP_ID = "exp_007"


def self_play_loop(model, X_pool, y_pool, rounds=5):
    history = []
    for round_n in range(rounds):
        model.eval()
        with torch.no_grad():
            preds = model.predict(torch.tensor(X_pool, dtype=torch.float32))
            errors = ((preds > 0.5) != torch.tensor(y_pool).bool()).numpy()

        hard_X = X_pool[errors]
        hard_y = y_pool[errors]

        if len(hard_X) == 0:
            log_event("info", "self-play converged", exp_id=EXP_ID, round=round_n)
            break

        acc = fine_tune(
            model,
            torch.tensor(hard_X, dtype=torch.float32),
            torch.tensor(hard_y, dtype=torch.float32),
            epochs=20,
        )
        history.append({"round": round_n, "n_hard": int(len(hard_X)), "acc": acc})
        log_event(
            "info",
            "self-play round complete",
            exp_id=EXP_ID,
            round=round_n,
            n_hard=len(hard_X),
            accuracy=acc,
        )

    return history


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    log_event("info", "experiment run started", exp_id=EXP_ID)

    X, y, _ = make_binary_classification(n_samples=300)
    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    model = QuantumNetBasic(n_qubits=4, n_layers=2, input_dim=2)
    model.train(
        X_t,
        y_t,
        exp_id=EXP_ID,
        model_name="self_play_base",
        epochs=30,
        lr=cfg["learning_rate"],
    )

    t0 = time.time()
    history = self_play_loop(model, X, y, rounds=cfg["rounds"])

    log = ExperimentLogger(EXP_ID, "self_play_history")
    for entry in history:
        log.log(entry["round"], n_hard=entry["n_hard"], accuracy=entry["acc"])
    log.finish(
        time.time() - t0,
        final_acc=history[-1]["acc"] if history else None,
        self_play_history=history,
    )

    log_event("info", "experiment run finished", exp_id=EXP_ID)
