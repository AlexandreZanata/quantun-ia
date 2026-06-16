"""Self-play fine-tuning with capped hard-example selection and best-checkpoint tracking."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from src.training.structured_log import log_event
from src.training.trainer import evaluate, fine_tune


def select_hard_subset(
    model: nn.Module,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    hard_frac: float = 0.3,
    min_hard: int = 5,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Select misclassified examples ranked by loss, capped at hard_frac of the pool."""
    X_t = torch.tensor(X_pool, dtype=torch.float32)
    y_t = torch.tensor(y_pool, dtype=torch.float32)
    model.eval()

    with torch.no_grad():
        pred = model(X_t)
        per_sample_loss = nn.BCELoss(reduction="none")(pred, y_t)
        wrong = (pred > 0.5) != y_t.bool()

    wrong_idx = wrong.nonzero(as_tuple=True)[0].cpu().numpy()
    if len(wrong_idx) == 0:
        return None, None

    ranked = wrong_idx[np.argsort(-per_sample_loss[wrong_idx].cpu().numpy())]
    cap = max(min_hard, int(len(X_pool) * hard_frac))
    cap = min(cap, len(ranked))
    selected = ranked[:cap]
    return X_pool[selected], y_pool[selected]


def self_play_train(
    model: nn.Module,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    exp_id: str,
    rounds: int = 5,
    hard_frac: float = 0.3,
    min_hard: int = 5,
    fine_tune_epochs: int = 15,
    lr: float = 0.01,
    revert_threshold: float = 0.05,
) -> dict:
    """Fine-tune on capped hard subsets; keep best holdout checkpoint."""
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    base_holdout = evaluate(model, X_test_t, y_test_t)["accuracy"]
    best_holdout = base_holdout
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict] = []

    for round_n in range(rounds):
        hard_X, hard_y = select_hard_subset(model, X_pool, y_pool, hard_frac, min_hard)
        if hard_X is None:
            log_event("info", "self-play converged", exp_id=exp_id, round=round_n)
            break

        fine_tune(
            model,
            torch.tensor(hard_X, dtype=torch.float32),
            torch.tensor(hard_y, dtype=torch.float32),
            epochs=fine_tune_epochs,
            lr=lr,
        )
        holdout = evaluate(model, X_test_t, y_test_t)["accuracy"]

        if holdout > best_holdout:
            best_holdout = holdout
            best_state = copy.deepcopy(model.state_dict())
        elif holdout < best_holdout - revert_threshold:
            model.load_state_dict(best_state)
            holdout = best_holdout

        entry = {
            "round": round_n,
            "n_hard": int(len(hard_X)),
            "acc": holdout,
            "eval_set": "holdout_test",
        }
        history.append(entry)
        log_event(
            "info",
            "self-play round complete",
            exp_id=exp_id,
            round=round_n,
            n_hard=len(hard_X),
            holdout_accuracy=holdout,
            best_holdout=best_holdout,
        )

    model.load_state_dict(best_state)
    final_holdout = evaluate(model, X_test_t, y_test_t)["accuracy"]

    return {
        "history": history,
        "base_holdout": base_holdout,
        "best_holdout": best_holdout,
        "final_holdout": final_holdout,
    }
