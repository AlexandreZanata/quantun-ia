"""Standard experiment protocol logging for reproducible research."""

from __future__ import annotations

import json
from datetime import datetime

from src.training import metrics as metrics_module
from src.training.structured_log import log_event


def log_experiment_protocol(exp_id: str, cfg: dict) -> dict:
    """Log dataset, sample size, noise, seeds, and split — required for publication."""
    seeds = cfg.get("seeds", [cfg["random_state"]])
    protocol = {
        "dataset": cfg.get("dataset", "moons"),
        "n_samples": cfg.get("n_samples", 300),
        "noise": cfg.get("noise", 0.1),
        "test_size": cfg.get("test_size", 0.3),
        "epochs": cfg.get("epochs", 50),
        "learning_rate": cfg.get("learning_rate", 0.01),
        "n_seeds": len(seeds),
        "seeds": seeds,
    }
    log_event("info", "experiment protocol", exp_id=exp_id, **protocol)
    return protocol


def task_learnable(holdout_accuracies: list[float], threshold: float = 0.55) -> bool:
    """True when mean holdout exceeds chance-level threshold (binary classification)."""
    if not holdout_accuracies:
        return False
    return sum(holdout_accuracies) / len(holdout_accuracies) >= threshold


def log_applicability_gate(
    exp_id: str,
    technique: str,
    applicable: bool,
    *,
    threshold: float,
    mean_holdout: float,
    reason: str = "",
) -> dict:
    """Log whether a training technique (curriculum, self-play) is applicable on this task."""
    status = "applicable" if applicable else "not_applicable"
    payload = {
        "exp_id": exp_id,
        "model_name": f"{exp_id}_{technique}_applicability",
        "record_type": "applicability_gate",
        "technique": technique,
        "status": status,
        "applicable": applicable,
        "threshold": threshold,
        "mean_holdout": mean_holdout,
        "reason": reason or (
            f"mean holdout {mean_holdout:.3f} {'>=' if applicable else '<'} threshold {threshold}"
        ),
        "started_at": datetime.now().isoformat(),
    }
    log_event(
        "info",
        "technique applicability gate",
        exp_id=exp_id,
        technique=technique,
        applicable=applicable,
        status=status,
        mean_holdout=mean_holdout,
        threshold=threshold,
    )
    log_path = metrics_module.LOGS_PATH
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(payload) + "\n")
    return payload
