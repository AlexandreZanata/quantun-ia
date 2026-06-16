"""Standard experiment protocol logging for reproducible research."""

from __future__ import annotations

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
