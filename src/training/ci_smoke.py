"""Fast CI smoke run for exp_001 reproducibility checks."""

from __future__ import annotations

from pathlib import Path

from src.classical.mlp import ClassicalNet
from src.data.generators import make_binary_classification
from src.data.splits import split_train_test
from src.training import metrics as metrics_module
from src.training.config import load_experiment_config
from src.training.holdout import train_with_holdout

EXP_KEY = "exp_001_quantum_vs_classical"
EXP_ID = "exp_001"
CI_MODEL = "classical_8"


def run_exp_001_ci(*, log_path: Path | None = None, models: list[str] | None = None) -> dict[str, list[float]]:
    """
    Run a minimal exp_001 holdout loop under the ``ci`` profile.

    Defaults to ``classical_8`` only for speed. Pass ``models`` to override.
    """
    original_log_path = metrics_module.LOGS_PATH
    if log_path is not None:
        metrics_module.LOGS_PATH = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_experiment_config(EXP_KEY, profile="ci")
    model_names = models or [CI_MODEL]
    results: dict[str, list[float]] = {name: [] for name in model_names}

    try:
        for seed in cfg["seeds"]:
            X, y, _ = make_binary_classification(
                n_samples=cfg["n_samples"],
                dataset=cfg["dataset"],
                noise=cfg["noise"],
                random_state=seed,
            )
            X_train, X_test, y_train, y_test = split_train_test(
                X, y, test_size=cfg["test_size"], random_state=seed
            )

            for name in model_names:
                lr = cfg.get("model_configs", {}).get(name, {}).get("learning_rate", cfg["learning_rate"])
                hidden = 8 if name == "classical_8" else 32
                model = ClassicalNet(hidden=hidden)
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
                    seed=seed,
                    profile=cfg.get("profile"),
                )
                results[name].append(metrics["accuracy"])
    finally:
        metrics_module.LOGS_PATH = original_log_path

    return results
