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

EXP_011_KEY = "exp_011_uci_tabular_qml"
EXP_011_ID = "exp_011"
EXP_011_MODEL = "perceptron"


def _run_holdout_loop(
    *,
    exp_key: str,
    exp_id: str,
    cfg: dict,
    model_names: list[str],
    build_model,
    log_path: Path | None,
) -> dict[str, list[float]]:
    original_log_path = metrics_module.LOGS_PATH
    if log_path is not None:
        metrics_module.LOGS_PATH = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[float]] = {name: [] for name in model_names}

    try:
        for seed in cfg["seeds"]:
            data = build_model(seed)
            X_train, X_test, y_train, y_test = data["splits"]
            for name in model_names:
                model, lr = data["models"][name]
                metrics = train_with_holdout(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    exp_id=exp_id,
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


def run_exp_001_ci(*, log_path: Path | None = None, models: list[str] | None = None) -> dict[str, list[float]]:
    """
    Run a minimal exp_001 holdout loop under the ``ci`` profile.

    Defaults to ``classical_8`` only for speed. Pass ``models`` to override.
    """
    cfg = load_experiment_config(EXP_KEY, profile="ci")
    model_names = models or [CI_MODEL]

    def build_model(seed: int) -> dict:
        X, y, _ = make_binary_classification(
            n_samples=cfg["n_samples"],
            dataset=cfg["dataset"],
            noise=cfg["noise"],
            random_state=seed,
        )
        X_train, X_test, y_train, y_test = split_train_test(
            X, y, test_size=cfg["test_size"], random_state=seed
        )
        models_map: dict[str, tuple] = {}
        for name in model_names:
            lr = cfg.get("model_configs", {}).get(name, {}).get("learning_rate", cfg["learning_rate"])
            hidden = 8 if name == "classical_8" else 32
            models_map[name] = (ClassicalNet(hidden=hidden), lr)
        return {
            "splits": (X_train, X_test, y_train, y_test),
            "models": models_map,
        }

    return _run_holdout_loop(
        exp_key=EXP_KEY,
        exp_id=EXP_ID,
        cfg=cfg,
        model_names=model_names,
        build_model=build_model,
        log_path=log_path,
    )


def run_exp_011_ci(*, log_path: Path | None = None) -> dict[str, list[float]]:
    """Fast CI smoke for exp_011 (perceptron on breast cancer, ci profile)."""
    from src.classical.perceptron import Perceptron
    from src.data.dataset_registry import prepare_dataset

    cfg = load_experiment_config(EXP_011_KEY, profile="ci")
    model_names = [EXP_011_MODEL]

    def build_model(seed: int) -> dict:
        X_train, X_test, y_train, y_test, _meta = prepare_dataset(
            "breast_cancer",
            test_size=cfg["test_size"],
            random_state=seed,
            scale=True,
        )
        mc = cfg.get("model_configs", {})
        lr = mc.get("perceptron", {}).get("learning_rate", cfg["learning_rate"])
        input_dim = X_train.shape[1]
        return {
            "splits": (X_train, X_test, y_train, y_test),
            "models": {EXP_011_MODEL: (Perceptron(input_dim=input_dim), lr)},
        }

    return _run_holdout_loop(
        exp_key=EXP_011_KEY,
        exp_id=EXP_011_ID,
        cfg=cfg,
        model_names=model_names,
        build_model=build_model,
        log_path=log_path,
    )
