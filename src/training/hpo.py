"""Hyperparameter optimization with Optuna."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.training.config import load_experiment_config
from src.training.tracking import RunTracker

HPO_LOG_PATH = Path("logs/hpo_results.jsonl")


def _write_hpo_record(record: dict[str, Any]) -> None:
    HPO_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HPO_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def run_optuna_study(
    exp_key: str,
    objective_fn: Callable[[Any], float],
    *,
    n_trials: int = 50,
    profile: str | None = None,
    direction: str = "maximize",
) -> dict[str, Any]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = load_experiment_config(exp_key, profile=profile)
    study_name = f"{exp_key}_{cfg.get('profile', 'default')}"

    study = optuna.create_study(direction=direction, study_name=study_name)
    study.optimize(objective_fn, n_trials=n_trials)

    best = {
        "exp_key": exp_key,
        "profile": cfg.get("profile"),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "finished_at": datetime.now().isoformat(),
    }
    _write_hpo_record(best)
    return best


def evaluate_uci_trial(
    exp_key: str,
    params: dict[str, float],
    *,
    profile: str | None = None,
    model_name: str = "classical_matched",
) -> float:
    """Objective helper: mean holdout accuracy over HPO seeds for exp_011."""
    from src.classical.mlp import ClassicalNet
    from src.classical.perceptron import Perceptron
    from src.data.dataset_registry import prepare_dataset
    from src.quantum.qnn_basic import QuantumNetBasic
    from src.training.holdout import train_with_holdout
    from src.training.param_match import build_param_matched_classical
    from src.training.trainer import count_parameters

    cfg = load_experiment_config(exp_key, profile=profile)
    dataset = cfg.get("dataset", "breast_cancer")
    seeds = cfg.get("hpo_seeds", cfg.get("seeds", [42])[:3])
    hidden = int(params.get("hidden", cfg.get("hpo_defaults", {}).get("hidden", 16)))
    lr = float(params.get("learning_rate", cfg["learning_rate"]))
    n_qubits = int(params.get("n_qubits", cfg.get("n_qubits", 4)))
    n_layers = int(params.get("n_layers", cfg.get("n_layers", 2)))

    accs: list[float] = []
    for seed in seeds:
        X_train, X_test, y_train, y_test, meta = prepare_dataset(
            dataset,
            random_state=seed,
            test_size=cfg["test_size"],
        )
        input_dim = meta.get("n_features", X_train.shape[1])

        if model_name == "perceptron":
            model = Perceptron(input_dim=input_dim)
        elif model_name == "classical_matched":
            qnn = QuantumNetBasic(n_qubits=n_qubits, n_layers=n_layers, input_dim=input_dim)
            model = build_param_matched_classical(count_parameters(qnn), input_dim=input_dim)
        elif model_name == "quantum_angle":
            model = QuantumNetBasic(n_qubits=n_qubits, n_layers=n_layers, input_dim=input_dim)
        else:
            model = ClassicalNet(hidden=hidden, input_dim=input_dim)

        metrics = train_with_holdout(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            exp_id=cfg.get("exp_id", exp_key),
            model_name=f"hpo_{model_name}_seed{seed}",
            epochs=cfg["epochs"],
            lr=lr,
            seed=seed,
            profile=cfg.get("profile"),
            save_checkpoints=False,
        )
        accs.append(metrics["accuracy"])
    return float(np.mean(accs))


def build_exp_011_objective(exp_key: str, profile: str | None, model_name: str):
    def objective(trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "n_qubits": trial.suggest_int("n_qubits", 3, 6),
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "hidden": trial.suggest_int("hidden", 8, 64),
        }
        tracker = RunTracker(
            exp_key,
            f"hpo_{model_name}",
            profile=profile,
        )
        tracker.log_params(params)
        value = evaluate_uci_trial(exp_key, params, profile=profile, model_name=model_name)
        tracker.log_metrics({"mean_holdout_accuracy": value})
        tracker.end()
        return value

    return objective
