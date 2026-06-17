"""
EXP 016 — Hybrid NAS
Optuna search over hybrid layouts, then multi-seed holdout vs EXP 002 baselines.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.generators import make_binary_classification
from src.data.splits import split_train_test
from src.quantum.hybrid_model import ClassicalFirst, HybridSandwich, QuantumFirst
from src.training.config import load_experiment_config
from src.training.holdout import compare_conditions_batch, summarize_multi_seed, train_with_holdout
from src.training.hpo import build_exp_016_objective, build_hybrid_from_params, run_optuna_study
from src.training.protocol import log_experiment_protocol
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_016_hybrid_nas"
EXP_ID = "exp_016"

BASELINE_CLASSES = {
    "hybrid_sandwich": HybridSandwich,
    "quantum_first": QuantumFirst,
    "classical_first": ClassicalFirst,
}


def build_baseline(name: str, cfg: dict) -> tuple:
    mc = cfg.get("model_configs", {}).get(name, {})
    reupload = mc.get("reupload", cfg.get("qnn_type", "basic") == "reupload")
    cls = BASELINE_CLASSES[name]
    model = cls(
        input_dim=2,
        n_qubits=mc.get("n_qubits", cfg.get("n_qubits", 4)),
        n_layers=mc.get("n_layers", cfg.get("n_layers", 3)),
        reupload=reupload,
    )
    lr = mc.get("learning_rate", cfg["learning_rate"])
    return model, lr


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    seeds = cfg.get("seeds", [cfg["random_state"]])
    baselines = cfg.get("baselines", list(BASELINE_CLASSES))
    n_trials = int(cfg.get("hpo_trials", 20))

    log_experiment_protocol(EXP_ID, cfg)
    log_event("info", "experiment run started", exp_id=EXP_ID, seeds=seeds, phase="nas")

    objective = build_exp_016_objective(EXP_KEY, cfg.get("profile"))
    best = run_optuna_study(EXP_KEY, objective, n_trials=n_trials, profile=cfg.get("profile"))
    best_params = best["best_params"]
    log_event(
        "info",
        "nas search finished",
        exp_id=EXP_ID,
        best_value=best["best_value"],
        best_params=best_params,
        n_trials=best["n_trials"],
    )

    model_names = ["nas_best"] + list(baselines)
    results_by_model: dict[str, list[float]] = {name: [] for name in model_names}

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

        nas_model, nas_lr = build_hybrid_from_params(best_params, input_dim=2)
        metrics = train_with_holdout(
            nas_model,
            X_train,
            y_train,
            X_test,
            y_test,
            exp_id=EXP_ID,
            model_name=f"nas_best_seed{seed}",
            epochs=cfg["epochs"],
            lr=nas_lr,
            seed=seed,
            profile=cfg.get("profile"),
        )
        results_by_model["nas_best"].append(metrics["accuracy"])

        for name in baselines:
            model, lr = build_baseline(name, cfg)
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
            results_by_model[name].append(metrics["accuracy"])

    summarize_multi_seed(EXP_ID, results_by_model)

    comparisons = []
    for baseline in baselines:
        comparisons.append(
            {
                "label_a": "nas_best",
                "label_b": baseline,
                "condition_a": results_by_model["nas_best"],
                "condition_b": results_by_model[baseline],
            }
        )
    compare_conditions_batch(EXP_ID, comparisons)

    log_event("info", "experiment run finished", exp_id=EXP_ID, nas_best_params=best_params)
