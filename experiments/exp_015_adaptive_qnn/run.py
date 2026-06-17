"""
EXP 015 — Adaptive QNN (gradient-variance learning rate)
Compares fixed vs adaptive LR on plateau-prone 6q QNN with ablation controls.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.generators import make_binary_classification
from src.data.splits import split_train_test
from src.quantum.qnn_basic import QuantumNetBasic
from src.training.adaptive_lr import AdaptiveLRConfig
from src.training.config import load_experiment_config
from src.training.holdout import (
    compare_conditions_batch,
    summarize_multi_seed,
    train_with_holdout,
    train_with_holdout_adaptive,
)
from src.training.param_match import build_param_matched_classical
from src.training.protocol import log_experiment_protocol
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters

EXP_KEY = "exp_015_adaptive_qnn"
EXP_ID = "exp_015"


def build_models(cfg: dict) -> dict[str, dict]:
    """Return model specs: {name: {model, lr, adaptive, adaptive_config}}."""
    mc = cfg.get("model_configs", {})
    adapt_cfg = cfg.get("adaptive_lr", {})
    base_adaptive = AdaptiveLRConfig(
        base_lr=adapt_cfg.get("base_lr", cfg["learning_rate"]),
        var_target=adapt_cfg.get("var_target", 0.015),
        min_scale=adapt_cfg.get("min_scale", 0.25),
        max_scale=adapt_cfg.get("max_scale", 4.0),
        warmup_epochs=adapt_cfg.get("warmup_epochs", 3),
        adapt_every=adapt_cfg.get("adapt_every", 1),
    )

    q6 = mc.get("quantum_6q_3l", {})
    q4 = mc.get("quantum_4q_2l", {})
    quantum_6 = QuantumNetBasic(
        n_qubits=q6.get("n_qubits", 6),
        n_layers=q6.get("n_layers", 3),
        input_dim=2,
    )
    target_params = count_parameters(quantum_6)
    matched = build_param_matched_classical(target_params, input_dim=2)
    hidden = matched.net[0].out_features

    return {
        "quantum_6q_3l_fixed": {
            "model": quantum_6,
            "lr": q6.get("learning_rate", cfg["learning_rate"]),
            "adaptive": False,
            "adaptive_config": None,
        },
        "quantum_6q_3l_adaptive": {
            "model": QuantumNetBasic(
                n_qubits=q6.get("n_qubits", 6),
                n_layers=q6.get("n_layers", 3),
                input_dim=2,
            ),
            "lr": q6.get("learning_rate", cfg["learning_rate"]),
            "adaptive": True,
            "adaptive_config": base_adaptive,
        },
        "quantum_4q_2l_fixed": {
            "model": QuantumNetBasic(
                n_qubits=q4.get("n_qubits", 4),
                n_layers=q4.get("n_layers", 2),
                input_dim=2,
            ),
            "lr": q4.get("learning_rate", cfg["learning_rate"]),
            "adaptive": False,
            "adaptive_config": None,
        },
        f"classical_matched_h{hidden}": {
            "model": matched,
            "lr": mc.get("classical_matched", {}).get("learning_rate", cfg["learning_rate"]),
            "adaptive": False,
            "adaptive_config": None,
        },
    }


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    seeds = cfg.get("seeds", [cfg["random_state"]])
    specs_probe = build_models(cfg)
    model_names: list[str] = []
    for name in cfg.get("models", list(specs_probe.keys())):
        if name == "classical_matched":
            classical_key = next(k for k in specs_probe if k.startswith("classical_matched"))
            model_names.append(classical_key)
        elif name in specs_probe:
            model_names.append(name)
        else:
            raise ValueError(f"Unknown model in config: {name}")

    log_experiment_protocol(EXP_ID, cfg)
    log_event(
        "info",
        "experiment run started",
        exp_id=EXP_ID,
        seeds=seeds,
        innovation="gradient_variance_adaptive_lr",
    )

    results_by_model: dict[str, list[float]] = {name: [] for name in model_names}

    for seed in seeds:
        specs = build_models(cfg)
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
            spec = specs[name]
            model = spec["model"]
            if spec["adaptive"]:
                metrics = train_with_holdout_adaptive(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    exp_id=EXP_ID,
                    model_name=f"{name}_seed{seed}",
                    epochs=cfg["epochs"],
                    adaptive_config=spec["adaptive_config"],
                    seed=seed,
                    profile=cfg.get("profile"),
                )
            else:
                metrics = train_with_holdout(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    exp_id=EXP_ID,
                    model_name=f"{name}_seed{seed}",
                    epochs=cfg["epochs"],
                    lr=spec["lr"],
                    seed=seed,
                    profile=cfg.get("profile"),
                )
            results_by_model[name].append(metrics["accuracy"])

    summarize_multi_seed(EXP_ID, results_by_model)

    comparisons = []
    if "quantum_6q_3l_adaptive" in results_by_model and "quantum_6q_3l_fixed" in results_by_model:
        comparisons.append(
            {
                "label_a": "quantum_6q_3l_adaptive",
                "label_b": "quantum_6q_3l_fixed",
                "condition_a": results_by_model["quantum_6q_3l_adaptive"],
                "condition_b": results_by_model["quantum_6q_3l_fixed"],
            }
        )
    if "quantum_6q_3l_adaptive" in results_by_model and "quantum_4q_2l_fixed" in results_by_model:
        comparisons.append(
            {
                "label_a": "quantum_6q_3l_adaptive",
                "label_b": "quantum_4q_2l_fixed",
                "condition_a": results_by_model["quantum_6q_3l_adaptive"],
                "condition_b": results_by_model["quantum_4q_2l_fixed"],
            }
        )
    classical_key = next((k for k in results_by_model if k.startswith("classical_matched")), None)
    if classical_key and "quantum_6q_3l_adaptive" in results_by_model:
        comparisons.append(
            {
                "label_a": "quantum_6q_3l_adaptive",
                "label_b": classical_key,
                "condition_a": results_by_model["quantum_6q_3l_adaptive"],
                "condition_b": results_by_model[classical_key],
            }
        )
    if comparisons:
        compare_conditions_batch(EXP_ID, comparisons)

    log_event("info", "experiment run finished", exp_id=EXP_ID)
