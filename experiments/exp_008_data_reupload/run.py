"""
EXP 008 — Data Re-uploading QNN on Circles
Compares basic QNN, re-upload QNN, and parameter-matched classical baseline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.generators import make_binary_classification
from src.data.splits import split_train_test
from src.quantum.qnn_basic import QuantumNetBasic
from src.quantum.qnn_reupload import QuantumNetReupload
from src.training.config import load_experiment_config
from src.training.holdout import compare_conditions_batch, summarize_multi_seed, train_with_holdout
from src.training.param_match import build_param_matched_classical
from src.training.protocol import log_experiment_protocol
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters

EXP_KEY = "exp_008_data_reupload"
EXP_ID = "exp_008"


def build_models_for_run(cfg: dict) -> dict[str, tuple]:
    mc = cfg.get("model_configs", {})
    basic_cfg = mc.get("quantum_basic", {})
    reupload_cfg = mc.get("quantum_reupload", {})

    reupload_model = QuantumNetReupload(
        n_qubits=reupload_cfg.get("n_qubits", 4),
        n_layers=reupload_cfg.get("n_layers", 3),
        input_dim=2,
    )
    target_params = count_parameters(reupload_model)
    matched = build_param_matched_classical(target_params, input_dim=2)
    hidden = matched.net[0].out_features

    return {
        "quantum_basic": (
            QuantumNetBasic(
                n_qubits=basic_cfg.get("n_qubits", 4),
                n_layers=basic_cfg.get("n_layers", 1),
                input_dim=2,
            ),
            basic_cfg.get("learning_rate", cfg["learning_rate"]),
        ),
        "quantum_reupload": (
            QuantumNetReupload(
                n_qubits=reupload_cfg.get("n_qubits", 4),
                n_layers=reupload_cfg.get("n_layers", 3),
                input_dim=2,
            ),
            reupload_cfg.get("learning_rate", cfg["learning_rate"]),
        ),
        f"classical_matched_h{hidden}": (
            build_param_matched_classical(target_params, input_dim=2),
            mc.get("classical_matched", {}).get("learning_rate", cfg["learning_rate"]),
        ),
    }


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    seeds = cfg.get("seeds", [cfg["random_state"]])
    log_experiment_protocol(EXP_ID, cfg)
    log_event("info", "experiment run started", exp_id=EXP_ID, seeds=seeds)

    probe = build_models_for_run(cfg)
    reupload_probe = probe["quantum_reupload"][0]
    classical_key = next(k for k in probe if k.startswith("classical_matched"))
    log_event(
        "info",
        "param match info",
        exp_id=EXP_ID,
        reupload_n_params=count_parameters(reupload_probe),
        classical_hidden=probe[classical_key][0].net[0].out_features,
        classical_n_params=count_parameters(probe[classical_key][0]),
    )

    model_names = list(probe.keys())
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

        for name, (model, lr) in build_models_for_run(cfg).items():
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
    if "quantum_reupload" in results_by_model and "quantum_basic" in results_by_model:
        comparisons.append(
            {
                "label_a": "quantum_reupload",
                "label_b": "quantum_basic",
                "condition_a": results_by_model["quantum_reupload"],
                "condition_b": results_by_model["quantum_basic"],
            }
        )
    if "quantum_reupload" in results_by_model:
        comparisons.append(
            {
                "label_a": classical_key,
                "label_b": "quantum_reupload",
                "condition_a": results_by_model[classical_key],
                "condition_b": results_by_model["quantum_reupload"],
            }
        )
    if comparisons:
        compare_conditions_batch(EXP_ID, comparisons)

    log_event("info", "experiment run finished", exp_id=EXP_ID)
