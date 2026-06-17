"""
EXP 011 — UCI Tabular QML vs Classical (breast cancer)
Perceptron, parameter-matched MLP, and angle-encoding QNN.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.classical.perceptron import Perceptron
from src.data.dataset_registry import prepare_dataset
from src.quantum.qnn_basic import QuantumNetBasic
from src.training.config import load_experiment_config
from src.training.holdout import compare_conditions_batch, summarize_multi_seed, train_with_holdout
from src.training.param_match import build_param_matched_classical
from src.training.protocol import log_experiment_protocol
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters

EXP_KEY = "exp_011_uci_tabular_qml"
EXP_ID = "exp_011"


def build_models(input_dim: int, cfg: dict) -> dict[str, tuple]:
    mc = cfg.get("model_configs", {})
    n_qubits = cfg.get("n_qubits", 4)
    n_layers = cfg.get("n_layers", 2)
    qnn = QuantumNetBasic(n_qubits=n_qubits, n_layers=n_layers, input_dim=input_dim)
    target_params = count_parameters(qnn)
    matched = build_param_matched_classical(target_params, input_dim=input_dim)
    hidden = matched.net[0].out_features

    return {
        "perceptron": (
            Perceptron(input_dim=input_dim),
            mc.get("perceptron", {}).get("learning_rate", cfg["learning_rate"]),
        ),
        f"classical_matched_h{hidden}": (
            matched,
            mc.get("classical_matched", {}).get("learning_rate", cfg["learning_rate"]),
        ),
        "quantum_angle": (
            qnn,
            mc.get("quantum_angle", {}).get("learning_rate", cfg["learning_rate"]),
        ),
    }


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    seeds = cfg.get("seeds", [cfg["random_state"]])
    dataset = cfg.get("dataset", "breast_cancer")
    log_experiment_protocol(EXP_ID, cfg)
    log_event("info", "experiment run started", exp_id=EXP_ID, dataset=dataset, seeds=seeds)

    probe_meta = prepare_dataset(dataset, random_state=seeds[0], test_size=cfg["test_size"])[4]
    input_dim = probe_meta.get("n_features", 30)
    model_names = list(build_models(input_dim, cfg).keys())
    results_by_model: dict[str, list[float]] = {name: [] for name in model_names}

    for seed in seeds:
        X_train, X_test, y_train, y_test, _ = prepare_dataset(
            dataset,
            random_state=seed,
            test_size=cfg["test_size"],
        )
        for name, (model, lr) in build_models(input_dim, cfg).items():
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

    classical_key = next(k for k in results_by_model if k.startswith("classical_matched"))
    comparisons = [
        {
            "label_a": "quantum_angle",
            "label_b": classical_key,
            "condition_a": results_by_model["quantum_angle"],
            "condition_b": results_by_model[classical_key],
        },
        {
            "label_a": "perceptron",
            "label_b": "quantum_angle",
            "condition_a": results_by_model["perceptron"],
            "condition_b": results_by_model["quantum_angle"],
        },
    ]
    compare_conditions_batch(EXP_ID, comparisons)
    log_event("info", "experiment run finished", exp_id=EXP_ID)
