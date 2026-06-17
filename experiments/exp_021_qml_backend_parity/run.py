"""
EXP 021 — PennyLane backend parity on breast cancer QNN.

Compares default.qubit vs lightning.qubit with identical angle-encoding QNN (exp_011 protocol).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset_registry import prepare_dataset
from src.quantum.pennylane_device import QmlDeviceError
from src.quantum.qnn_basic import QuantumNetBasic
from src.training.config import load_experiment_config
from src.training.holdout import compare_conditions_batch, summarize_multi_seed, train_with_holdout
from src.training.protocol import log_experiment_protocol
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_021_qml_backend_parity"
EXP_ID = "exp_021"


def build_models(input_dim: int, cfg: dict) -> dict[str, tuple]:
    mc = cfg.get("model_configs", {})
    n_qubits = cfg.get("n_qubits", 4)
    n_layers = cfg.get("n_layers", 2)
    default_lr = cfg.get("learning_rate", 0.02)
    models: dict[str, tuple] = {}

    for name, model_cfg in mc.items():
        qml_device = model_cfg.get("qml_device")
        lr = model_cfg.get("learning_rate", default_lr)
        try:
            model = QuantumNetBasic(
                n_qubits=n_qubits,
                n_layers=n_layers,
                input_dim=input_dim,
                qml_device=qml_device,
            )
        except QmlDeviceError as exc:
            log_event(
                "warning",
                "qml backend unavailable — skipping model",
                exp_id=EXP_ID,
                model_name=name,
                qml_device=qml_device,
                error=str(exc),
            )
            continue
        models[name] = (model, lr)

    if not models:
        raise RuntimeError("no QML backends available for exp_021")
    return models


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

    backend_keys = [k for k in results_by_model if k.startswith("quantum_")]
    if len(backend_keys) >= 2:
        compare_conditions_batch(
            EXP_ID,
            [
                {
                    "label_a": backend_keys[0],
                    "label_b": backend_keys[1],
                    "condition_a": results_by_model[backend_keys[0]],
                    "condition_b": results_by_model[backend_keys[1]],
                }
            ],
        )
    else:
        log_event(
            "warning",
            "backend parity comparison skipped — fewer than two backends ran",
            exp_id=EXP_ID,
            backends=backend_keys,
        )

    log_event("info", "experiment run finished", exp_id=EXP_ID)
