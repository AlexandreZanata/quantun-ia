"""
EXP 023 — Encoding × backend interaction on PCA-MNIST.

2×2 factorial: angle vs amplitude encoding × default.qubit vs lightning.qubit.
Combines exp_012 (encoding) and exp_021 (backend parity) protocols.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset_registry import prepare_dataset
from src.quantum.pennylane_device import QmlDeviceError
from src.quantum.qnn_amplitude import QuantumNetAmplitude
from src.quantum.qnn_basic import QuantumNetBasic
from src.training.config import load_experiment_config
from src.training.holdout import compare_conditions_batch, summarize_multi_seed, train_with_holdout
from src.training.protocol import log_experiment_protocol
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_023_encoding_backend"
EXP_ID = "exp_023"


def build_models(input_dim: int, cfg: dict) -> dict[str, tuple]:
    mc = cfg.get("model_configs", {})
    n_qubits = cfg.get("n_qubits", 4)
    n_layers = cfg.get("n_layers", 2)
    default_lr = cfg.get("learning_rate", 0.02)
    models: dict[str, tuple] = {}

    for name, model_cfg in mc.items():
        encoding = model_cfg.get("encoding", "angle")
        qml_device = model_cfg.get("qml_device", "default.qubit")
        lr = model_cfg.get("learning_rate", default_lr)
        try:
            if encoding == "amplitude":
                model = QuantumNetAmplitude(
                    n_qubits=n_qubits,
                    n_layers=n_layers,
                    input_dim=input_dim,
                    qml_device=qml_device,
                )
            else:
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
                encoding=encoding,
                qml_device=qml_device,
                error=str(exc),
            )
            continue
        models[name] = (model, lr)

    if not models:
        raise RuntimeError("no encoding×backend models available for exp_023")
    return models


def _paired_comparisons(results: dict[str, list[float]]) -> list[dict]:
    comparisons: list[dict] = []
    pairs = [
        ("angle_default", "angle_lightning"),
        ("amplitude_default", "amplitude_lightning"),
        ("angle_default", "amplitude_default"),
        ("angle_lightning", "amplitude_lightning"),
    ]
    for label_a, label_b in pairs:
        if label_a not in results or label_b not in results:
            continue
        a_vals = results[label_a]
        b_vals = results[label_b]
        if len(a_vals) == 0 or len(b_vals) == 0 or len(a_vals) != len(b_vals):
            log_event(
                "warning",
                "skipping paired comparison — unequal seed coverage",
                exp_id=EXP_ID,
                label_a=label_a,
                label_b=label_b,
                n_a=len(a_vals),
                n_b=len(b_vals),
            )
            continue
        comparisons.append(
            {
                "label_a": label_a,
                "label_b": label_b,
                "condition_a": a_vals,
                "condition_b": b_vals,
            }
        )
    return comparisons


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    seeds = cfg.get("seeds", [cfg["random_state"]])
    n_components = cfg.get("n_components", 8)
    log_experiment_protocol(EXP_ID, cfg)
    log_event(
        "info",
        "experiment run started",
        exp_id=EXP_ID,
        dataset=cfg.get("dataset", "mnist_binary"),
        seeds=seeds,
        n_components=n_components,
    )

    probe_meta = prepare_dataset(
        cfg.get("dataset", "mnist_binary"),
        random_state=seeds[0],
        test_size=cfg["test_size"],
        n_samples=cfg.get("n_samples", 500),
        n_components=n_components,
    )
    input_dim = probe_meta[0].shape[1]
    model_names = list(build_models(input_dim, cfg).keys())
    results_by_model: dict[str, list[float]] = {name: [] for name in model_names}

    for seed in seeds:
        X_train, X_test, y_train, y_test, meta = prepare_dataset(
            cfg.get("dataset", "mnist_binary"),
            random_state=seed,
            test_size=cfg["test_size"],
            n_samples=cfg.get("n_samples", 500),
            n_components=n_components,
        )
        log_event(
            "info",
            "pca prepared",
            exp_id=EXP_ID,
            seed=seed,
            n_components=n_components,
            input_dim=X_train.shape[1],
            pca_explained=sum(meta.get("pca").explained_variance_ratio_) if meta.get("pca") else None,
        )
        for name, (model, lr) in build_models(input_dim, cfg).items():
            try:
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
            except (RuntimeError, ValueError) as exc:
                log_event(
                    "warning",
                    "model training failed — skipping seed",
                    exp_id=EXP_ID,
                    model_name=name,
                    seed=seed,
                    error=str(exc),
                )

    summarize_multi_seed(EXP_ID, results_by_model)

    comparisons = _paired_comparisons(results_by_model)
    if comparisons:
        compare_conditions_batch(EXP_ID, comparisons)
    else:
        log_event(
            "warning",
            "encoding×backend comparisons skipped — insufficient model cells",
            exp_id=EXP_ID,
            models=model_names,
        )

    log_event("info", "experiment run finished", exp_id=EXP_ID)
