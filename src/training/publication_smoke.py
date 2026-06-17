"""Fast publication-profile smoke runs for golden_publication.json regression."""

from __future__ import annotations

from pathlib import Path

from src.training import metrics as metrics_module
from src.training.config import load_experiment_config

EXP_011_KEY = "exp_011_uci_tabular_qml"
EXP_011_ID = "exp_011"
EXP_011_MODEL = "perceptron"

EXP_021_KEY = "exp_021_qml_backend_parity"
EXP_021_ID = "exp_021"

# Two seeds and reduced epochs keep CI fast while using publication holdout protocol.
PUBLICATION_SMOKE_SEEDS = [42, 123]
PUBLICATION_SMOKE_EPOCHS = 15


def _run_holdout_loop(
    *,
    exp_id: str,
    cfg: dict,
    model_names: list[str],
    build_model,
    log_path: Path | None,
    epochs: int,
) -> dict[str, list[float]]:
    original_log_path = metrics_module.LOGS_PATH
    if log_path is not None:
        metrics_module.LOGS_PATH = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[float]] = {name: [] for name in model_names}

    try:
        for seed in PUBLICATION_SMOKE_SEEDS:
            data = build_model(seed)
            X_train, X_test, y_train, y_test = data["splits"]
            for name in model_names:
                model, lr = data["models"][name]
                from src.training.holdout import train_with_holdout

                metrics = train_with_holdout(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    exp_id=exp_id,
                    model_name=f"{name}_seed{seed}",
                    epochs=epochs,
                    lr=lr,
                    seed=seed,
                    profile=cfg.get("profile"),
                )
                results[name].append(metrics["accuracy"])
    finally:
        metrics_module.LOGS_PATH = original_log_path

    return results


def run_exp_011_publication_smoke(*, log_path: Path | None = None) -> dict[str, list[float]]:
    """Publication-profile smoke: perceptron on breast cancer (2 seeds, 15 epochs)."""
    from src.classical.perceptron import Perceptron
    from src.data.dataset_registry import prepare_dataset

    cfg = load_experiment_config(EXP_011_KEY, profile="publication")
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
        exp_id=EXP_011_ID,
        cfg=cfg,
        model_names=model_names,
        build_model=build_model,
        log_path=log_path,
        epochs=PUBLICATION_SMOKE_EPOCHS,
    )


def run_exp_021_publication_smoke(*, log_path: Path | None = None) -> dict[str, list[float]]:
    """Publication-profile smoke: QNN backends on breast cancer (2 seeds, 15 epochs)."""
    from src.data.dataset_registry import prepare_dataset
    from src.quantum.pennylane_device import QmlDeviceError, resolve_qml_device
    from src.quantum.qnn_basic import QuantumNetBasic

    cfg = load_experiment_config(EXP_021_KEY, profile="publication")
    mc = cfg.get("model_configs", {})
    n_qubits = cfg.get("n_qubits", 4)
    n_layers = cfg.get("n_layers", 2)
    model_names: list[str] = []

    for name, model_cfg in mc.items():
        qml_device = model_cfg.get("qml_device", "default.qubit")
        try:
            resolve_qml_device(n_qubits, qml_device)
            model_names.append(name)
        except QmlDeviceError:
            if qml_device == "default.qubit":
                raise

    assert model_names, "exp_021 publication smoke requires at least default.qubit"

    def build_model(seed: int) -> dict:
        X_train, X_test, y_train, y_test, meta = prepare_dataset(
            cfg.get("dataset", "breast_cancer"),
            test_size=cfg["test_size"],
            random_state=seed,
            scale=True,
        )
        input_dim = meta.get("n_features", X_train.shape[1])
        models: dict[str, tuple] = {}
        for name in model_names:
            model_cfg = mc[name]
            qml_device = model_cfg.get("qml_device", "default.qubit")
            lr = model_cfg.get("learning_rate", cfg.get("learning_rate", 0.02))
            models[name] = (
                QuantumNetBasic(
                    n_qubits=n_qubits,
                    n_layers=n_layers,
                    input_dim=input_dim,
                    qml_device=qml_device,
                ),
                lr,
            )
        return {
            "splits": (X_train, X_test, y_train, y_test),
            "models": models,
        }

    return _run_holdout_loop(
        exp_id=EXP_021_ID,
        cfg=cfg,
        model_names=model_names,
        build_model=build_model,
        log_path=log_path,
        epochs=PUBLICATION_SMOKE_EPOCHS,
    )
