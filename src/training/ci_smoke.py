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
    save_checkpoints: bool = False,
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
                    save_checkpoints=save_checkpoints,
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


EXP_016_KEY = "exp_016_hybrid_nas"
EXP_016_ID = "exp_016"


def run_exp_016_ci(*, log_path: Path | None = None) -> dict[str, list[float]]:
    """Fast CI smoke: 3 Optuna trials + nas_best vs hybrid_sandwich on ci profile."""
    from src.quantum.hybrid_model import HybridSandwich
    from src.training.hpo import build_exp_016_objective, build_hybrid_from_params, run_optuna_study

    cfg = load_experiment_config(EXP_016_KEY, profile="ci")
    model_names = ["nas_best", "hybrid_sandwich"]
    n_trials = int(cfg.get("hpo_trials", 3))

    objective = build_exp_016_objective(EXP_016_KEY, "ci")
    best = run_optuna_study(EXP_016_KEY, objective, n_trials=n_trials, profile="ci")
    best_params = best["best_params"]

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
        nas_model, nas_lr = build_hybrid_from_params(best_params, input_dim=2)
        mc = cfg.get("model_configs", {}).get("hybrid_sandwich", {})
        baseline = HybridSandwich(
            input_dim=2,
            n_qubits=mc.get("n_qubits", 4),
            n_layers=mc.get("n_layers", 3),
            reupload=mc.get("reupload", True),
        )
        baseline_lr = mc.get("learning_rate", cfg["learning_rate"])
        return {
            "splits": (X_train, X_test, y_train, y_test),
            "models": {
                "nas_best": (nas_model, nas_lr),
                "hybrid_sandwich": (baseline, baseline_lr),
            },
        }

    return _run_holdout_loop(
        exp_key=EXP_016_KEY,
        exp_id=EXP_016_ID,
        cfg=cfg,
        model_names=model_names,
        build_model=build_model,
        log_path=log_path,
    )


EXP_017_KEY = "exp_017_poison_topology"
EXP_017_ID = "exp_017"


def run_exp_017_ci(*, log_path: Path | None = None) -> dict[str, list[float]]:
    """Fast CI smoke: hybrid_sandwich at 0% and 30% poison, ci profile."""
    import torch

    from src.data.poisoning import poison_dataset
    from src.quantum.hybrid_model import HybridSandwich
    from src.training import metrics as metrics_module

    cfg = load_experiment_config(EXP_017_KEY, profile="ci")
    rates = [0.0, 0.3]
    model_names = [f"hybrid_sandwich_poison_{int(r * 100)}" for r in rates]
    results: dict[str, list[float]] = {name: [] for name in model_names}

    original_log_path = metrics_module.LOGS_PATH
    if log_path is not None:
        metrics_module.LOGS_PATH = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        mc = cfg.get("model_configs", {}).get("hybrid_sandwich", {})
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
            X_test_t = torch.tensor(X_test)
            y_test_t = torch.tensor(y_test)
            lr = mc.get("learning_rate", cfg["learning_rate"])

            for rate in rates:
                _, y_poisoned, _ = poison_dataset(X_train, y_train, poison_rate=rate, seed=seed)
                model = HybridSandwich(
                    input_dim=2,
                    n_qubits=mc.get("n_qubits", 4),
                    n_layers=mc.get("n_layers", 3),
                    reupload=mc.get("reupload", True),
                )
                model.train(
                    torch.tensor(X_train),
                    torch.tensor(y_poisoned),
                    exp_id=EXP_017_ID,
                    model_name=f"hybrid_sandwich_poison_{int(rate * 100)}_seed{seed}",
                    epochs=cfg["epochs"],
                    lr=lr,
                    X_test=X_test_t,
                    y_test=y_test_t,
                )
                acc = model.evaluate(X_test_t, y_test_t)["accuracy"]
                results[f"hybrid_sandwich_poison_{int(rate * 100)}"].append(acc)
    finally:
        metrics_module.LOGS_PATH = original_log_path

    return results


EXP_018_KEY = "exp_018_feature_fusion"
EXP_018_ID = "exp_018"


def run_exp_018_ci(*, log_path: Path | None = None) -> dict[str, list[float]]:
    """Fast CI smoke: transformer_qnn_fusion on sequential_phase, ci profile."""
    import torch

    from src.data.dataset_registry import prepare_dataset
    from src.quantum.transformer_qnn_fusion import TransformerQNNFusion
    from src.training import metrics as metrics_module
    from src.training.trainer import train_model

    cfg = load_experiment_config(EXP_018_KEY, profile="ci")
    model_name = "transformer_qnn_fusion"
    results: dict[str, list[float]] = {model_name: []}

    original_log_path = metrics_module.LOGS_PATH
    if log_path is not None:
        metrics_module.LOGS_PATH = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        mc = cfg.get("model_configs", {}).get(model_name, {})
        d_model = mc.get("d_model", 16)
        lr = mc.get("learning_rate", cfg["learning_rate"])
        for seed in cfg["seeds"]:
            X_train, X_test, y_train, y_test, _ = prepare_dataset(
                cfg.get("dataset", "sequential_phase"),
                random_state=seed,
                test_size=cfg["test_size"],
                n_samples=cfg["n_samples"],
                seq_len=cfg.get("seq_len", 12),
                input_dim=cfg.get("input_dim", 4),
                noise=cfg.get("noise", 0.15),
            )
            model = TransformerQNNFusion(
                input_dim=cfg.get("input_dim", 4),
                d_model=d_model,
                n_qubits=cfg.get("n_qubits", 4),
                n_layers=cfg.get("n_layers", 2),
            )
            X_train_t = torch.tensor(X_train)
            y_train_t = torch.tensor(y_train)
            X_test_t = torch.tensor(X_test)
            y_test_t = torch.tensor(y_test)
            train_model(
                model,
                X_train_t,
                y_train_t,
                EXP_018_ID,
                f"{model_name}_seed{seed}",
                epochs=cfg["epochs"],
                lr=lr,
                X_test=X_test_t,
                y_test=y_test_t,
                seed=seed,
                profile=cfg.get("profile"),
            )
            acc = model.evaluate(X_test_t, y_test_t)["accuracy"]
            results[model_name].append(acc)
    finally:
        metrics_module.LOGS_PATH = original_log_path

    return results


EXP_021_KEY = "exp_021_qml_backend_parity"
EXP_021_ID = "exp_021"


def run_exp_021_ci(*, log_path: Path | None = None) -> dict[str, list[float]]:
    """Fast CI smoke: QNN on breast cancer with default.qubit (and lightning when available)."""
    from src.data.dataset_registry import prepare_dataset
    from src.quantum.pennylane_device import QmlDeviceError, resolve_qml_device
    from src.quantum.qnn_basic import QuantumNetBasic

    cfg = load_experiment_config(EXP_021_KEY, profile="ci")
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

    assert model_names, "exp_021 CI requires at least default.qubit"

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
        exp_key=EXP_021_KEY,
        exp_id=EXP_021_ID,
        cfg=cfg,
        model_names=model_names,
        build_model=build_model,
        log_path=log_path,
    )


EXP_022_ID = "exp_022"
EXP_022_CI_DATASET = "breast_cancer"
EXP_022_CI_MODEL = "hybrid_sandwich"


def run_exp_022_ci(*, log_path: Path | None = None, epochs: int = 15) -> dict[str, object]:
    """Fast CI smoke: hybrid_sandwich vs parameter-matched classical on breast cancer."""
    from src.application.dto import NanoParityBenchDTO
    from src.application.nano_parity_bench import execute
    from src.application.parity_config import load_parity_config, profile_settings
    from src.shared.result import Fail, Ok
    from src.training import metrics as metrics_module

    original_log_path = metrics_module.LOGS_PATH
    if log_path is not None:
        metrics_module.LOGS_PATH = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        cfg = load_parity_config()
        prof = profile_settings(cfg, "ci")
        seeds = list(prof.get("seeds", [42, 123, 456]))[:3]

        dto = NanoParityBenchDTO(
            quantum_model=EXP_022_CI_MODEL,
            dataset=EXP_022_CI_DATASET,
            profile="ci",
            exp_id=EXP_022_ID,
            seeds=seeds,
            epochs=epochs,
        )
        outcome = execute(dto)
        if isinstance(outcome, Fail):
            raise RuntimeError(outcome.error.message)
        assert isinstance(outcome, Ok)
        result = outcome.value
        return {
            EXP_022_CI_MODEL: result.quantum_accuracies,
            result.classical_label: result.classical_accuracies,
            "quantum_mean": result.quantum_mean,
            "classical_mean": result.classical_mean,
            "mean_diff": result.comparison["mean_diff"],
            "param_delta": result.param_delta,
        }
    finally:
        metrics_module.LOGS_PATH = original_log_path


EXP_023_KEY = "exp_023_encoding_backend"
EXP_023_ID = "exp_023"
EXP_023_CI_MODELS = ("angle_default", "amplitude_default", "angle_lightning")


def run_exp_023_ci(*, log_path: Path | None = None) -> dict[str, list[float]]:
    """Fast CI smoke: encoding×backend 2×2 on PCA-MNIST (default.qubit required)."""
    from src.data.dataset_registry import prepare_dataset
    from src.quantum.pennylane_device import QmlDeviceError, resolve_qml_device
    from src.quantum.qnn_amplitude import QuantumNetAmplitude
    from src.quantum.qnn_basic import QuantumNetBasic

    cfg = load_experiment_config(EXP_023_KEY, profile="ci")
    mc = cfg.get("model_configs", {})
    n_qubits = cfg.get("n_qubits", 4)
    n_layers = cfg.get("n_layers", 2)
    n_components = cfg.get("n_components", 8)
    model_names: list[str] = []

    for name, model_cfg in mc.items():
        qml_device = model_cfg.get("qml_device", "default.qubit")
        try:
            resolve_qml_device(n_qubits, qml_device)
            model_names.append(name)
        except QmlDeviceError:
            if qml_device == "default.qubit":
                raise

    assert model_names, "exp_023 CI requires at least default.qubit cells"
    model_names = [name for name in EXP_023_CI_MODELS if name in model_names] or model_names

    def build_model(seed: int) -> dict:
        X_train, X_test, y_train, y_test, _meta = prepare_dataset(
            cfg.get("dataset", "mnist_binary"),
            test_size=cfg["test_size"],
            random_state=seed,
            n_samples=cfg["n_samples"],
            n_components=n_components,
        )
        input_dim = X_train.shape[1]
        models: dict[str, tuple] = {}
        for name in model_names:
            model_cfg = mc[name]
            encoding = model_cfg.get("encoding", "angle")
            qml_device = model_cfg.get("qml_device", "default.qubit")
            lr = model_cfg.get("learning_rate", cfg.get("learning_rate", 0.02))
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
            models[name] = (model, lr)
        return {
            "splits": (X_train, X_test, y_train, y_test),
            "models": models,
        }

    return _run_holdout_loop(
        exp_key=EXP_023_KEY,
        exp_id=EXP_023_ID,
        cfg=cfg,
        model_names=model_names,
        build_model=build_model,
        log_path=log_path,
    )


EXP_024_KEY = "exp_024_quantum_nano_bc"
EXP_024_ID = "exp_024"


def run_exp_024_ci(*, log_path: Path | None = None) -> dict[str, list[float]]:
    """Fast CI smoke for exp_024 QuantumNano-BC (all baselines, ci profile)."""
    from src.classical.logistic_baseline import LogisticBaseline
    from src.classical.perceptron import Perceptron
    from src.classical.xgboost_baseline import XGBoostShallow
    from src.data.dataset_registry import prepare_dataset
    from src.quantum.hybrid_model import HybridSandwich
    from src.training.param_match import build_param_matched_classical
    from src.training.trainer import count_parameters

    cfg = load_experiment_config(EXP_024_KEY, profile="ci")

    def build_model(seed: int) -> dict:
        X_train, X_test, y_train, y_test, _meta = prepare_dataset(
            cfg.get("dataset", "breast_cancer"),
            test_size=cfg["test_size"],
            random_state=seed,
            scale=True,
        )
        input_dim = int(X_train.shape[1])
        mc = cfg.get("model_configs", {})
        hybrid = HybridSandwich(
            input_dim=input_dim,
            n_qubits=cfg.get("n_qubits", 4),
            n_layers=cfg.get("n_layers", 2),
            reupload=bool(mc.get("hybrid_sandwich", {}).get("reupload", True)),
        )
        matched = build_param_matched_classical(count_parameters(hybrid), input_dim=input_dim)
        hidden = matched.net[0].out_features
        models = {
            "logistic_regression": (LogisticBaseline(input_dim=input_dim), 1.0),
            "perceptron": (
                Perceptron(input_dim=input_dim),
                mc.get("perceptron", {}).get("learning_rate", cfg["learning_rate"]),
            ),
            f"classical_matched_h{hidden}": (
                matched,
                mc.get("classical_matched", {}).get("learning_rate", cfg["learning_rate"]),
            ),
            "hybrid_sandwich": (
                hybrid,
                mc.get("hybrid_sandwich", {}).get("learning_rate", cfg["learning_rate"]),
            ),
            "xgboost_shallow": (XGBoostShallow(input_dim=input_dim), 0.1),
        }
        return {"splits": (X_train, X_test, y_train, y_test), "models": models}

    model_names = list(build_model(cfg["seeds"][0])["models"].keys())
    return _run_holdout_loop(
        exp_key=EXP_024_KEY,
        exp_id=EXP_024_ID,
        cfg=cfg,
        model_names=model_names,
        build_model=build_model,
        log_path=log_path,
        save_checkpoints=False,
    )
