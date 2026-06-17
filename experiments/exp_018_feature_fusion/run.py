"""
EXP 018 — Feature Fusion (Transformer-mini → QNN)
Phase-sensitive sequences where PCA on flat windows is insufficient.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.classical.transformer_mini import TransformerMini
from src.data.dataset_registry import prepare_dataset
from src.data.scaling import pca_train_test
from src.quantum.qnn_basic import QuantumNetBasic
from src.quantum.transformer_qnn_fusion import TransformerQNNFusion
from src.training.config import load_experiment_config
from src.training.holdout import compare_conditions_batch, summarize_multi_seed
from src.training.protocol import log_experiment_protocol
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import train_model

EXP_KEY = "exp_018_feature_fusion"
EXP_ID = "exp_018"


def train_sequence_holdout(model, X_train, y_train, X_test, y_test, exp_id, model_name, **kwargs):
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)
    train_model(
        model,
        X_train_t,
        y_train_t,
        exp_id,
        model_name,
        X_test=X_test_t,
        y_test=y_test_t,
        **kwargs,
    )
    return model.evaluate(X_test_t, y_test_t)


def build_models(cfg: dict, flat_dim: int, input_dim: int) -> dict[str, tuple]:
    mc = cfg.get("model_configs", {})
    n_qubits = cfg.get("n_qubits", 4)
    n_layers = cfg.get("n_layers", 2)
    d_model = mc.get("transformer_qnn_fusion", {}).get("d_model", 16)

    return {
        "transformer_qnn_fusion": (
            TransformerQNNFusion(
                input_dim=input_dim,
                d_model=d_model,
                n_qubits=n_qubits,
                n_layers=n_layers,
                reupload=cfg.get("reupload", False),
            ),
            mc.get("transformer_qnn_fusion", {}).get("learning_rate", cfg["learning_rate"]),
        ),
        "transformer_mini": (
            TransformerMini(input_dim=input_dim, d_model=d_model),
            mc.get("transformer_mini", {}).get("learning_rate", cfg["learning_rate"]),
        ),
        "quantum_pca": (
            QuantumNetBasic(n_qubits=n_qubits, n_layers=n_layers, input_dim=n_qubits),
            mc.get("quantum_pca", {}).get("learning_rate", 0.02),
        ),
        "quantum_flat": (
            QuantumNetBasic(n_qubits=n_qubits, n_layers=n_layers, input_dim=flat_dim),
            mc.get("quantum_flat", {}).get("learning_rate", 0.02),
        ),
    }


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    seeds = cfg.get("seeds", [cfg["random_state"]])
    seq_len = cfg.get("seq_len", 12)
    input_dim = cfg.get("input_dim", 4)
    flat_dim = seq_len * input_dim
    pca_components = int(cfg.get("pca_components", cfg.get("n_qubits", 4)))
    model_names = cfg.get(
        "models", ["transformer_qnn_fusion", "transformer_mini", "quantum_pca", "quantum_flat"]
    )

    log_experiment_protocol(EXP_ID, cfg)
    log_event(
        "info",
        "experiment run started",
        exp_id=EXP_ID,
        seeds=seeds,
        dataset=cfg.get("dataset", "sequential_phase"),
    )

    results_by_model: dict[str, list[float]] = {name: [] for name in model_names}

    for seed in seeds:
        X_train, X_test, y_train, y_test, _ = prepare_dataset(
            cfg.get("dataset", "sequential_phase"),
            random_state=seed,
            test_size=cfg["test_size"],
            n_samples=cfg.get("n_samples", 300),
            seq_len=seq_len,
            input_dim=input_dim,
            noise=cfg.get("noise", 0.15),
        )

        flat_train = X_train.reshape(len(X_train), -1)
        flat_test = X_test.reshape(len(X_test), -1)
        pca_train, pca_test, _ = pca_train_test(
            flat_train, flat_test, pca_components, random_state=seed
        )

        models = build_models(cfg, flat_dim, input_dim)

        for name in model_names:
            model, lr = models[name]
            if name == "quantum_pca":
                metrics = train_sequence_holdout(
                    model,
                    pca_train,
                    y_train,
                    pca_test,
                    y_test,
                    exp_id=EXP_ID,
                    model_name=f"{name}_seed{seed}",
                    epochs=cfg["epochs"],
                    lr=lr,
                    seed=seed,
                    profile=cfg.get("profile"),
                )
            elif name == "quantum_flat":
                metrics = train_sequence_holdout(
                    model,
                    flat_train,
                    y_train,
                    flat_test,
                    y_test,
                    exp_id=EXP_ID,
                    model_name=f"{name}_seed{seed}",
                    epochs=cfg["epochs"],
                    lr=lr,
                    seed=seed,
                    profile=cfg.get("profile"),
                )
            else:
                metrics = train_sequence_holdout(
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
    if "transformer_qnn_fusion" in results_by_model and "quantum_pca" in results_by_model:
        comparisons.append(
            {
                "label_a": "transformer_qnn_fusion",
                "label_b": "quantum_pca",
                "condition_a": results_by_model["transformer_qnn_fusion"],
                "condition_b": results_by_model["quantum_pca"],
            }
        )
    if "transformer_qnn_fusion" in results_by_model and "transformer_mini" in results_by_model:
        comparisons.append(
            {
                "label_a": "transformer_qnn_fusion",
                "label_b": "transformer_mini",
                "condition_a": results_by_model["transformer_qnn_fusion"],
                "condition_b": results_by_model["transformer_mini"],
            }
        )
    if "transformer_qnn_fusion" in results_by_model and "quantum_flat" in results_by_model:
        comparisons.append(
            {
                "label_a": "transformer_qnn_fusion",
                "label_b": "quantum_flat",
                "condition_a": results_by_model["transformer_qnn_fusion"],
                "condition_b": results_by_model["quantum_flat"],
            }
        )
    if comparisons:
        compare_conditions_batch(EXP_ID, comparisons)

    log_event("info", "experiment run finished", exp_id=EXP_ID)
