"""
EXP 014 — Sequence baselines: RNNMini vs TransformerMini vs flattened QNN.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.classical.rnn_mini import RNNMini
from src.classical.transformer_mini import TransformerMini
from src.data.dataset_registry import prepare_dataset
from src.quantum.qnn_basic import QuantumNetBasic
from src.training.config import load_experiment_config
from src.training.holdout import compare_conditions_batch, summarize_multi_seed
from src.training.protocol import log_experiment_protocol
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import train_model

EXP_KEY = "exp_014_sequence_baselines"
EXP_ID = "exp_014"


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


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    seeds = cfg.get("seeds", [cfg["random_state"]])
    seq_len = cfg.get("seq_len", 8)
    input_dim = cfg.get("input_dim", 2)
    log_experiment_protocol(EXP_ID, cfg)
    log_event("info", "experiment run started", exp_id=EXP_ID, seeds=seeds)

    results_by_model: dict[str, list[float]] = {
        "rnn_mini": [],
        "transformer_mini": [],
        "quantum_flat": [],
    }

    for seed in seeds:
        X_train, X_test, y_train, y_test, _ = prepare_dataset(
            "sequential_binary",
            random_state=seed,
            test_size=cfg["test_size"],
            n_samples=cfg.get("n_samples", 300),
            seq_len=seq_len,
            input_dim=input_dim,
            noise=cfg.get("noise", 0.1),
        )

        flat_dim = seq_len * input_dim
        mc = cfg.get("model_configs", {})

        models = {
            "rnn_mini": (
                RNNMini(input_dim=input_dim, hidden_dim=mc.get("rnn_mini", {}).get("hidden_dim", 16)),
                mc.get("rnn_mini", {}).get("learning_rate", cfg["learning_rate"]),
            ),
            "transformer_mini": (
                TransformerMini(input_dim=input_dim, d_model=mc.get("transformer_mini", {}).get("d_model", 16)),
                mc.get("transformer_mini", {}).get("learning_rate", cfg["learning_rate"]),
            ),
            "quantum_flat": (
                QuantumNetBasic(
                    n_qubits=cfg.get("n_qubits", 4),
                    n_layers=cfg.get("n_layers", 2),
                    input_dim=flat_dim,
                ),
                mc.get("quantum_flat", {}).get("learning_rate", 0.02),
            ),
        }

        for name, (model, lr) in models.items():
            X_tr, X_te = X_train, X_test
            if name == "quantum_flat":
                X_tr = X_train.reshape(len(X_train), -1)
                X_te = X_test.reshape(len(X_test), -1)
            metrics = train_sequence_holdout(
                model,
                X_tr,
                y_train,
                X_te,
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
    compare_conditions_batch(
        EXP_ID,
        [
            {
                "label_a": "transformer_mini",
                "label_b": "rnn_mini",
                "condition_a": results_by_model["transformer_mini"],
                "condition_b": results_by_model["rnn_mini"],
            },
            {
                "label_a": "quantum_flat",
                "label_b": "rnn_mini",
                "condition_a": results_by_model["quantum_flat"],
                "condition_b": results_by_model["rnn_mini"],
            },
        ],
    )
    log_event("info", "experiment run finished", exp_id=EXP_ID)
