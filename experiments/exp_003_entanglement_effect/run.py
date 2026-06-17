"""
EXP 003 — Quantum Entanglement Effect
Holdout evaluation on 30% test split; repeated across 10 seeds (publication profile).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.generators import make_binary_classification
from src.data.splits import split_train_test
from src.quantum.qnn_entangled import QuantumNetEntangled
from src.training.config import load_experiment_config
from src.training.holdout import compare_conditions_batch, summarize_multi_seed, train_with_holdout
from src.training.protocol import log_experiment_protocol
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_003_entanglement_effect"
EXP_ID = "exp_003"


def build_entangled(entanglement: str, cfg: dict) -> tuple[QuantumNetEntangled, float]:
    name = f"entanglement_{entanglement}"
    model_cfg = cfg.get("model_configs", {}).get(name, {})
    n_qubits = cfg.get("n_qubits", 4)
    n_layers = model_cfg.get("n_layers", cfg.get("n_layers", 2))
    reupload = cfg.get("qnn_type", "basic") == "reupload"
    lr = model_cfg.get("learning_rate", cfg["learning_rate"])
    return QuantumNetEntangled(
        n_qubits=n_qubits,
        n_layers=n_layers,
        entanglement=entanglement,
        input_dim=2,
        reupload=reupload,
    ), lr


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    seeds = cfg.get("seeds", [cfg["random_state"]])
    log_experiment_protocol(EXP_ID, cfg)
    log_event("info", "experiment run started", exp_id=EXP_ID, seeds=seeds)

    results_by_model: dict[str, list[float]] = {
        f"entanglement_{e}": [] for e in cfg["entanglement_types"]
    }

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

        for entanglement in cfg["entanglement_types"]:
            name = f"entanglement_{entanglement}"
            model, lr = build_entangled(entanglement, cfg)
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
    baseline = "entanglement_none"
    if baseline in results_by_model:
        compare_conditions_batch(
            EXP_ID,
            [
                {
                    "label_a": f"entanglement_{e}",
                    "label_b": baseline,
                    "condition_a": results_by_model[f"entanglement_{e}"],
                    "condition_b": results_by_model[baseline],
                }
                for e in cfg["entanglement_types"]
                if e != "none" and f"entanglement_{e}" in results_by_model
            ],
        )
    log_event("info", "experiment run finished", exp_id=EXP_ID)
