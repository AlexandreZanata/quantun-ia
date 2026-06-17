"""
EXP 017 — Poisoning × Hybrid Topology
Train on poisoned labels; evaluate on clean holdout across hybrid layouts + NAS preset.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.data.generators import make_binary_classification
from src.data.poisoning import measure_robustness, poison_dataset
from src.data.splits import split_train_test
from src.quantum.hybrid_model import ClassicalFirst, HybridSandwich, QuantumFirst
from src.training.config import load_experiment_config
from src.training.holdout import compare_conditions_batch, summarize_multi_seed
from src.training.hpo import build_hybrid_from_params
from src.training.protocol import log_experiment_protocol
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_017_poison_topology"
EXP_ID = "exp_017"

TOPOLOGY_CLASSES = {
    "hybrid_sandwich": HybridSandwich,
    "quantum_first": QuantumFirst,
    "classical_first": ClassicalFirst,
}


def build_topology(name: str, cfg: dict):
    if name == "nas_preset":
        preset = cfg.get("nas_preset", {})
        return build_hybrid_from_params(preset, input_dim=2)
    mc = cfg.get("model_configs", {}).get(name, {})
    reupload = mc.get("reupload", cfg.get("qnn_type", "basic") == "reupload")
    cls = TOPOLOGY_CLASSES[name]
    model = cls(
        input_dim=2,
        n_qubits=mc.get("n_qubits", cfg.get("n_qubits", 4)),
        n_layers=mc.get("n_layers", cfg.get("n_layers", 3)),
        reupload=reupload,
    )
    lr = mc.get("learning_rate", cfg["learning_rate"])
    return model, lr


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    seeds = cfg.get("seeds", [cfg["random_state"]])
    topologies = cfg.get("topologies", list(TOPOLOGY_CLASSES))
    poison_rates = cfg["poison_rates"]
    log_experiment_protocol(EXP_ID, cfg)
    log_event("info", "experiment run started", exp_id=EXP_ID, seeds=seeds, topologies=topologies)

    by_topology: dict[str, dict[float, list[float]]] = {
        topo: {r: [] for r in poison_rates} for topo in topologies
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
        X_test_t = torch.tensor(X_test)
        y_test_t = torch.tensor(y_test)

        for rate in poison_rates:
            _, y_train_poisoned, _ = poison_dataset(X_train, y_train, poison_rate=rate, seed=seed)
            X_train_t = torch.tensor(X_train)
            y_train_t = torch.tensor(y_train_poisoned)

            for topo in topologies:
                model, lr = build_topology(topo, cfg)
                model.train(
                    X_train_t,
                    y_train_t,
                    exp_id=EXP_ID,
                    model_name=f"{topo}_poison_{int(rate * 100)}_seed{seed}",
                    epochs=cfg["epochs"],
                    lr=lr,
                    X_test=X_test_t,
                    y_test=y_test_t,
                )
                acc = model.evaluate(X_test_t, y_test_t)["accuracy"]
                by_topology[topo][rate].append(acc)

    summary_rates = [r for r in (0.0, 0.3) if r in poison_rates]
    summary_at_rates = {}
    for rate in summary_rates:
        for topo in topologies:
            key = f"{topo}_poison_{int(rate * 100)}"
            summary_at_rates[key] = by_topology[topo][rate]
    summarize_multi_seed(EXP_ID, summary_at_rates)

    comparisons = []
    if "quantum_first" in topologies and "hybrid_sandwich" in topologies:
        for rate in summary_rates:
            suffix = int(rate * 100)
            comparisons.append(
                {
                    "label_a": f"quantum_first_poison_{suffix}",
                    "label_b": f"hybrid_sandwich_poison_{suffix}",
                    "condition_a": by_topology["quantum_first"][rate],
                    "condition_b": by_topology["hybrid_sandwich"][rate],
                }
            )
    if "nas_preset" in topologies and "hybrid_sandwich" in topologies:
        comparisons.append(
            {
                "label_a": "nas_preset_poison_30",
                "label_b": "hybrid_sandwich_poison_30",
                "condition_a": by_topology["nas_preset"][0.3],
                "condition_b": by_topology["hybrid_sandwich"][0.3],
            }
        )
    if "classical_first" in topologies and "hybrid_sandwich" in topologies:
        comparisons.append(
            {
                "label_a": "classical_first_poison_30",
                "label_b": "hybrid_sandwich_poison_30",
                "condition_a": by_topology["classical_first"][0.3],
                "condition_b": by_topology["hybrid_sandwich"][0.3],
            }
        )
    if comparisons:
        compare_conditions_batch(EXP_ID, comparisons)

    robustness = {
        topo: measure_robustness({r: sum(accs) / len(accs) for r, accs in rates.items()})
        for topo, rates in by_topology.items()
    }
    log_event("info", "poison topology robustness", exp_id=EXP_ID, topologies=robustness)
    log_event("info", "experiment run finished", exp_id=EXP_ID)
