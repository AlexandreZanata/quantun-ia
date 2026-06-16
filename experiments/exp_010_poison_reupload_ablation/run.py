"""
EXP 010 — Re-upload poisoning ablation (layers / learning rate).
Compares reupload_3l (exp_004 baseline) vs 2 layers and lower LR.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.data.generators import make_binary_classification
from src.data.poisoning import measure_robustness, poison_dataset
from src.data.splits import split_train_test
from src.quantum.qnn_factory import build_qnn
from src.training.config import load_experiment_config
from src.training.holdout import compare_conditions_batch, summarize_multi_seed
from src.training.protocol import log_experiment_protocol
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_010_poison_reupload_ablation"
EXP_ID = "exp_010"


def build_variant(variant: str, cfg: dict):
    model_cfg = cfg.get("model_configs", {}).get(variant, {})
    lr = model_cfg.get("learning_rate", cfg["learning_rate"])
    qcfg = {**cfg, **model_cfg, "qnn_type": "reupload"}
    return build_qnn(qcfg), lr


if __name__ == "__main__":
    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY)
    seeds = cfg.get("seeds", [cfg["random_state"]])
    variants = cfg["reupload_variants"]
    log_experiment_protocol(EXP_ID, cfg)
    log_event("info", "experiment run started", exp_id=EXP_ID, seeds=seeds, variants=variants)

    by_variant: dict[str, dict[float, list[float]]] = {
        v: {r: [] for r in cfg["poison_rates"]} for v in variants
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

        for rate in cfg["poison_rates"]:
            _, y_train_poisoned, _ = poison_dataset(X_train, y_train, poison_rate=rate)
            X_train_t = torch.tensor(X_train)
            y_train_t = torch.tensor(y_train_poisoned)

            for variant in variants:
                model, lr = build_variant(variant, cfg)
                model.train(
                    X_train_t,
                    y_train_t,
                    exp_id=EXP_ID,
                    model_name=f"{variant}_poison_{int(rate * 100)}_seed{seed}",
                    epochs=cfg["epochs"],
                    lr=lr,
                    X_test=X_test_t,
                    y_test=y_test_t,
                )
                acc = model.evaluate(X_test_t, y_test_t)["accuracy"]
                by_variant[variant][rate].append(acc)

    summary_at_rates = {}
    for rate in [0.0, 0.3]:
        for variant in variants:
            key = f"{variant}_poison_{int(rate * 100)}"
            summary_at_rates[key] = by_variant[variant][rate]
    summarize_multi_seed(EXP_ID, summary_at_rates)

    baseline = cfg.get("baseline_variant", "reupload_3l")
    compare_conditions_batch(
        EXP_ID,
        [
            {
                "label_a": f"{v}_poison_0",
                "label_b": f"{baseline}_poison_0",
                "condition_a": by_variant[v][0.0],
                "condition_b": by_variant[baseline][0.0],
            }
            for v in variants
            if v != baseline
        ]
        + [
            {
                "label_a": f"{v}_poison_30",
                "label_b": f"{baseline}_poison_30",
                "condition_a": by_variant[v][0.3],
                "condition_b": by_variant[baseline][0.3],
            }
            for v in variants
            if v != baseline
        ],
    )

    robustness = {
        v: measure_robustness({r: sum(accs) / len(accs) for r, accs in rates.items()})
        for v, rates in by_variant.items()
    }
    log_event("info", "poison ablation robustness", exp_id=EXP_ID, variants=robustness)
    log_event("info", "experiment run finished", exp_id=EXP_ID)
