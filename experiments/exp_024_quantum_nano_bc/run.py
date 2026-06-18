"""
EXP 024 — QuantumNano-BC flagship nano model on Wisconsin Breast Cancer.

Compares hybrid_sandwich against logistic regression, shallow XGBoost,
perceptron, and parameter-matched classical MLP on the full UCI dataset.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.classical.logistic_baseline import LogisticBaseline
from src.classical.perceptron import Perceptron
from src.classical.xgboost_baseline import XGBoostShallow
from src.data.dataset_registry import prepare_dataset
from src.quantum.hybrid_model import HybridSandwich
from src.training.config import load_experiment_config
from src.training.holdout import compare_conditions_batch, summarize_multi_seed, train_with_holdout
from src.training.param_match import build_param_matched_classical
from src.training.protocol import log_experiment_protocol
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters

EXP_KEY = "exp_024_quantum_nano_bc"
EXP_ID = "exp_024"
EXP_DIR = Path(__file__).resolve().parent


def _resolve_profile(cfg: dict, cli_profile: str | None) -> str:
    if cli_profile:
        return cli_profile
    env_profile = os.environ.get("QML_PROFILE")
    if env_profile:
        return env_profile
    return cfg.get("profile", "ci")


def build_models(input_dim: int, cfg: dict) -> dict[str, tuple]:
    mc = cfg.get("model_configs", {})
    n_qubits = cfg.get("n_qubits", 4)
    n_layers = cfg.get("n_layers", 2)
    hybrid = HybridSandwich(
        input_dim=input_dim,
        n_qubits=n_qubits,
        n_layers=n_layers,
        reupload=bool(mc.get("hybrid_sandwich", {}).get("reupload", True)),
    )
    target_params = count_parameters(hybrid)
    matched = build_param_matched_classical(target_params, input_dim=input_dim)
    hidden = matched.net[0].out_features

    models: dict[str, tuple] = {
        "logistic_regression": (
            LogisticBaseline(input_dim=input_dim),
            mc.get("logistic_regression", {}).get("learning_rate", 1.0),
        ),
        "perceptron": (
            Perceptron(input_dim=input_dim),
            mc.get("perceptron", {}).get("learning_rate", cfg.get("learning_rate", 0.02)),
        ),
        f"classical_matched_h{hidden}": (
            matched,
            mc.get("classical_matched", {}).get("learning_rate", cfg.get("learning_rate", 0.01)),
        ),
        "hybrid_sandwich": (
            hybrid,
            mc.get("hybrid_sandwich", {}).get("learning_rate", cfg.get("learning_rate", 0.02)),
        ),
    }

    if cfg.get("include_xgboost", True):
        models["xgboost_shallow"] = (
            XGBoostShallow(input_dim=input_dim),
            mc.get("xgboost_shallow", {}).get("learning_rate", 0.1),
        )

    return models


def _paired_comparisons(results: dict[str, list[float]]) -> list[dict]:
    classical_key = next((k for k in results if k.startswith("classical_matched")), None)
    specs = [
        ("hybrid_sandwich", "logistic_regression"),
        ("hybrid_sandwich", "xgboost_shallow"),
        ("hybrid_sandwich", classical_key),
        ("logistic_regression", "perceptron"),
    ]
    comparisons: list[dict] = []
    for label_a, label_b in specs:
        if not label_b or label_a not in results or label_b not in results:
            continue
        a_vals, b_vals = results[label_a], results[label_b]
        if not a_vals or not b_vals or len(a_vals) != len(b_vals):
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
    parser = argparse.ArgumentParser(description="Run exp_024 QuantumNano-BC flagship benchmark")
    parser.add_argument("--profile", default=None, help="Profile override (ci, publication)")
    parser.add_argument(
        "--write-results",
        action="store_true",
        help="Write experiments/exp_024_quantum_nano_bc/results.md after run",
    )
    parser.add_argument(
        "--write-model-card",
        action="store_true",
        help="Generate model_cards/quantum_nano_bc.md after run",
    )
    args = parser.parse_args()

    init_correlation_id()
    base_cfg = load_experiment_config(EXP_KEY)
    profile = _resolve_profile(base_cfg, args.profile)
    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seeds = cfg.get("seeds", [cfg["random_state"]])
    dataset = cfg.get("dataset", "breast_cancer")
    save_ckpt = bool(cfg.get("save_checkpoints", profile == "publication"))

    log_experiment_protocol(EXP_ID, cfg)
    log_event(
        "info",
        "experiment run started",
        exp_id=EXP_ID,
        dataset=dataset,
        profile=profile,
        seeds=seeds,
        n_seeds=len(seeds),
    )

    probe = prepare_dataset(dataset, random_state=seeds[0], test_size=cfg["test_size"])
    input_dim = int(probe[0].shape[1])
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
                profile=profile,
                save_checkpoints=save_ckpt,
            )
            results_by_model[name].append(metrics["accuracy"])

    summarize_multi_seed(EXP_ID, results_by_model)
    comparisons = _paired_comparisons(results_by_model)
    if comparisons:
        compare_conditions_batch(EXP_ID, comparisons)

    hybrid_mean = sum(results_by_model["hybrid_sandwich"]) / len(results_by_model["hybrid_sandwich"])
    logistic_mean = sum(results_by_model["logistic_regression"]) / len(
        results_by_model["logistic_regression"]
    )
    diff_pp = (hybrid_mean - logistic_mean) * 100
    log_event(
        "info",
        "experiment run finished",
        exp_id=EXP_ID,
        hybrid_mean=hybrid_mean,
        logistic_mean=logistic_mean,
        diff_pp=diff_pp,
    )
    print(
        f"exp_024 complete — hybrid={hybrid_mean * 100:.1f}% "
        f"logistic={logistic_mean * 100:.1f}% (Δ={diff_pp:+.1f} pp)"
    )

    if args.write_results or profile == "publication":
        from src.training.results_writer import write_results_md

        path = write_results_md(
            EXP_ID,
            EXP_DIR,
            exp_title="EXP 024 (QuantumNano-BC)",
            dataset_note="breast_cancer (UCI Wisconsin), full 569 samples, 30% holdout",
            conclusion_hint=(
                "Flagship QuantumNano-BC benchmark: hybrid sandwich vs clinical baselines "
                "on full Wisconsin Breast Cancer. See model_cards/quantum_nano_bc.md."
            ),
        )
        print(f"Wrote {path}")

    if args.write_model_card or profile == "publication":
        from scripts.generate_model_card import write_quantum_nano_bc_card

        card_path = write_quantum_nano_bc_card()
        print(f"Wrote {card_path}")
