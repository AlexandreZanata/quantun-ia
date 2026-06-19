"""Holdout training and multi-seed evaluation helpers."""

from __future__ import annotations

import json
from datetime import datetime

import torch

from src.training import metrics as metrics_module
from src.training.adaptive_lr import AdaptiveLRConfig, train_model_adaptive
from src.training.statistics import holm_bonferroni, paired_comparison, seed_summary
from src.training.structured_log import log_event
from src.training.trainer import evaluate as evaluate_model
from src.training.trainer import train_model


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _align_eval_tensors(model, X_test_t: torch.Tensor, y_test_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dev = _model_device(model)
    return X_test_t.to(dev), y_test_t.to(dev)


def _is_sklearn_model(model) -> bool:
    from src.classical.sklearn_wrapper import SklearnBinaryClassifier

    return isinstance(model, SklearnBinaryClassifier)


def train_with_holdout(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    exp_id: str,
    model_name: str,
    epochs: int = 50,
    lr: float = 0.01,
    seed: int | None = None,
    profile: str | None = None,
    save_checkpoints: bool = True,
) -> dict:
    """Train on train split; log and return metrics on held-out test split."""
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)

    if _is_sklearn_model(model):
        model.train(
            X_train_t,
            y_train_t,
            exp_id=exp_id,
            model_name=model_name,
            epochs=epochs,
            lr=lr,
            X_test=X_test_t,
            y_test=y_test_t,
            seed=seed,
            profile=profile,
            save_checkpoints=save_checkpoints,
        )
        return evaluate_model(model, *_align_eval_tensors(model, X_test_t, y_test_t))

    train_model(
        model,
        X_train_t,
        y_train_t,
        exp_id,
        model_name,
        epochs=epochs,
        lr=lr,
        X_test=X_test_t,
        y_test=y_test_t,
        seed=seed,
        profile=profile,
        save_checkpoints=save_checkpoints,
    )
    X_eval, y_eval = _align_eval_tensors(model, X_test_t, y_test_t)
    return evaluate_model(model, X_eval, y_eval)


def train_with_holdout_adaptive(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    exp_id: str,
    model_name: str,
    epochs: int = 50,
    adaptive_config: AdaptiveLRConfig | None = None,
    seed: int | None = None,
    profile: str | None = None,
    save_checkpoints: bool = True,
) -> dict:
    """Train with gradient-variance adaptive LR; return holdout metrics."""
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)

    train_model_adaptive(
        model,
        X_train_t,
        y_train_t,
        exp_id,
        model_name,
        epochs=epochs,
        config=adaptive_config,
        X_test=X_test_t,
        y_test=y_test_t,
        seed=seed,
        profile=profile,
        save_checkpoints=save_checkpoints,
    )
    X_eval, y_eval = _align_eval_tensors(model, X_test_t, y_test_t)
    return evaluate_model(model, X_eval, y_eval)


def _write_summary_record(record: dict) -> None:
    log_path = metrics_module.LOGS_PATH
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def summarize_multi_seed(
    exp_id: str,
    results_by_model: dict[str, list[float]],
    *,
    log_jsonl: bool = True,
) -> dict[str, dict]:
    """Log mean ± std, bootstrap CI, and optional JSONL research summary."""
    summary: dict[str, dict] = {}
    for name, accs in results_by_model.items():
        stats = seed_summary(accs)
        summary[name] = stats
        log_event(
            "info",
            "multi-seed summary",
            exp_id=exp_id,
            model=name,
            mean_holdout_acc=stats["mean"],
            std_holdout_acc=stats["std"],
            ci_low=stats["ci_low"],
            ci_high=stats["ci_high"],
            n_seeds=stats["n_seeds"],
        )

    if log_jsonl and summary:
        _write_summary_record(
            {
                "exp_id": exp_id,
                "model_name": f"{exp_id}_multi_seed_summary",
                "record_type": "multi_seed_summary",
                "started_at": datetime.now().isoformat(),
                "summary": summary,
            }
        )
    return summary


def compare_conditions(
    exp_id: str,
    condition_a: list[float],
    condition_b: list[float],
    label_a: str,
    label_b: str,
    *,
    log_jsonl: bool = True,
) -> dict:
    """Paired statistical comparison between two seed-aligned conditions."""
    comparison = paired_comparison(condition_a, condition_b)
    comparison.update({"label_a": label_a, "label_b": label_b})
    log_event(
        "info",
        "paired comparison",
        exp_id=exp_id,
        label_a=label_a,
        label_b=label_b,
        mean_diff=comparison["mean_diff"],
        p_value=comparison["p_value"],
        significant=comparison["significant"],
    )
    if log_jsonl:
        _write_summary_record(
            {
                "exp_id": exp_id,
                "model_name": f"{exp_id}_paired_{label_a}_vs_{label_b}",
                "record_type": "paired_comparison",
                "started_at": datetime.now().isoformat(),
                "comparison": comparison,
            }
        )
    return comparison


def compare_conditions_batch(
    exp_id: str,
    comparisons: list[dict],
    *,
    alpha: float = 0.05,
    log_jsonl: bool = True,
) -> list[dict]:
    """
    Run multiple paired Wilcoxon tests and apply Holm-Bonferroni correction.
    Each item: {label_a, label_b, condition_a, condition_b}.
    """
    results: list[dict] = []
    p_values: list[float | None] = []

    for spec in comparisons:
        comp = paired_comparison(spec["condition_a"], spec["condition_b"], alpha=alpha)
        comp.update({"label_a": spec["label_a"], "label_b": spec["label_b"]})
        results.append(comp)
        p_values.append(comp.get("p_value"))
        log_event(
            "info",
            "effect size",
            exp_id=exp_id,
            label_a=spec["label_a"],
            label_b=spec["label_b"],
            cohens_d=comp.get("effect_size_cohens_d"),
        )

    holm = holm_bonferroni([p if p is not None else 1.0 for p in p_values], alpha=alpha)
    for comp, adj in zip(results, holm):
        comp["p_value_holm"] = adj["p_adjusted"]
        comp["significant_holm"] = adj["significant_holm"]
        log_event(
            "info",
            "paired comparison",
            exp_id=exp_id,
            label_a=comp["label_a"],
            label_b=comp["label_b"],
            mean_diff=comp["mean_diff"],
            p_value=comp["p_value"],
            p_value_holm=comp["p_value_holm"],
            significant=comp["significant"],
            significant_holm=comp["significant_holm"],
        )

    if log_jsonl and results:
        _write_summary_record(
            {
                "exp_id": exp_id,
                "model_name": f"{exp_id}_paired_comparison_batch",
                "record_type": "paired_comparison_batch",
                "started_at": datetime.now().isoformat(),
                "comparisons": results,
                "correction": "holm_bonferroni",
                "alpha": alpha,
            }
        )
    return results
