"""Nano Parity Bench — prove quantum nanomodel vs parameter-matched classical."""

from __future__ import annotations

import time
from typing import Any

from src.application.dto import NanoParityBenchDTO, NanoParityBenchResult, ParityPairMeta
from src.application.model_registry import build_model
from src.application.parity_config import (
    dataset_config,
    is_tabular_dataset,
    load_parity_config,
    profile_settings,
)
from src.application.parity_datasets import ensure_datasets_available
from src.classical.mlp import ClassicalNet
from src.data.dataset_registry import prepare_dataset
from src.shared.result import Fail, Ok, fail, ok
from src.training.effect_size import cohens_d_magnitude
from src.training.holdout import compare_conditions, summarize_multi_seed, train_with_holdout
from src.training.param_match import (
    build_param_matched_classical,
    classical_n_params,
    nearest_classical_hidden,
)
from src.training.structured_log import init_correlation_id, log_event, set_experiment_context
from src.training.trainer import count_parameters


class NanoParityBenchError:
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message


def build_parity_pair(
    quantum_model_name: str,
    *,
    input_dim: int,
    classical_lr: float = 0.01,
) -> tuple[Any, ClassicalNet, ParityPairMeta]:
    """Build quantum nanomodel and nearest parameter-matched classical MLP."""
    quantum_model, quantum_lr = build_model(quantum_model_name, input_dim=input_dim)
    quantum_n_params = count_parameters(quantum_model)
    classical_hidden = nearest_classical_hidden(quantum_n_params, input_dim)
    classical_model = build_param_matched_classical(quantum_n_params, input_dim)
    classical_n = classical_n_params(classical_hidden, input_dim)
    label = f"classical_matched_h{classical_hidden}"
    meta = ParityPairMeta(
        quantum_model=quantum_model_name,
        classical_label=label,
        quantum_n_params=quantum_n_params,
        classical_n_params=classical_n,
        classical_hidden=classical_hidden,
        param_delta=quantum_n_params - classical_n,
        quantum_learning_rate=quantum_lr,
        classical_learning_rate=classical_lr,
    )
    return quantum_model, classical_model, meta


def _dataset_kwargs(cfg: dict[str, Any], dataset: str) -> dict[str, Any]:
    ds = dataset_config(cfg, dataset)
    kwargs: dict[str, Any] = {}
    for key in ("n_samples", "n_components"):
        if key in ds:
            kwargs[key] = ds[key]
    return kwargs


def _verdict(comparison: dict, *, min_mean_diff_pp: float = 0.0) -> tuple[bool, str]:
    mean_diff_pp = comparison.get("mean_diff", 0.0) * 100.0
    significant = comparison.get("significant_holm")
    if significant is None:
        significant = comparison.get("significant")
    if significant and mean_diff_pp >= min_mean_diff_pp:
        return True, "accepted"
    if mean_diff_pp > 0 and not significant:
        return False, "inconclusive"
    return False, "rejected"


def execute(dto: NanoParityBenchDTO) -> Ok[NanoParityBenchResult] | Fail[NanoParityBenchError]:
    """Run multi-seed parity benchmark: quantum nanomodel vs matched classical."""
    init_correlation_id()
    cfg = load_parity_config()
    prof = profile_settings(cfg, dto.profile)

    if dto.quantum_model not in cfg.get("quantum_models", []):
        return fail(NanoParityBenchError("UNKNOWN_MODEL", f"Unknown quantum model: {dto.quantum_model}"))
    if dto.dataset not in cfg.get("datasets", {}):
        return fail(NanoParityBenchError("INVALID_PAIR", f"Unknown dataset: {dto.dataset}"))
    if not is_tabular_dataset(cfg, dto.dataset):
        return fail(
            NanoParityBenchError(
                "INVALID_PAIR",
                f"Parity bench supports tabular datasets only; got {dto.dataset}",
            )
        )

    seeds = dto.seeds if dto.seeds is not None else list(prof.get("seeds", [42]))
    epochs = dto.epochs if dto.epochs is not None else int(prof.get("epochs", 15))
    classical_lr = dto.classical_learning_rate or float(prof.get("classical_learning_rate", 0.01))
    ds_kwargs = _dataset_kwargs(cfg, dto.dataset)

    datasets_status = ensure_datasets_available([dto.dataset])
    if datasets_status.get(dto.dataset) != "ready":
        return fail(NanoParityBenchError("DATASET_ERROR", f"Dataset not ready: {dto.dataset}"))

    set_experiment_context(experiment_id=dto.exp_id, profile=dto.profile)
    log_event(
        "info",
        "nano parity bench started",
        exp_id=dto.exp_id,
        quantum_model=dto.quantum_model,
        dataset=dto.dataset,
        profile=dto.profile,
        seeds=seeds,
        record_source="nano_parity_bench",
    )

    quantum_accs: list[float] = []
    classical_accs: list[float] = []
    pair_meta: ParityPairMeta | None = None
    t0 = time.perf_counter()

    try:
        for seed in seeds:
            X_train, X_test, y_train, y_test, _meta = prepare_dataset(
                dto.dataset,
                random_state=seed,
                test_size=dto.test_size,
                **ds_kwargs,
            )
            input_dim = int(X_train.shape[1])
            quantum_model, classical_model, pair_meta = build_parity_pair(
                dto.quantum_model,
                input_dim=input_dim,
                classical_lr=classical_lr,
            )
            q_metrics = train_with_holdout(
                quantum_model,
                X_train,
                y_train,
                X_test,
                y_test,
                exp_id=dto.exp_id,
                model_name=f"{dto.quantum_model}_seed{seed}",
                epochs=epochs,
                lr=pair_meta.quantum_learning_rate,
                seed=seed,
                profile=dto.profile,
                save_checkpoints=dto.save_checkpoints,
            )
            c_metrics = train_with_holdout(
                classical_model,
                X_train,
                y_train,
                X_test,
                y_test,
                exp_id=dto.exp_id,
                model_name=f"{pair_meta.classical_label}_seed{seed}",
                epochs=epochs,
                lr=classical_lr,
                seed=seed,
                profile=dto.profile,
                save_checkpoints=dto.save_checkpoints,
            )
            quantum_accs.append(float(q_metrics["accuracy"]))
            classical_accs.append(float(c_metrics["accuracy"]))
    except Exception as exc:
        return fail(NanoParityBenchError("TRAINING_ERROR", str(exc)))

    if pair_meta is None:
        return fail(NanoParityBenchError("TRAINING_ERROR", "No seeds executed"))

    q_summary = summarize_multi_seed(
        dto.exp_id,
        {dto.quantum_model: quantum_accs},
        log_jsonl=True,
    )[dto.quantum_model]
    c_summary = summarize_multi_seed(
        dto.exp_id,
        {pair_meta.classical_label: classical_accs},
        log_jsonl=True,
    )[pair_meta.classical_label]

    comparison = compare_conditions(
        dto.exp_id,
        quantum_accs,
        classical_accs,
        dto.quantum_model,
        pair_meta.classical_label,
        log_jsonl=True,
    )
    comparison["effect_size_magnitude"] = cohens_d_magnitude(comparison.get("effect_size_cohens_d", float("nan")))

    primary = cfg.get("primary_claim", {})
    min_diff_pp = float(primary.get("min_mean_diff_pp", 0.0))
    quantum_wins, verdict = _verdict(comparison, min_mean_diff_pp=min_diff_pp)

    elapsed = time.perf_counter() - t0
    log_event(
        "info",
        "nano parity bench finished",
        exp_id=dto.exp_id,
        quantum_model=dto.quantum_model,
        dataset=dto.dataset,
        quantum_mean=q_summary["mean"],
        classical_mean=c_summary["mean"],
        mean_diff=comparison["mean_diff"],
        verdict=verdict,
        elapsed_s=round(elapsed, 3),
        record_source="nano_parity_bench",
    )

    return ok(
        NanoParityBenchResult(
            exp_id=dto.exp_id,
            quantum_model=dto.quantum_model,
            dataset=dto.dataset,
            profile=dto.profile,
            classical_label=pair_meta.classical_label,
            quantum_n_params=pair_meta.quantum_n_params,
            classical_n_params=pair_meta.classical_n_params,
            classical_hidden=pair_meta.classical_hidden,
            param_delta=pair_meta.param_delta,
            quantum_accuracies=quantum_accs,
            classical_accuracies=classical_accs,
            quantum_mean=float(q_summary["mean"]),
            classical_mean=float(c_summary["mean"]),
            quantum_summary=q_summary,
            classical_summary=c_summary,
            comparison=comparison,
            quantum_wins=quantum_wins,
            verdict=verdict,
            datasets_status=datasets_status,
        )
    )


def run_suite(
    *,
    profile: str = "ci",
    exp_id: str = "exp_022",
) -> list[NanoParityBenchResult]:
    """Run all suite entries from config/nano_parity_bench.yaml."""
    cfg = load_parity_config()
    results: list[NanoParityBenchResult] = []
    for entry in cfg.get("suite", []):
        dto = NanoParityBenchDTO(
            quantum_model=entry["model"],
            dataset=entry["dataset"],
            profile=profile,
            exp_id=exp_id,
        )
        outcome = execute(dto)
        if isinstance(outcome, Fail):
            raise RuntimeError(f"{entry}: {outcome.error.message}")
        results.append(outcome.value)
    return results
