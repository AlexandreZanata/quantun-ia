"""Train a single nanomodel on a real dataset — Nano Trainer use case."""

from __future__ import annotations

import time
from typing import Any

from src.application.dto import TrainNanomodelDTO, TrainNanomodelResult
from src.application.model_registry import build_model, validate_model_dataset_pair
from src.application.nanotrainer_config import (
    dataset_kind,
    load_nanotrainer_config,
    profile_settings,
)
from src.data.dataset_registry import prepare_dataset
from src.shared.result import Result, fail, ok
from src.training.holdout import train_with_holdout
from src.training.structured_log import init_correlation_id, log_event, set_experiment_context
from src.training.trainer import count_parameters


class TrainNanomodelError:
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message


def _prepare_data(
    dataset: str,
    ds_cfg: dict[str, Any],
    *,
    seed: int,
    test_size: float,
    n_samples: int | None,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    kwargs: dict[str, Any] = {"random_state": seed, "test_size": test_size}
    if n_samples is not None:
        kwargs["n_samples"] = n_samples
    for key in ("seq_len", "input_dim", "noise", "n_components"):
        if key in ds_cfg:
            kwargs[key] = ds_cfg[key]
    return prepare_dataset(dataset, **kwargs)


def execute(dto: TrainNanomodelDTO) -> Result[TrainNanomodelResult, TrainNanomodelError]:
    """Run one mini training session and append metrics to experiments.jsonl."""
    init_correlation_id()
    cfg = load_nanotrainer_config()

    try:
        validate_model_dataset_pair(dto.model_name, dto.dataset)
    except ValueError as exc:
        return fail(TrainNanomodelError("INVALID_PAIR", str(exc)))

    prof = profile_settings(cfg, dto.profile)
    seed = dto.seed if dto.seed is not None else int(prof.get("seed", 42))
    epochs = dto.epochs if dto.epochs is not None else int(prof.get("epochs", 8))
    n_samples = prof.get("n_samples")
    save_checkpoints = dto.save_checkpoints or bool(prof.get("save_checkpoints", False))

    ds_cfg = cfg.get("datasets", {}).get(dto.dataset, {})
    kind = dataset_kind(cfg, dto.dataset)

    set_experiment_context(experiment_id=dto.exp_id, seed=seed, profile=dto.profile)
    log_event(
        "info",
        "nano trainer started",
        exp_id=dto.exp_id,
        model=dto.model_name,
        dataset=dto.dataset,
        profile=dto.profile,
        record_source="nanotrainer",
    )

    try:
        X_train, X_test, y_train, y_test, meta = _prepare_data(
            dto.dataset,
            ds_cfg,
            seed=seed,
            test_size=dto.test_size,
            n_samples=n_samples,
        )
    except Exception as exc:
        return fail(TrainNanomodelError("DATASET_ERROR", str(exc)))

    if kind == "sequence":
        input_dim_seq = int(meta.get("input_dim", ds_cfg.get("input_dim", 2)))
        seq_len = int(meta.get("seq_len", ds_cfg.get("seq_len", 8)))
        model, lr = build_model(
            dto.model_name,
            input_dim=0,
            seq_len=seq_len,
            input_dim_seq=input_dim_seq,
        )
    else:
        input_dim = int(X_train.shape[1])
        model, lr = build_model(dto.model_name, input_dim=input_dim)

    t0 = time.perf_counter()
    metrics = train_with_holdout(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        exp_id=dto.exp_id,
        model_name=f"{dto.model_name}_{dto.dataset}",
        epochs=epochs,
        lr=lr,
        seed=seed,
        profile=dto.profile,
        save_checkpoints=save_checkpoints,
    )
    elapsed = time.perf_counter() - t0

    result = TrainNanomodelResult(
        exp_id=dto.exp_id,
        model_name=dto.model_name,
        dataset=dto.dataset,
        profile=dto.profile,
        seed=seed,
        accuracy=float(metrics["accuracy"]),
        loss=float(metrics["loss"]),
        elapsed_s=round(elapsed, 3),
        n_params=count_parameters(model),
        n_epochs=epochs,
    )

    log_event(
        "info",
        "nano trainer finished",
        exp_id=dto.exp_id,
        model=dto.model_name,
        dataset=dto.dataset,
        test_accuracy=result.accuracy,
        elapsed_s=result.elapsed_s,
        record_source="nanotrainer",
    )
    return ok(result)
