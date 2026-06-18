"""Run inference from a saved nanomodel checkpoint on raw tabular features."""

from __future__ import annotations

import numpy as np
import torch

from src.application.dto import PredictNanomodelDTO, PredictNanomodelResult
from src.application.model_registry import build_model
from src.application.nanotrainer_config import dataset_kind, load_nanotrainer_config
from src.data.scaling import transform_with_scaler
from src.shared.result import Result, fail, ok
from src.training.checkpoints import load_checkpoint_bundle, load_scaler, resolve_checkpoint_dir
from src.training.device import resolve_device
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import predict


class PredictNanomodelError:
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message


def execute(dto: PredictNanomodelDTO) -> Result[PredictNanomodelResult, PredictNanomodelError]:
    """Load checkpoint + scaler and predict malignancy probability per row."""
    init_correlation_id()

    if not dto.features:
        return fail(PredictNanomodelError("INVALID_FEATURES", "features must not be empty"))

    cfg = load_nanotrainer_config()
    if dataset_kind(cfg, dto.dataset) != "tabular":
        return fail(
            PredictNanomodelError(
                "UNSUPPORTED_DATASET",
                f"prediction supports tabular datasets only (got {dto.dataset})",
            )
        )

    try:
        bundle = load_checkpoint_bundle(
            dto.exp_id,
            dto.model_name,
            dto.dataset,
            seed=dto.seed,
        )
        scaler = load_scaler(bundle.directory)
    except FileNotFoundError as exc:
        return fail(PredictNanomodelError("CHECKPOINT_NOT_FOUND", str(exc)))

    input_dim = int(bundle.config.get("input_dim", len(dto.features[0])))
    if any(len(row) != input_dim for row in dto.features):
        return fail(
            PredictNanomodelError(
                "INVALID_FEATURES",
                f"each row must have {input_dim} features for {dto.dataset}",
            )
        )

    model, _ = build_model(dto.model_name, input_dim=input_dim)
    model.load_state_dict(bundle.state_dict)
    dev = resolve_device(None, model=model)
    model = model.to(dev)

    raw = np.asarray(dto.features, dtype=np.float32)
    scaled = transform_with_scaler(scaler, raw)
    x = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        probs = predict(model, x).cpu().numpy().astype(float).tolist()
    labels = [1 if p >= 0.5 else 0 for p in probs]

    ckpt_path = str(resolve_checkpoint_dir(dto.exp_id, dto.model_name, dto.dataset, seed=dto.seed))
    log_event(
        "info",
        "nanomodel prediction",
        exp_id=dto.exp_id,
        model=dto.model_name,
        dataset=dto.dataset,
        seed=dto.seed,
        n_rows=len(dto.features),
        checkpoint_path=ckpt_path,
        record_source="nanotrainer_predict",
    )

    return ok(
        PredictNanomodelResult(
            exp_id=dto.exp_id,
            model_name=dto.model_name,
            dataset=dto.dataset,
            seed=dto.seed,
            probabilities=probs,
            labels=labels,
            checkpoint_path=ckpt_path,
        )
    )
