"""Model checkpoint persistence for reproducible experiment artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ARTIFACTS_ROOT = Path("artifacts")
SCALER_FILENAME = "scaler.joblib"


@dataclass(frozen=True)
class CheckpointBundle:
    directory: Path
    config: dict[str, Any]
    metadata: dict[str, Any]
    state_dict: dict[str, Any]


def checkpoint_path(exp_id: str, model_name: str, seed: int | None) -> Path:
    seed_label = f"seed_{seed}" if seed is not None else "seed_unknown"
    safe_name = model_name.replace("/", "_")
    return ARTIFACTS_ROOT / exp_id / safe_name / seed_label


def save_checkpoint(
    model: nn.Module,
    directory: Path,
    *,
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    """Persist model weights and run metadata."""
    directory.mkdir(parents=True, exist_ok=True)
    weights_path = directory / "best.pt"
    config_path = directory / "config.json"
    torch.save(model.state_dict(), weights_path)
    with open(config_path, "w") as f:
        json.dump({"config": config, "metadata": metadata}, f, indent=2)
    return weights_path


def save_best_checkpoint(
    model: nn.Module,
    exp_id: str,
    model_name: str,
    seed: int | None,
    metric_value: float,
    *,
    best_metric: float | None,
    higher_is_better: bool = True,
    config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[float, Path | None]:
    """
    Save checkpoint when metric improves.

    Returns (new_best_metric, path_or_none).
    """
    improved = best_metric is None
    if best_metric is not None:
        improved = metric_value > best_metric if higher_is_better else metric_value < best_metric

    if not improved:
        return best_metric, None

    directory = checkpoint_path(exp_id, model_name, seed)
    path = save_checkpoint(
        model,
        directory,
        config=config or {},
        metadata=metadata or {"metric_value": metric_value},
    )
    return metric_value, path


def save_scaler(scaler: Any, directory: Path) -> Path:
    """Persist sklearn StandardScaler for inference on raw features."""
    import joblib

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SCALER_FILENAME
    joblib.dump(scaler, path)
    return path


def load_scaler(directory: Path) -> Any:
    import joblib

    path = directory / SCALER_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"scaler not found: {path}")
    return joblib.load(path)


def resolve_checkpoint_dir(
    exp_id: str,
    model_name: str,
    dataset: str,
    *,
    seed: int,
) -> Path:
    """Return checkpoint directory for nanotrainer naming convention."""
    return checkpoint_path(exp_id, f"{model_name}_{dataset}", seed)


def load_checkpoint_bundle(
    exp_id: str,
    model_name: str,
    dataset: str,
    *,
    seed: int,
) -> CheckpointBundle:
    """Load weights and metadata from a saved nanotrainer checkpoint."""
    directory = resolve_checkpoint_dir(exp_id, model_name, dataset, seed=seed)
    weights_path = directory / "best.pt"
    config_path = directory / "config.json"
    if not weights_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"checkpoint not found under {directory}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    return CheckpointBundle(
        directory=directory,
        config=payload.get("config", {}),
        metadata=payload.get("metadata", {}),
        state_dict=state_dict,
    )
