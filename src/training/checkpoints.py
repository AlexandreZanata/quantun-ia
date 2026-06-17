"""Model checkpoint persistence for reproducible experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ARTIFACTS_ROOT = Path("artifacts")


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
