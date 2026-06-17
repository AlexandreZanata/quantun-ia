"""Load nanotrainer.yaml configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "nanotrainer.yaml"


def load_nanotrainer_config() -> dict[str, Any]:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def profile_settings(config: dict[str, Any], profile: str) -> dict[str, Any]:
    defaults = config.get("defaults", {})
    prof = config.get("profiles", {}).get(profile, {})
    return {**defaults, **prof, "profile": profile}


def is_allowed_pair(config: dict[str, Any], model_name: str, dataset: str) -> bool:
    pairs = config.get("pairs", [])
    return any(p.get("model") == model_name and p.get("dataset") == dataset for p in pairs)


def dataset_kind(config: dict[str, Any], dataset: str) -> str:
    ds = config.get("datasets", {}).get(dataset)
    if ds is None:
        raise ValueError(f"Unknown dataset: {dataset}")
    return str(ds.get("kind", "tabular"))


def model_kind(config: dict[str, Any], model_name: str) -> str:
    mc = config.get("models", {}).get(model_name)
    if mc is None:
        raise ValueError(f"Unknown model: {model_name}")
    return str(mc.get("kind", "tabular"))
