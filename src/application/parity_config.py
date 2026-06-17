"""Load nano_parity_bench.yaml configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "nano_parity_bench.yaml"


def load_parity_config() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def profile_settings(config: dict[str, Any], profile: str) -> dict[str, Any]:
    defaults = config.get("defaults", {})
    prof = config.get("profiles", {}).get(profile, {})
    return {**defaults, **prof, "profile": profile}


def dataset_config(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    ds = config.get("datasets", {}).get(dataset)
    if ds is None:
        raise ValueError(f"Unknown parity dataset: {dataset}")
    return ds


def is_tabular_dataset(config: dict[str, Any], dataset: str) -> bool:
    return dataset_config(config, dataset).get("kind") == "tabular"
