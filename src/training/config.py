"""Load experiment configuration from config/experiments.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "experiments.yaml"


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_experiment_config(exp_key: str) -> dict[str, Any]:
    """Merge defaults with experiment-specific settings."""
    cfg = load_config()
    defaults = cfg.get("defaults", {})
    exp_cfg = cfg.get("experiments", {}).get(exp_key, {})
    return {**defaults, **exp_cfg}
