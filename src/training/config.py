"""Load experiment configuration from config/experiments.yaml."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "experiments.yaml"


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_experiment_config(exp_key: str, profile: str | None = None) -> dict[str, Any]:
    """Merge defaults, optional profile, and experiment-specific settings."""
    cfg = load_config()
    defaults = cfg.get("defaults", {})
    profile_name = profile or os.environ.get("QML_PROFILE") or defaults.get("profile", "publication")
    profile_cfg = cfg.get("profiles", {}).get(profile_name, {})
    exp_cfg = cfg.get("experiments", {}).get(exp_key, {})
    overrides = (exp_cfg.get("profile_overrides") or {}).get(profile_name, {})
    exp_cfg_clean = {k: v for k, v in exp_cfg.items() if k != "profile_overrides"}
    merged = {**defaults, **profile_cfg, **exp_cfg_clean, **overrides}
    merged["profile"] = profile_name
    return merged
