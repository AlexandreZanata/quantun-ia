"""Load and validate shippable nanomodel entries from config/nanomodel_registry.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "config" / "nanomodel_registry.yaml"
DIST_SERVE_ROOT = Path("dist") / "serve_models"


@dataclass(frozen=True)
class NanomodelSpec:
    registry_key: str
    description: str
    train_model: str
    dataset: str
    exp_id: str
    seed: int
    profile: str
    serve_kind: str
    train_kind: str
    experiment_key: str | None
    gate_test: str | None
    calibration_exp_id: str | None
    exports: tuple[str, ...]
    model_card: str
    depends_on: str | None

    @property
    def bundle_dir(self) -> Path:
        return DIST_SERVE_ROOT / self.registry_key


def _as_str_list(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(str(v) for v in value)


def _parse_entry(registry_key: str, raw: dict[str, Any], defaults: dict[str, Any]) -> NanomodelSpec:
    serve = dict(raw.get("serve") or {})
    train = dict(raw.get("train") or {})
    return NanomodelSpec(
        registry_key=registry_key,
        description=str(raw.get("description", registry_key)),
        train_model=str(raw["train_model"]),
        dataset=str(raw["dataset"]),
        exp_id=str(raw.get("exp_id", registry_key)),
        seed=int(raw.get("seed", defaults.get("seed", 42))),
        profile=str(raw.get("profile", defaults.get("profile", "publication"))),
        serve_kind=str(serve.get("kind", "nanotrainer")),
        train_kind=str(train.get("kind", "nanotrainer")),
        experiment_key=str(train["experiment_key"]) if train.get("experiment_key") else None,
        gate_test=str(raw["gate_test"]) if raw.get("gate_test") else None,
        calibration_exp_id=str(raw["calibration_exp_id"]) if raw.get("calibration_exp_id") else None,
        exports=_as_str_list(raw.get("exports")),
        model_card=str(raw.get("model_card", f"model_cards/{registry_key}.md")),
        depends_on=str(raw["depends_on"]) if raw.get("depends_on") else None,
    )


@lru_cache(maxsize=1)
def _load_registry_cached() -> dict[str, NanomodelSpec]:
    return _parse_registry_file(REGISTRY_PATH)


def _parse_registry_file(registry_file: Path) -> dict[str, NanomodelSpec]:
    payload = yaml.safe_load(registry_file.read_text(encoding="utf-8"))
    defaults = dict(payload.get("defaults") or {})
    models = dict(payload.get("models") or {})
    return {
        key: _parse_entry(key, dict(value), defaults)
        for key, value in models.items()
    }


def load_registry(path: Path | None = None) -> dict[str, NanomodelSpec]:
    """Return all registry entries keyed by registry_key."""
    if path is None:
        return _load_registry_cached()
    return _parse_registry_file(path)


def get_nanomodel_spec(registry_key: str, *, path: Path | None = None) -> NanomodelSpec:
    registry = load_registry(path)
    if registry_key not in registry:
        known = ", ".join(sorted(registry))
        msg = f"unknown nanomodel registry key: {registry_key} (known: {known})"
        raise KeyError(msg)
    return registry[registry_key]


def list_registry_keys(*, path: Path | None = None) -> list[str]:
    return sorted(load_registry(path))
