"""Unit tests for nanomodel registry configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.application.nanomodel_registry import get_nanomodel_spec, load_registry


def test_registry_loads_p0_models():
    keys = load_registry()
    assert "large_nano_mlp_synthea" in keys
    assert "large_nano_mlp_higgs" in keys
    assert "quantum_nano_bc" in keys
    assert "residual_nano_distill_acyd_maize" in keys


def test_residual_distill_maize_registry_entry():
    spec = get_nanomodel_spec("residual_nano_distill_acyd_maize")
    assert spec.train_model == "residual_nano_distill"
    assert spec.dataset == "acyd_maize_brazil_v1"
    assert spec.exp_id == "exp_092"
    assert spec.serve_kind == "open_residual_nano"
    assert spec.train_kind == "none"


def test_registry_entry_fields():
    spec = get_nanomodel_spec("large_nano_mlp_synthea")
    assert spec.train_model == "large_nano_mlp"
    assert spec.dataset == "synthea_cv_risk_v1"
    assert spec.exp_id == "exp_034"
    assert "onnx" in spec.exports


def test_unknown_registry_key_raises():
    with pytest.raises(KeyError, match="unknown nanomodel"):
        get_nanomodel_spec("not_a_real_model")


def test_registry_contract_required_fields():
    registry = load_registry()
    for key, spec in registry.items():
        assert spec.train_model
        assert spec.dataset
        assert spec.exp_id
        assert spec.serve_kind in {
            "open_large_nano",
            "open_hybrid",
            "open_residual_nano",
            "nanotrainer",
        }
        assert spec.bundle_dir.name == key


def test_registry_yaml_parseable():
    path = Path("config/nanomodel_registry.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "models" in payload
    assert len(payload["models"]) >= 3
