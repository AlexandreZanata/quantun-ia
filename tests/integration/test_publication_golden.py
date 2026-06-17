"""Integration smoke — publication-profile golden bounds (fast 2-seed subset)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.publication_smoke import (
    PUBLICATION_SMOKE_EPOCHS,
    PUBLICATION_SMOKE_SEEDS,
    run_exp_011_publication_smoke,
    run_exp_021_publication_smoke,
)

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "regression" / "golden_publication.json"


@pytest.fixture
def golden_bounds() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _assert_mean_in_bounds(
    accs: list[float],
    bounds: dict,
    *,
    label: str,
) -> None:
    mean_acc = sum(accs) / len(accs)
    assert bounds["mean_min"] <= mean_acc <= bounds["mean_max"], (
        f"{label} mean holdout {mean_acc:.4f} outside golden range "
        f"[{bounds['mean_min']}, {bounds['mean_max']}]"
    )
    for acc in accs:
        assert bounds["per_seed_min"] <= acc <= bounds["per_seed_max"], (
            f"{label} per-seed {acc:.4f} outside "
            f"[{bounds['per_seed_min']}, {bounds['per_seed_max']}]"
        )


def test_golden_publication_metadata_matches_smoke_config(golden_bounds):
    assert golden_bounds["profile"] == "publication"
    assert golden_bounds["smoke_seeds"] == PUBLICATION_SMOKE_SEEDS
    assert golden_bounds["smoke_epochs"] == PUBLICATION_SMOKE_EPOCHS


def test_exp_011_publication_smoke_in_golden_range(tmp_path, monkeypatch, golden_bounds):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    log_file = tmp_path / "experiments.jsonl"

    results = run_exp_011_publication_smoke(log_path=log_file)

    assert "perceptron" in results
    accs = results["perceptron"]
    assert len(accs) == len(PUBLICATION_SMOKE_SEEDS)
    _assert_mean_in_bounds(accs, golden_bounds["exp_011"]["perceptron"], label="exp_011.perceptron")
    assert log_file.exists()


def test_exp_021_publication_smoke_in_golden_range(tmp_path, monkeypatch, golden_bounds):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    log_file = tmp_path / "experiments.jsonl"

    results = run_exp_021_publication_smoke(log_path=log_file)

    for model_name, accs in results.items():
        bounds = golden_bounds["exp_021"].get(model_name)
        if bounds is None:
            continue
        assert len(accs) == len(PUBLICATION_SMOKE_SEEDS)
        _assert_mean_in_bounds(accs, bounds, label=f"exp_021.{model_name}")

    assert log_file.exists()
