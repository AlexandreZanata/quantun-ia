"""Unit tests for publication smoke helpers."""

from __future__ import annotations

import json
from pathlib import Path

from src.training.publication_smoke import (
    PUBLICATION_SMOKE_EPOCHS,
    PUBLICATION_SMOKE_SEEDS,
    run_exp_011_publication_smoke,
)


def test_publication_smoke_constants():
    assert len(PUBLICATION_SMOKE_SEEDS) == 2
    assert PUBLICATION_SMOKE_EPOCHS >= 10


def test_run_exp_011_publication_smoke_returns_perceptron_accuracies(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    log_file = tmp_path / "experiments.jsonl"

    results = run_exp_011_publication_smoke(log_path=log_file)

    assert "perceptron" in results
    assert len(results["perceptron"]) == 2
    assert all(0.0 <= acc <= 1.0 for acc in results["perceptron"])
    assert log_file.exists()


def test_golden_publication_json_loads():
    path = Path(__file__).resolve().parents[1] / "regression" / "golden_publication.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["profile"] == "publication"
    assert "exp_011" in data
    assert "exp_021" in data
