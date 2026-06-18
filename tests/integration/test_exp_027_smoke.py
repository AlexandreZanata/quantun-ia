"""Integration smoke test — exp_027 continuous retrain gate under CI profile."""

from __future__ import annotations

import json

import pytest

from experiments.exp_027_continuous_retrain.run import run_exp_027


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", artifacts)
    log_file = tmp_path / "experiments.jsonl"
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", log_file)
    return artifacts, log_file


def test_exp_027_ci_continuous_retrain_smoke(isolated_artifacts, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    artifacts, log_file = isolated_artifacts

    results = run_exp_027(profile="ci", verbose=False, require_cuda=False, fresh_champion=True)

    assert len(results) == 4
    assert all(0.5 <= r.challenger_accuracy <= 1.0 for r in results)
    assert (artifacts / "champion" / "manifest.json").is_file()
    assert (artifacts / "champion" / "checkpoint").exists()

    lines = [json.loads(line) for line in log_file.read_text().strip().splitlines()]
    exp_ids = {rec.get("exp_id") for rec in lines}
    assert "exp_027" in exp_ids or "quantum_nano_bc_app" in exp_ids
