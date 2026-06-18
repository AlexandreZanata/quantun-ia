"""Integration smoke test — exp_031 clinical curriculum ablation."""

from __future__ import annotations

from experiments.exp_031_curriculum_clinical.run import run_exp_031


def test_exp_031_ci_curriculum_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setenv("QML_DEVICE", "cpu")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_exp_031(profile="ci", verbose=False, require_cuda=False)

    assert result.n_seeds == 3
    assert result.applicable
    assert len(result.random_accuracies) == 3
    assert len(result.curriculum_accuracies) == 3
    assert all(0.5 <= a <= 1.0 for a in result.random_accuracies)
    assert all(0.5 <= a <= 1.0 for a in result.curriculum_accuracies)
    assert result.advantage_pp > result.min_advantage_pp
