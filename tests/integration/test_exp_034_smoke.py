"""Integration smoke test — exp_034 LargeNanoMLP on Synthea subset."""

from __future__ import annotations

from experiments.exp_034_large_nano_synthea.run import run_exp_034


def test_exp_034_ci_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setenv("QML_DEVICE", "cpu")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_exp_034(profile="ci", verbose=False, require_cuda=False)

    assert result.n_train_rows == 50000
    assert result.n_val_rows == 10000
    assert result.n_params >= 1_000_000
    assert 0.0 <= result.logistic_val_auc <= 1.0
    assert 0.0 <= result.nano_val_auc <= 1.0
