"""Integration smoke test — exp_032 LargeNanoMLP on HIGGS subset."""

from __future__ import annotations

import pytest

from experiments.exp_032_large_nano_higgs.run import run_exp_032

pytestmark = pytest.mark.real


def test_exp_032_ci_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setenv("QML_DEVICE", "cpu")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_exp_032(profile="ci", verbose=False, require_cuda=False)

    assert result.n_train_rows == 50000
    assert result.n_val_rows == 10000
    assert result.n_params >= 1_000_000
    assert 0.0 <= result.logistic_val_auc <= 1.0
    assert 0.0 <= result.nano_val_auc <= 1.0
