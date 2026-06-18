"""Real gate — exp_032 LargeNanoMLP beats logistic on HIGGS val AUC (RTX 4060)."""

from __future__ import annotations

import pytest

from experiments.exp_032_large_nano_higgs.run import run_exp_032

pytestmark = pytest.mark.real


def test_exp_032_large_nano_higgs_publication_gate(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_032 real gate")

    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    result = run_exp_032(profile="publication", verbose=False, require_cuda=True)

    assert result.n_params >= 1_000_000
    assert result.n_train_rows == 805_000
    assert result.auc_advantage_pp >= result.min_auc_advantage_pp
